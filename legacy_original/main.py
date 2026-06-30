#!/usr/bin/env python3
import sys, subprocess, importlib, ctypes, os, time, threading, traceback, argparse

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(['"' + arg + '"' if ' ' in arg else arg for arg in sys.argv]), None, 1)
    sys.exit(0)

def ensure(pkg, import_name=None, pip_name=None):
    name = import_name if import_name else pkg
    pip_pkg = pip_name if pip_name else pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", MIRROR, pip_pkg])

from log_utils import log

for lib, imp in [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pyyaml", "yaml"),
    ("pyserial", "serial"),
    ("dxcam", "dxcam"),
    ("keyboard", "keyboard"),
    ("onnxruntime", "onnxruntime"),
    ("mss", "mss"),
]:
    ensure(lib, imp)

import cv2, numpy as np, math, atexit
from datetime import datetime
from collections import deque
from pathlib import Path
from dataclasses import dataclass, replace
import keyboard, dxcam, mss
from config import Config
from detector import (
    detect_targets as shared_detect_targets,
    format_detect_stats as shared_format_detect_stats,
    new_detect_stats as shared_new_detect_stats,
)
from tracker import TargetTracker, Target
from leonardo_driver import LeonardoMouseDriver
from fire_classifier import FireClassifier

# ---------- DPI 感知 ----------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except:
    pass

@atexit.register
def restore_timer():
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except:
        pass

def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except:
        return 1920, 1080


def wait_until(last_t: float, interval: float, spin: float = 0.00025):
    """高频循环用的轻量等待：保留末端短自旋，减少整段 busy-wait。"""
    target_t = last_t + interval
    while True:
        rem = target_t - time.perf_counter()
        if rem <= 0:
            return
        if rem > spin:
            time.sleep(rem - spin)


def target_bbox(t):
    bbox_x = getattr(t, "bbox_x", None)
    bbox_y = getattr(t, "bbox_y", None)
    x1 = int(bbox_x) if bbox_x is not None else int(t.x - t.w / 2)
    y1 = int(bbox_y) if bbox_y is not None else int(t.y - t.h * 0.1)
    return x1, y1, x1 + int(t.w), y1 + int(t.h)


def crop_target_bgr(frame_rgb, t):
    """只保存目标 ROI，避免 hard_neg_cache 缓存整帧造成内存膨胀。"""
    if frame_rgb is None or t is None:
        return None
    x1, y1, x2, y2 = target_bbox(t)
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi_rgb = frame_rgb[y1:y2, x1:x2]
    if roi_rgb.size <= 0:
        return None
    return cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)


def crop_center_bgr(frame_rgb, half=100):
    if frame_rgb is None:
        return None
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, w // 2 - half), max(0, h // 2 - half)
    x2, y2 = min(w, w // 2 + half), min(h, h // 2 + half)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)


def count_image_files(path: str) -> int:
    p = Path(path)
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def load_fire_classifier(cfg, color_lower_np, color_upper_np):
    if not os.path.exists(cfg.model_path):
        log(f"模型文件不存在: {cfg.model_path}，无法过滤", "WARN")
        return None
    try:
        classifier = FireClassifier(
            cfg.model_path, cfg.img_size, cfg.fire_threshold,
            cfg.model_filter_threshold, cfg.model_filter_consecutive, cfg.model_filter_cache_ttl,
            color_lower_np=color_lower_np, color_upper_np=color_upper_np)
        log("模型已加载，将在主循环中过滤已知背景", "SUCCESS")
        return classifier
    except Exception as e:
        log(f"模型加载失败: {e}，将不对目标过滤", "WARN")
        return None


class AutoTrainManager:
    """轻量自动训练调度器：只在样本数达标且没有训练进程运行时启动 train_model.py。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None
        self.last_check = 0.0
        self.last_total_samples = 0
        self.last_model_mtime = self._model_mtime()

    def _model_mtime(self):
        try:
            ready_path = self.cfg.model_path + ".ready.json"
            if os.path.exists(ready_path):
                return os.path.getmtime(ready_path)
            return os.path.getmtime(self.cfg.model_path)
        except OSError:
            return 0.0

    def model_changed(self) -> bool:
        mt = self._model_mtime()
        if mt > 0 and mt != self.last_model_mtime:
            self.last_model_mtime = mt
            return True
        return False

    def poll(self):
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.proc = None
            if code == 0:
                log("自动训练进程已完成，可尝试热加载新模型", "SUCCESS")
            else:
                log(f"自动训练进程异常退出 code={code}", "WARN")

    def maybe_start(self):
        if not self.cfg.auto_train_enabled:
            return
        self.poll()
        if self.proc is not None:
            return
        now = time.perf_counter()
        if now - self.last_check < self.cfg.train_check_interval:
            return
        self.last_check = now
        fire_n = count_image_files("dataset/fire")
        no_fire_n = count_image_files("dataset/no_fire")
        total = fire_n + no_fire_n
        if fire_n < self.cfg.train_sample_threshold or no_fire_n < self.cfg.train_sample_threshold:
            return
        if total <= self.last_total_samples:
            return
        self.last_total_samples = total
        try:
            self.proc = subprocess.Popen([sys.executable, "train_model.py"], cwd=os.getcwd())
            log(f"自动训练已启动 fire={fire_n} no_fire={no_fire_n}", "SUCCESS")
        except Exception as e:
            log(f"自动训练启动失败: {e}", "WARN")

# ---------- 共享状态 ----------
class SharedState:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.enabled = True
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.frame_updated = False
        self.latest_frame_for_motor = None
        self.ex = self.ey = 0.0
        self.raw_ex = self.raw_ey = 0.0
        self.vx = self.vy = 0.0
        self.can_fire = False
        self.has_real_detection = False
        self.acquired_time = 0.0
        self.is_firing = False
        self.target_frame_id = 0
        self.target_distance = 999.0
        self.ego_dx = self.ego_dy = 0.0
        self.best_target = None
        self.target_confidence = 0.0

    @property
    def running(self):
        return not self.stop_event.is_set()

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.state_lock:
            best_target = self.best_target
            if best_target is not None:
                try:
                    best_target = replace(best_target)
                except Exception:
                    pass
            return StateSnapshot(
                self.enabled, self.ex, self.ey, self.raw_ex, self.raw_ey,
                self.vx, self.vy, self.can_fire, self.has_real_detection,
                self.acquired_time, self.target_frame_id, self.target_distance,
                self.is_firing, best_target, self.target_confidence)

@dataclass
class StateSnapshot:
    enabled: bool
    ex: float
    ey: float
    raw_ex: float
    raw_ey: float
    vx: float
    vy: float
    can_fire: bool
    has_real_detection: bool
    acquired_time: float
    target_frame_id: int
    target_distance: float
    is_firing: bool
    best_target: object
    target_confidence: float = 0.0

shared_state = SharedState()

# ---------- 捕获线程 ----------
def capture_thread(cfg):
    try:
        sw, sh = get_screen_size()
        left, top = (sw - cfg.roi_width) // 2, (sh - cfg.roi_height) // 2
        region = (left, top, left + cfg.roi_width, top + cfg.roi_height)
        log(f"尝试捕获区域: {region}")
        camera = None
        use_mss = False
        try:
            camera = dxcam.create(output_color="RGB")
            camera.start(target_fps=cfg.target_fps, region=region)
            for _ in range(50):
                if camera.get_latest_frame() is not None:
                    log("dxcam 捕获成功", "SUCCESS")
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("dxcam 无帧")
        except Exception as e:
            log(f"dxcam 失败: {e}，切换 mss", "WARN")
            use_mss = True
            if camera:
                try: camera.stop()
                except: pass
            if use_mss:
                sct = mss.mss()
                monitor = {"top": top, "left": left, "width": cfg.roi_width, "height": cfg.roi_height}

        frame_miss = 0
        while shared_state.running:
            try:
                if use_mss:
                    img = np.array(sct.grab(monitor))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = camera.get_latest_frame()
                if img is not None:
                    captured_at = time.perf_counter()
                    with shared_state.frame_lock:
                        shared_state.latest_frame = img.copy()
                        shared_state.latest_frame_for_motor = shared_state.latest_frame
                        shared_state.latest_frame_time = captured_at
                        shared_state.frame_updated = True
                    frame_miss = 0
                else:
                    frame_miss += 1
                    if frame_miss % 100 == 0:
                        log(f"捕获丢帧 {frame_miss} 次", "WARN")
                    time.sleep(0.001)
            except:
                time.sleep(0.01)
        if not use_mss and camera: camera.stop()
        elif use_mss: sct.close()
    except Exception as e:
        log(f"捕获线程崩溃: {e}", "ERROR")
        shared_state.stop()

# ---------- 电机线程 ----------
# ---------- 电机线程 ----------
def motor_thread(cfg):
    log("电机线程初始化...", "INFO")
    drv = None
    is_pressing = False
    try:
        drv = LeonardoMouseDriver(cfg.leonardo_port, cfg.leonardo_baud)
        if not drv.initialized:
            log("Leonardo 驱动未初始化，移动功能不可用", "WARN")
        os.makedirs("dataset/no_fire", exist_ok=True)
        neg_counter = 0
        pos_counter = 0
        manual_key_was_pressed = False
        pos_key_was_pressed = False
        neg_manual_key = 0xC0
        aim_key_hex = int(cfg.aim_key, 16)
        residual_x = residual_y = 0.0
        last_consumed_frame_id = -1
        trigger_ready_since = None
        last_shot_time = 0.0
        burst_count = 0
        burst_cooldown_until = 0.0
        click_release_at = 0.0
        last_firing_state = False
        click_hold_seconds = 0.012
        move_limit = max(1, min(127, int(cfg.max_move)))
        trigger_conf_threshold = max(float(cfg.fire_threshold), float(cfg.model_filter_threshold))
        MOTOR_DT = 1.0 / 1000.0
        last_motor_t = time.perf_counter()
        _last_warn_block = 0.0  # ← 修复1: 初始化变量
        log("电机线程已启动 | 左Shift移动 | ~ 键截取误判背景", "SUCCESS")
        while shared_state.running:
            wait_until(last_motor_t, MOTOR_DT)
            last_motor_t = time.perf_counter()
            now_motor = last_motor_t
            if is_pressing and now_motor >= click_release_at:
                drv.release_left_click()
                is_pressing = False
            user_aim = bool(ctypes.windll.user32.GetAsyncKeyState(aim_key_hex) & 0x8000)
            snap = shared_state.snapshot()
            if not snap.enabled:
                residual_x = residual_y = 0.0
                trigger_ready_since = None
                burst_count = 0
                if is_pressing:
                    drv.release_left_click()
                    is_pressing = False
                if last_firing_state:
                    with shared_state.state_lock:
                        shared_state.is_firing = False
                    last_firing_state = False
                continue
            if not snap.can_fire or math.isnan(snap.ex) or math.isnan(snap.ey):
                residual_x = residual_y = 0.0
            target_valid = (snap.can_fire and snap.target_confidence >= cfg.model_filter_threshold)
            if not target_valid:
                residual_x = residual_y = 0.0
                trigger_ready_since = None
                burst_count = 0
                if is_pressing:
                    drv.release_left_click()
                    is_pressing = False
                if user_aim and snap.can_fire:
                    if now_motor - _last_warn_block > 1.0:
                        log(f"[电机阻断] 按下瞄准键，但目标置信度 {snap.target_confidence:.2f} < {cfg.model_filter_threshold}，移动被拦截", "WARN")
                        _last_warn_block = now_motor  # ← 修复2: 缩进进入 if 块内
            key_pressed = bool(ctypes.windll.user32.GetAsyncKeyState(neg_manual_key) & 0x8000)
            if key_pressed and not manual_key_was_pressed:
                with shared_state.frame_lock:
                    frame_rgb = shared_state.latest_frame_for_motor
                roi = crop_center_bgr(frame_rgb, half=100)
                if roi is not None and roi.size > 0:
                    fname = f"dataset/no_fire/{datetime.now().strftime('%H%M%S%f')}_manual.png"
                    cv2.imwrite(fname, roi)
                    neg_counter += 1
                    log(f"[+] 手动负样本已保存 (总计{neg_counter})", "SUCCESS")
            manual_key_was_pressed = key_pressed
            pos_manual_key = 0xDD
            key_pressed_pos = bool(ctypes.windll.user32.GetAsyncKeyState(pos_manual_key) & 0x8000)
            if key_pressed_pos and not pos_key_was_pressed:
                with shared_state.frame_lock:
                    frame_rgb = shared_state.latest_frame_for_motor
                roi = crop_center_bgr(frame_rgb, half=100)
                if roi is not None and roi.size > 0:
                    os.makedirs("dataset/fire", exist_ok=True)
                    fname = f"dataset/fire/{datetime.now().strftime('%H%M%S%f')}_enemy.png"
                    cv2.imwrite(fname, roi)
                    pos_counter += 1
                    log(f"[+] 敌人正样本已保存 (总计{pos_counter})", "SUCCESS")
            pos_key_was_pressed = key_pressed_pos
            if snap.can_fire and target_valid and snap.target_frame_id != last_consumed_frame_id:
                last_consumed_frame_id = snap.target_frame_id
                if (abs(snap.ex) > cfg.dead_zone or abs(snap.ey) > cfg.dead_zone) and user_aim:
                    effective_sens = cfg.sensitivity_scaler * (
                        cfg.sensitivity_boost_close if snap.target_distance < cfg.close_range_threshold else 1.0)
                    total_x = snap.ex * effective_sens
                    total_y = snap.ey * effective_sens
                    if 0 < abs(total_x) < cfg.min_kinetic_speed:
                        total_x = math.copysign(cfg.min_kinetic_speed, total_x)
                    if 0 < abs(total_y) < cfg.min_kinetic_speed:
                        total_y = math.copysign(cfg.min_kinetic_speed, total_y)
                    if (residual_x > 0 and total_x < 0) or (residual_x < 0 and total_x > 0):
                        residual_x = 0.0
                    if (residual_y > 0 and total_y < 0) or (residual_y < 0 and total_y > 0):
                        residual_y = 0.0
                    residual_x += total_x
                    residual_y += total_y
            if user_aim and target_valid:
                mx = max(-move_limit, min(move_limit, int(residual_x)))
                my = max(-move_limit, min(move_limit, int(residual_y)))
                if mx != 0 or my != 0:
                    drv.move(mx, my)
                    with shared_state.state_lock:
                        shared_state.ego_dx += mx
                        shared_state.ego_dy += my
                    residual_x -= mx
                    residual_y -= my
                if abs(residual_x) < 0.1:
                    residual_x = 0.0
                if abs(residual_y) < 0.1:
                    residual_y = 0.0

            trigger_allowed = bool(
                cfg.trigger_enabled and user_aim and target_valid and snap.can_fire
                and snap.target_confidence >= trigger_conf_threshold
            )
            if trigger_allowed:
                aligned = abs(snap.raw_ex) <= cfg.trigger_tolerance and abs(snap.raw_ey) <= cfg.trigger_tolerance
                velocity_per_frame = math.hypot(snap.vx, snap.vy) / max(1.0, float(cfg.target_fps))
                vy_per_frame = snap.vy / max(1.0, float(cfg.target_fps))
                velocity_ok = velocity_per_frame <= cfg.trigger_max_velocity_px_per_frame
                falling_ok = vy_per_frame <= cfg.ignore_falling_speed_px_per_frame
                if aligned and velocity_ok and falling_ok:
                    if trigger_ready_since is None:
                        trigger_ready_since = now_motor
                    delay_ok = (now_motor - trigger_ready_since) >= cfg.trigger_delay_first_shot
                    cooldown_ok = now_motor >= burst_cooldown_until
                    interval_ok = burst_count == 0 or (now_motor - last_shot_time) >= cfg.trigger_burst_interval
                    if delay_ok and cooldown_ok and interval_ok and not is_pressing:
                        if drv.press_left_click():
                            is_pressing = True
                            click_release_at = now_motor + click_hold_seconds
                            last_shot_time = now_motor
                            burst_count += 1
                            residual_y += float(cfg.rcs_pull_down_pixels)
                            if burst_count >= max(1, int(cfg.burst_shots_limit)):
                                burst_count = 0
                                trigger_ready_since = None
                                burst_cooldown_until = now_motor + cfg.burst_cooldown
                else:
                    trigger_ready_since = None
                    if now_motor >= burst_cooldown_until:
                        burst_count = 0
            else:
                trigger_ready_since = None
                burst_count = 0
                if is_pressing:
                    drv.release_left_click()
                    is_pressing = False

            if is_pressing != last_firing_state:
                with shared_state.state_lock:
                    shared_state.is_firing = is_pressing
                last_firing_state = is_pressing
    except Exception as e:
        log(f"电机线程崩溃: {e}\n{traceback.format_exc()}", "ERROR")
        shared_state.stop()
    finally:
        try:
            if drv is not None and drv.initialized:
                drv.release_left_click()
        except:
            pass
        try:
            if drv is not None:
                drv.close()
        except:
            pass


# ---------- 目标检测 ----------
def _clamp_float(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _split_row_bands(valid_rows, gap_tolerance: int):
    """把有效行切成若干连续行带，用于避开顶部零散噪声。"""
    if valid_rows.size == 0:
        return []
    bands = []
    start = int(valid_rows[0])
    prev = int(valid_rows[0])
    for r in valid_rows[1:]:
        r = int(r)
        if r - prev <= gap_tolerance + 1:
            prev = r
        else:
            bands.append((start, prev))
            start = prev = r
    bands.append((start, prev))
    return bands


def estimate_head_point_from_mask(mask, x: int, y: int, w: int, h: int, cfg):
    """
    OpenCV 几何头部估计：
    1) 根据姿态宽高比自适应扩大上部搜索区；
    2) 用行投影找稳定上部主体行，避开孤立噪声；
    3) 对行中心做加权平均，得到更稳的 x；
    4) y 偏移按目标高度自适应，避免远距离小目标被固定像素偏移拉飞。

    返回：final_hx, final_hy, debug_dict
    这里的 hx/hy 是 bbox 内坐标，不含 bbox 左上角 x/y。
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0, {"quality": 0.0, "pose": "invalid"}

    width_ratio = w / max(1.0, float(h))
    is_wide_pose = width_ratio >= float(cfg.head_wide_width_ratio_threshold)
    pose_hint = "wide" if is_wide_pose else "upright"

    search_ratio = float(cfg.head_search_ratio_wide if is_wide_pose else cfg.head_search_ratio)
    search_h = int(round(h * search_ratio))
    search_h = max(int(cfg.head_min_search_px), search_h)
    search_h = min(h, max(1, search_h))

    head_roi = mask[y:y + search_h, x:x + w]
    fallback_hx = w * 0.5
    fallback_hy = min(search_h - 1, max(0, int(search_h * 0.30)))

    debug = {
        "roi_x": x,
        "roi_y": y,
        "roi_w": w,
        "roi_h": search_h,
        "raw_hx": fallback_hx,
        "raw_hy": fallback_hy,
        "quality": 0.0,
        "pose": pose_hint,
    }

    if head_roi.size == 0:
        return fallback_hx, fallback_hy, debug

    rows, cols = np.nonzero(head_roi)
    if cols.size == 0:
        return fallback_hx, fallback_hy, debug

    row_count = np.bincount(rows, minlength=search_h).astype(np.float64)
    row_min = np.full(search_h, w, dtype=np.int32)
    row_max = np.full(search_h, -1, dtype=np.int32)
    np.minimum.at(row_min, rows, cols)
    np.maximum.at(row_max, rows, cols)

    min_pixels = max(int(cfg.head_row_min_pixels), int(math.ceil(w * float(cfg.head_row_min_density))))
    valid = row_count >= max(1, min_pixels)
    if not np.any(valid):
        # 小目标/极窄目标：退回所有非空行，但仍然避免直接崩掉。
        valid = row_count > 0
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size == 0:
        return fallback_hx, fallback_hy, debug

    bands = _split_row_bands(valid_rows, int(cfg.head_row_gap_tolerance))
    if not bands:
        return fallback_hx, fallback_hy, debug

    # 选择“足够强且足够靠上”的行带。不是简单选最宽行，避免下方肩/胸把头部点拖下去。
    best_band = bands[0]
    best_score = -1.0
    for b0, b1 in bands:
        rr = np.arange(b0, b1 + 1)
        counts = row_count[rr]
        if counts.size == 0:
            continue
        # 行带越靠上越优先；白色像素越稳定越优先；太短的孤立噪声会被 length 项压低。
        length = max(1, b1 - b0 + 1)
        vertical_penalty = math.exp(-float(cfg.head_vertical_decay) * b0)
        score = float(np.sum(np.power(np.maximum(counts, 1.0), float(cfg.head_row_weight_power))))
        score *= vertical_penalty * math.sqrt(length)
        if score > best_score:
            best_score = score
            best_band = (b0, b1)

    b0, b1 = best_band
    rr = np.arange(b0, b1 + 1)
    centers = (row_min[rr].astype(np.float64) + row_max[rr].astype(np.float64)) * 0.5
    counts = row_count[rr]

    weights = np.power(np.maximum(counts, 1.0), float(cfg.head_row_weight_power))
    # 同一行带内仍稍微偏向上方，防止肩部宽行过度控制 x/y。
    weights *= np.exp(-float(cfg.head_vertical_decay) * (rr - b0))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raw_hx = float(np.median(centers))
        raw_hy = float(b0 + (b1 - b0) * float(cfg.head_y_band_position))
    else:
        raw_hx = float(np.sum(centers * weights) / weight_sum)
        band_y = b0 + (b1 - b0) * float(cfg.head_y_band_position)
        weighted_y = float(np.sum(rr * weights) / weight_sum)
        # y 使用“行带内部比例点”和“加权行中心”融合；可避免点被单一宽肩行拖得过低。
        raw_hy = 0.65 * band_y + 0.35 * weighted_y

    # 按目标高度做 y 自适应偏移；宽姿态/下蹲时减少上移，避免打到头顶空气。
    offset_y = float(cfg.head_offset_y_ratio) * float(h)
    if is_wide_pose:
        offset_y *= float(cfg.head_wide_offset_scale)
    lo = min(float(cfg.head_offset_y_min), float(cfg.head_offset_y_max))
    hi = max(float(cfg.head_offset_y_min), float(cfg.head_offset_y_max))
    offset_y = _clamp_float(offset_y, lo, hi)

    final_hx = _clamp_float(raw_hx, 0, w - 1)
    final_hy = _clamp_float(raw_hy + offset_y, 0, h - 1)

    # 质量分仅用于调试显示：头部候选区内有效像素占比和行带分数的粗略组合。
    density = float(np.count_nonzero(head_roi)) / max(1.0, float(head_roi.size))
    band_strength = float(np.sum(row_count[rr])) / max(1.0, float(w * max(1, len(rr))))
    quality = max(0.0, min(1.0, 0.45 * density * 8.0 + 0.55 * band_strength))

    debug.update({
        "raw_hx": raw_hx,
        "raw_hy": raw_hy,
        "offset_y": offset_y,
        "band_start": int(b0),
        "band_end": int(b1),
        "quality": quality,
        "pose": pose_hint,
    })
    return final_hx, final_hy, debug


def _new_detect_stats():
    return {
        "raw": 0,
        "invalid": 0,
        "size": 0,
        "edge": 0,
        "area_min": 0,
        "area_max": 0,
        "shape": 0,
        "pre_head": 0,
        "head_estimated": 0,
        "head_limited": 0,
        "rescued_close": 0,
        "final": 0,
    }


def format_detect_stats(stats):
    if not stats:
        return "raw=0 final=0"
    return (
        f"raw={stats.get('raw', 0)} final={stats.get('final', 0)} "
        f"size={stats.get('size', 0)} edge={stats.get('edge', 0)} "
        f"area<{stats.get('area_min', 0)}/>{stats.get('area_max', 0)} "
        f"shape={stats.get('shape', 0)} "
        f"head={stats.get('head_estimated', 0)}/{stats.get('pre_head', 0)} "
        f"limit={stats.get('head_limited', 0)} rescue={stats.get('rescued_close', 0)}"
    )


def _bbox_hits_center_zone(x, y, w, h, cx, cy, radius):
    if radius <= 0:
        return x <= cx <= x + w and y <= cy <= y + h
    return (
        x <= cx + radius and x + w >= cx - radius and
        y <= cy + radius and y + h >= cy - radius
    )


def _is_close_target_rescue(x, y, w, h, area, cfg, cx, cy, roi_w, roi_h):
    if not cfg.close_target_rescue_enabled:
        return False
    max_area = max(float(cfg.max_contour_area), float(roi_w * roi_h) * float(cfg.close_target_max_area_ratio))
    if area <= 0 or area > max_area:
        return False
    return _bbox_hits_center_zone(x, y, w, h, cx, cy, int(cfg.close_target_center_zone_radius))


def detect_targets(img, gray, cfg, cx, cy, morph_kernel, color_lower_np, color_upper_np, prev_gray=None, motion_kernel=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, color_lower_np, color_upper_np)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=cfg.morph_iterations)

    if cfg.motion_enabled and prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, motion = cv2.threshold(diff, cfg.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        if motion_kernel is None:
            ksize = max(1, cfg.motion_dilate_kernel)
            motion_kernel = np.ones((ksize, ksize), np.uint8)
        motion = cv2.dilate(motion, motion_kernel, iterations=1)
        mask = cv2.bitwise_and(mask, motion)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = _new_detect_stats()
    stats["raw"] = len(cnts)
    res = []
    prelim = []
    roi_w = cfg.roi_width
    roi_h = cfg.roi_height
    edge = cfg.edge_margin

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            stats["invalid"] += 1
            continue
        area = cv2.contourArea(c)
        rescue_close = _is_close_target_rescue(x, y, w, h, area, cfg, cx, cy, roi_w, roi_h)

        size_hit = w > cfg.filter_max_width or h > cfg.filter_max_height or h < cfg.filter_min_height
        if size_hit:
            stats["size"] += 1
            if not rescue_close:
                continue

        edge_hit = x < edge or y < edge or (x + w) > (roi_w - edge) or (y + h) > (roi_h - edge)
        if edge_hit:
            stats["edge"] += 1
            if not rescue_close:
                continue

        if area < cfg.min_contour_area:
            stats["area_min"] += 1
            continue
        if area > cfg.max_contour_area:
            stats["area_max"] += 1
            if not rescue_close:
                continue

        perim = cv2.arcLength(c, True)
        circ = (4 * math.pi * area / (perim * perim)) if perim > 0 else 0
        if circ > cfg.filter_circularity_max:
            stats["shape"] += 1
            continue

        wh = w * h
        if wh <= 0:
            stats["invalid"] += 1
            continue
        extent = area / wh
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if extent < cfg.filter_rectangularity_min:
            stats["shape"] += 1
            continue
        if extent > cfg.filter_extent_max:
            stats["shape"] += 1
            continue
        if solidity > cfg.filter_solidity_max or solidity < cfg.filter_solidity_min:
            stats["shape"] += 1
            continue

        ar = h / w
        if ar < cfg.filter_aspect_min:
            stats["shape"] += 1
            continue

        if rescue_close and (size_hit or edge_hit or area > cfg.max_contour_area):
            stats["rescued_close"] += 1

        approx_head_x = x + w * 0.5
        approx_head_y = y + h * 0.25
        approx_dist = math.hypot(approx_head_x - cx, approx_head_y - cy)
        prelim.append((approx_dist, -float(area), x, y, w, h, float(area), float(ar), float(circ)))

    stats["pre_head"] = len(prelim)
    if len(prelim) > cfg.max_head_estimation_candidates:
        prelim.sort(key=lambda item: (item[0], item[1]))
        stats["head_limited"] = len(prelim) - cfg.max_head_estimation_candidates
        prelim = prelim[:cfg.max_head_estimation_candidates]

    stats["head_estimated"] = len(prelim)

    for _, _, x, y, w, h, area, ar, circ in prelim:
        hx, hy, head_dbg = estimate_head_point_from_mask(mask, x, y, w, h, cfg)

        raw_fx = x + float(head_dbg.get("raw_hx", hx))
        raw_fy = y + float(head_dbg.get("raw_hy", hy))
        fx = x + hx + cfg.targeting_offset_x
        fy = y + hy + cfg.targeting_offset_y
        dist = math.hypot(fx - cx, fy - cy)

        if cfg.dynamic_offset_enabled:
            if dist <= cfg.dynamic_offset_near_dist:
                oy_dyn = cfg.dynamic_offset_near_y
            elif dist >= cfg.dynamic_offset_far_dist:
                oy_dyn = cfg.dynamic_offset_far_y
            else:
                t = (dist - cfg.dynamic_offset_near_dist) / \
                    (cfg.dynamic_offset_far_dist - cfg.dynamic_offset_near_dist)
                oy_dyn = cfg.dynamic_offset_near_y + t * (cfg.dynamic_offset_far_y - cfg.dynamic_offset_near_y)
            fy = fy + oy_dyn
            dist = math.hypot(fx - cx, fy - cy)

        t = Target(float(fx), float(fy), float(area), dist, int(w), int(h), float(ar), float(circ), int(x), int(y))
        t.raw_head_x = float(raw_fx)
        t.raw_head_y = float(raw_fy)
        t.head_roi_x = int(head_dbg.get("roi_x", x))
        t.head_roi_y = int(head_dbg.get("roi_y", y))
        t.head_roi_w = int(head_dbg.get("roi_w", w))
        t.head_roi_h = int(head_dbg.get("roi_h", max(1, h)))
        t.head_quality = float(head_dbg.get("quality", 0.0))
        t.head_pose_hint = str(head_dbg.get("pose", "upright"))
        res.append(t)
    stats["final"] = len(res)
    return res, mask, stats

# ---------- 运行模式 ----------
def parse_runtime_args():
    parser = argparse.ArgumentParser(description="实时视觉系统入口")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", choices=("debug", "benchmark", "runtime"), help="覆盖 config.yaml 中的 run_mode")
    parser.add_argument("--frames", type=int, default=None, help="最多处理多少帧；0 表示不限制")
    return parser.parse_args()


def apply_run_mode(cfg):
    mode = cfg.run_mode
    cfg.auto_train_enabled = False
    if mode == "debug":
        cfg.show_debug = True
        cfg.control_enabled = False
        cfg.debug_draw_stride = max(1, int(cfg.debug_draw_stride))
    elif mode == "benchmark":
        cfg.show_debug = False
        cfg.debug_show_mask = False
        cfg.control_enabled = False
        cfg.auto_save_hard_negatives = False
    elif mode == "runtime":
        cfg.show_debug = False
        cfg.debug_show_mask = False
    return cfg


# ---------- 主循环 ----------
def main():
    args = parse_runtime_args()
    cfg = Config.from_yaml(args.config)
    if args.mode:
        cfg.run_mode = args.mode
    if args.frames is not None:
        cfg.benchmark_frames = max(0, int(args.frames))
    cfg = apply_run_mode(cfg)
    cx, cy = cfg.roi_width // 2, cfg.roi_height // 2
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, cfg.morph_kernel_width), max(1, cfg.morph_kernel_height)))
    motion_kernel = None
    if cfg.motion_enabled:
        ksize = max(1, int(cfg.motion_dilate_kernel))
        motion_kernel = np.ones((ksize, ksize), np.uint8)
    color_lower_np = np.array(cfg.color_lower, dtype=np.uint8)
    color_upper_np = np.array(cfg.color_upper, dtype=np.uint8)
    tracker = TargetTracker(cfg)
    if cfg.model_inference_in_main:
        fire_classifier = load_fire_classifier(cfg, color_lower_np, color_upper_np)
    else:
        fire_classifier = None
        log("model_inference_in_main=false，当前按纯视觉模式运行", "WARN")
    auto_trainer = AutoTrainManager(cfg)
    if cfg.auto_train_enabled:
        log("auto_train_enabled=true: runtime may start train_model.py and hurt latency; use only for offline training.", "WARN")

    os.makedirs("dataset/no_fire", exist_ok=True)
    frame_time = 1.0 / cfg.target_fps
    if cfg.show_debug:
        dummy_w = cfg.roi_width * (2 if cfg.debug_show_mask else 1)
        dummy = np.zeros((cfg.roi_height, dummy_w, 3), dtype=np.uint8)
        cv2.putText(dummy, "Starting...", (50, cfg.roi_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("V79.0 Leonardo HID", dummy)
        cv2.waitKey(1)

    t_cap = threading.Thread(target=capture_thread, args=(cfg,), daemon=True)
    t_motor = threading.Thread(target=motor_thread, args=(cfg,), daemon=True) if cfg.control_enabled else None
    t_cap.start()
    if t_motor is not None:
        t_motor.start()

    def toggle():
        with shared_state.state_lock:
            shared_state.enabled = not shared_state.enabled
            log(f"引擎: {'ON' if shared_state.enabled else 'OFF'}")
    if cfg.run_mode != "benchmark":
        keyboard.add_hotkey("F12", toggle)
    log(f"V79.0 异常检测版 | mode={cfg.run_mode} control={'ON' if cfg.control_enabled else 'OFF'}", "SUCCESS")
    log("主循环已启动，等待第一帧...", "INFO")

    last_t = time.perf_counter()
    frame_times = deque(maxlen=100)
    detect_times = deque(maxlen=100)
    track_times = deque(maxlen=100)
    infer_times = deque(maxlen=100)
    prev_gray = None
    detect_stats = shared_new_detect_stats()
    debug_frame_id = 0
    frames_processed = 0
    reject_counter = 0
    hard_neg_cache = {}
    hard_neg_counter = 0
    _prev_best_pos = None

    _log_last_accept = 0.0
    _log_last_reject_warn = 0.0
    _log_last_periodic = 0.0
    _log_last_no_target = 0.0
    _log_accept_count = 0
    _log_reject_count = 0
    _log_cache_hit = 0
    _log_cache_miss = 0
    _log_fire_count = 0
    LOG_THROTTLE = 0.3
    LOG_PERIODIC = 2.0
    LOG_NO_TARGET = 3.0

    try:
        while shared_state.running:
            wait_until(last_t, frame_time)
            loop_start = time.perf_counter()
            dt = min(loop_start - last_t, 0.1)
            last_t = loop_start

            with shared_state.frame_lock:
                new_frame = shared_state.frame_updated
                if new_frame:
                    img = shared_state.latest_frame
                    frame_captured_at = shared_state.latest_frame_time
                    shared_state.frame_updated = False
                else:
                    img = None
                    frame_captured_at = 0.0

            if img is None:
                time.sleep(0.001)
                continue

            gray_t = time.perf_counter()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_ms = (time.perf_counter() - gray_t) * 1000
            with shared_state.state_lock:
                edx, edy = shared_state.ego_dx, shared_state.ego_dy
                shared_state.ego_dx = shared_state.ego_dy = 0
            tracker.kalman.apply_ego_motion(edx, edy, cfg.ego_scaler)

            t0 = time.perf_counter()
            cands, mask, detect_stats = shared_detect_targets(img, gray, cfg, cx, cy, mk, color_lower_np, color_upper_np, prev_gray, motion_kernel, source_color="RGB")
            detect_stats.capture_age_ms = (loop_start - frame_captured_at) * 1000 if frame_captured_at > 0 else 0.0
            detect_stats.preprocess_ms += gray_ms
            detect_ms = (time.perf_counter() - t0) * 1000
            detect_times.append(detect_ms)
            prev_gray = gray

            t_track = time.perf_counter()
            ex, ey, rex, rey, vx, vy, cf, hrd, best = tracker.update(cands, cx, cy, dt)
            track_ms = (time.perf_counter() - t_track) * 1000
            detect_stats.track_ms = track_ms
            if getattr(tracker, "last_reject_reason", None) == "static":
                detect_stats.reject_by_static += 1
            track_times.append(track_ms)
            now = time.perf_counter()
            if best is None:
                reject_counter = 0
                _prev_best_pos = None
            else:
                cur_best_pos = (float(best.x), float(best.y))
                if _prev_best_pos is not None:
                    switch_threshold = max(50.0, float(cfg.model_filter_reject_radius))
                    if (abs(cur_best_pos[0] - _prev_best_pos[0]) > switch_threshold or
                            abs(cur_best_pos[1] - _prev_best_pos[1]) > switch_threshold):
                        reject_counter = 0
                _prev_best_pos = cur_best_pos
            
            # ===== 核心修复：模型未加载时降级为纯视觉模式，保证能移动 =====
            if fire_classifier is None:
                target_confidence = 1.0 if (best is not None and cf) else 0.0
            else:
                target_confidence = 0.0
            # ===========================================================

            if best is None:
                if now - _log_last_no_target > LOG_NO_TARGET:
                    _log_last_no_target = now
                    avg_det = sum(detect_times) / len(detect_times) if detect_times else 0
                    log(f"[IDLE] 无目标 | 候选数={len(cands)} 检测={detect_ms:.1f}ms avg_det={avg_det:.1f}ms {shared_format_detect_stats(detect_stats)} 禁区={len(tracker._rejected_positions)}", "INFO")

            elif fire_classifier is not None and cf:

                t_inf = time.perf_counter()
                cache_key = fire_classifier._cache_key(best)
                was_cached = (cache_key in fire_classifier._cache and now - fire_classifier._cache[cache_key][1] < fire_classifier.cache_ttl)
                if cfg.auto_save_hard_negatives:
                    hard_x = getattr(best, "bbox_x", None)
                    hard_y = getattr(best, "bbox_y", None)
                    hard_key = (int(hard_x if hard_x is not None else best.x) >> 3,
                                int(hard_y if hard_y is not None else best.y) >> 3)
                    if hard_key not in hard_neg_cache:
                        roi_bgr = crop_target_bgr(img, best)
                        if roi_bgr is not None:
                            hard_neg_cache[hard_key] = roi_bgr
                try:
                    prob = fire_classifier.predict_proba(img, mask, best, use_cache=True) 
                except Exception as e:
                    prob = 0.0
                    log(f"模型推理异常: {e}，本轮拒绝", "WARN")
                infer_ms = (time.perf_counter() - t_inf) * 1000
                detect_stats.inference_ms = infer_ms
                infer_times.append(infer_ms)
                best.confidence = prob
                target_confidence = prob

                if was_cached: _log_cache_hit += 1
                else: _log_cache_miss += 1

                if prob >= cfg.model_filter_threshold:
                    _log_accept_count += 1
                    reject_counter = 0
                    if prob >= cfg.fire_threshold:
                        _log_fire_count += 1
                        if now - _log_last_accept > LOG_THROTTLE:
                            _log_last_accept = now
                            log(f"[ACCEPT ✓✓] 异常得分={prob:.4f} ≥ {cfg.fire_threshold:.1f} (非背景，开火级) pos=({best.x:.0f},{best.y:.0f}) dist={best.distance:.0f} w={best.w} h={best.h} inf={infer_ms:.1f}ms cache={'HIT' if was_cached else 'MISS'} det={detect_ms:.1f}ms", "SUCCESS")
                    else:
                        if now - _log_last_accept > LOG_THROTTLE:
                            _log_last_accept = now
                            log(f"[ACCEPT ✓ ] 异常得分={prob:.4f} (瞄准级 {cfg.model_filter_threshold:.1f}~{cfg.fire_threshold:.1f}) pos=({best.x:.0f},{best.y:.0f}) inf={infer_ms:.1f}ms cache={'HIT' if was_cached else 'MISS'}", "INFO")
                else:
                    _log_reject_count += 1
                    reject_counter += 1
                    
                    # ===== 核心修复：静止单帧拒绝的高频日志，防止 I/O 阻塞主循环 =====
                    if reject_counter >= cfg.model_filter_consecutive:
                        log(f"[CONFIRMED REJECT ✗✗] 连续 {reject_counter} 帧判定为背景，确认误检！pos=({best.x:.0f},{best.y:.0f})", "ERROR")
                        if cfg.auto_save_hard_negatives:
                            hard_x = getattr(best, "bbox_x", None)
                            hard_y = getattr(best, "bbox_y", None)
                            key = (int(hard_x if hard_x is not None else best.x) >> 3,
                                   int(hard_y if hard_y is not None else best.y) >> 3)
                            roi = hard_neg_cache.pop(key, None)
                            if roi is None:
                                roi = crop_target_bgr(img, best)
                            if roi is not None and roi.size > 0:
                                fname = f"dataset/no_fire/{datetime.now().strftime('%H%M%S%f')}_hard.png"
                                cv2.imwrite(fname, roi)
                                hard_neg_counter += 1
                                log(f"[HARD NEG SAVED] 硬负样本 #{hard_neg_counter} 已保存 → {fname}", "SUCCESS")
                        tracker.add_rejected_position(best.x, best.y)
                        log(f"[REJECT ZONE] 禁区已建立 当前禁区数={len(tracker._rejected_positions)}", "WARN")
                        tracker.reset()
                        reject_counter = 0
                        cf = False
                        hrd = False
                        best = None
                        ex = ey = rex = rey = vx = vy = 0.0
                        target_confidence = 0.0
                if len(hard_neg_cache) > 50: hard_neg_cache.clear()

            with shared_state.state_lock:
                shared_state.ex = ex
                shared_state.ey = ey
                shared_state.raw_ex = rex
                shared_state.raw_ey = rey
                shared_state.vx = vx
                shared_state.vy = vy
                shared_state.can_fire = cf
                shared_state.has_real_detection = hrd
                shared_state.acquired_time = tracker.target_acquired_time
                shared_state.target_distance = best.distance if best else 999
                shared_state.target_frame_id += 1
                shared_state.best_target = replace(best) if best else None
                shared_state.target_confidence = target_confidence

            auto_trainer.maybe_start()
            if cfg.model_inference_in_main and auto_trainer.model_changed():
                reloaded = load_fire_classifier(cfg, color_lower_np, color_upper_np)
                if reloaded is not None:
                    fire_classifier = reloaded
                    log("检测到模型文件更新，已完成热加载", "SUCCESS")

            if now - _log_last_periodic > LOG_PERIODIC:
                _log_last_periodic = now
                total_infer = _log_cache_hit + _log_cache_miss
                hit_rate = (_log_cache_hit / total_infer * 100) if total_infer > 0 else 0
                avg_inf = sum(infer_times) / len(infer_times) if infer_times else 0
                avg_det = sum(detect_times) / len(detect_times) if detect_times else 0
                avg_track = sum(track_times) / len(track_times) if track_times else 0
                avg_frame = sum(frame_times) / len(frame_times) if frame_times else 0
                log(f"[SUMMARY] 接受={_log_accept_count} 开火={_log_fire_count} 拒绝={_log_reject_count} 硬负样本={hard_neg_counter} | 缓存命中={hit_rate:.0f}% ({_log_cache_hit}/{total_infer}) | 推理={avg_inf:.1f}ms 检测={avg_det:.1f}ms 追踪={avg_track:.2f}ms 帧耗={avg_frame:.1f}ms | {shared_format_detect_stats(detect_stats)} | 禁区={len(tracker._rejected_positions)}", "INFO")
                _log_accept_count = 0
                _log_reject_count = 0
                _log_cache_hit = 0
                _log_cache_miss = 0
                _log_fire_count = 0

            debug_frame_id += 1
            if cfg.show_debug and (debug_frame_id % cfg.debug_draw_stride == 0):
                t_debug = time.perf_counter()
                dbg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                def target_rect(t):
                    bbox_x = getattr(t, "bbox_x", None)
                    bbox_y = getattr(t, "bbox_y", None)
                    x1 = int(bbox_x) if bbox_x is not None else int(t.x - t.w / 2)
                    y1 = int(bbox_y) if bbox_y is not None else int(t.y - t.h * 0.1)
                    return (x1, y1), (x1 + int(t.w), y1 + int(t.h))

                def draw_head_debug(canvas, t, color, thick=1):
                    if getattr(cfg, "show_head_roi", True):
                        hrx = getattr(t, "head_roi_x", None)
                        hry = getattr(t, "head_roi_y", None)
                        hrw = getattr(t, "head_roi_w", None)
                        hrh = getattr(t, "head_roi_h", None)
                        if None not in (hrx, hry, hrw, hrh):
                            cv2.rectangle(canvas, (int(hrx), int(hry)), (int(hrx + hrw), int(hry + hrh)), color, 1)
                    if getattr(cfg, "show_aim_point", True):
                        raw_x = getattr(t, "raw_head_x", None)
                        raw_y = getattr(t, "raw_head_y", None)
                        if raw_x is not None and raw_y is not None:
                            cv2.circle(canvas, (int(raw_x), int(raw_y)), 2, (255, 255, 255), -1)
                        cv2.circle(canvas, (int(t.x), int(t.y)), 4, color, -1)
                        cv2.drawMarker(canvas, (int(t.x), int(t.y)), color, cv2.MARKER_CROSS, 12, thick)
                        q = getattr(t, "head_quality", 0.0)
                        pose = getattr(t, "head_pose_hint", "")
                        cv2.putText(canvas, f"head:{q:.2f}/{pose}", (int(t.x) + 8, int(t.y) + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                
                # 核心判断：只有神经网络置信度达标，才算真敌人
                is_confirmed = (best is not None and cf and target_confidence >= cfg.model_filter_threshold)
                
                # 1. 绘制未确认的候选目标 (灰色细框)
                draw_cands = cands
                if cfg.debug_max_draw_candidates > 0 and len(draw_cands) > cfg.debug_max_draw_candidates:
                    draw_cands = draw_cands[:cfg.debug_max_draw_candidates]
                for t in draw_cands:
                    if best and t is best and (is_confirmed or target_confidence > 0.0):
                        continue
                    p1, p2 = target_rect(t)
                    cv2.rectangle(dbg, p1, p2, (80, 80, 80), 1)
                    if getattr(cfg, "show_head_roi", True):
                        draw_head_debug(dbg, t, (120, 120, 120), 1)

                # 2. 绘制追踪器锁定但未达开火阈值的目标 (黄色框，表示等待模型确认)
                if best is not None and cf and not is_confirmed:
                    col = (0, 255, 255) # 黄色
                    p1, p2 = target_rect(best)
                    cv2.rectangle(dbg, p1, p2, col, 1)
                    draw_head_debug(dbg, best, col, 1)
                    cv2.putText(dbg, f"PENDING {target_confidence:.2f}", (int(best.x) + 10, int(best.y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # 3. 绘制经过神经网络完全确认的敌人 (绿色粗框 + 十字准星)
                if is_confirmed:
                    col = (0, 255, 0) # 绿色
                    thick = 2
                    p1, p2 = target_rect(best)
                    cv2.rectangle(dbg, p1, p2, col, thick)
                    draw_head_debug(dbg, best, col, 2)
                    
                    conf_text = f"conf:{target_confidence:.2f}"
                    cv2.putText(dbg, conf_text, (int(best.x) + 10, int(best.y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # 4. 绘制预测与瞄准向量 (仅当锁定敌人时)
                if cf and not (math.isnan(rex) or math.isnan(rey)):
                    cv2.line(dbg, (cx, cy), (int(cx + rex), int(cy + rey)), (0, 255, 255), 2)
                    cv2.circle(dbg, (int(cx + rex), int(cy + rey)), 4, (0, 255, 255), -1)
                if not (math.isnan(ex) or math.isnan(ey)):
                    cv2.line(dbg, (cx, cy), (int(cx + ex), int(cy + ey)), (255, 0, 255), 1, cv2.LINE_AA)
                if cfg.show_prediction_vector and tracker.kalman.initialized:
                    px = int(cx + tracker.kalman.state[0] + tracker.kalman.state[2] * cfg.prediction_vector_dt)
                    py = int(cy + tracker.kalman.state[1] + tracker.kalman.state[3] * cfg.prediction_vector_dt)
                    cv2.arrowedLine(dbg, (int(cx + ex), int(cy + ey)), (px, py), (255, 255, 0), 1, tipLength=0.3)

                # 5. 绘制拒绝禁区 (红色圆圈)
                for rx, ry, rt in tracker._rejected_positions:
                    age = time.perf_counter() - rt
                    if age < tracker.reject_cooldown:
                        alpha = 1.0 - age / tracker.reject_cooldown
                        col_r = int(128 * alpha)
                        cv2.circle(dbg, (int(rx), int(ry)), int(tracker.reject_radius), (col_r, 0, 0), 1)
                
                # 6. 状态文字；mask 仅在显式打开时拼接，避免 debug 每帧双倍绘制。
                debug_canvas = dbg
                if cfg.debug_show_mask:
                    mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    cv2.putText(mask_show, "Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    debug_canvas = np.hstack((dbg, mask_show))
                
                avg_frame = sum(frame_times) / len(frame_times) if frame_times else 0
                avg_detect = sum(detect_times) / len(detect_times) if detect_times else 0
                avg_infer = sum(infer_times) / len(infer_times) if infer_times else 0
                fps = int(1000 / avg_frame) if avg_frame > 0 else 999
                
                status = "LOST"
                if tracker.kalman.initialized:
                    status = f"TRACK H:{tracker.hits} L:{tracker.lost_frames}"
                    if cf: status += " [LOCK]" if hrd else " [COAST]"
                if not shared_state.enabled: status += " [PAUSE]"
                
                cv2.putText(debug_canvas, f"FPS:{fps} {status} det:{avg_detect:.1f}ms inf:{avg_infer:.1f}ms cand:{len(cands)} rej:{len(tracker._rejected_positions)} hard:{hard_neg_counter}", 
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("V79.0 Leonardo HID", debug_canvas)
                key = cv2.waitKey(1) & 0xFF
                detect_stats.debug_draw_ms = (time.perf_counter() - t_debug) * 1000
                if key == 27:
                    break

            total_loop_ms = (time.perf_counter() - loop_start) * 1000
            detect_stats.total_loop_ms = total_loop_ms
            frame_times.append(total_loop_ms)
            frames_processed += 1
            if cfg.run_mode == "benchmark":
                log(
                    "[BENCH] "
                    f"frame={frames_processed} total_loop_ms={detect_stats.total_loop_ms:.2f} "
                    f"capture_age_ms={detect_stats.capture_age_ms:.2f} preprocess_ms={detect_stats.preprocess_ms:.2f} "
                    f"contour_ms={detect_stats.contour_ms:.2f} filter_ms={detect_stats.filter_ms:.2f} "
                    f"head_estimate_ms={detect_stats.head_estimate_ms:.2f} track_ms={detect_stats.track_ms:.2f} "
                    f"inference_ms={detect_stats.inference_ms:.2f} debug_draw_ms={detect_stats.debug_draw_ms:.2f} "
                    f"candidate_count={detect_stats.candidate_count} raw_contour_count={detect_stats.raw_contour_count}",
                    "INFO",
                )
                if cfg.benchmark_frames > 0 and frames_processed >= cfg.benchmark_frames:
                    break


    except KeyboardInterrupt:
        log("用户中断")
    finally:
        keyboard.unhook_all()
        shared_state.stop()
        time.sleep(0.1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
