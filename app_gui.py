from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception as exc:
    raise SystemExit(f"VisionForge GUI 初始化失败：Tkinter 不可用。{exc!r}")

try:
    import customtkinter as ctk  # type: ignore
except Exception:
    ctk = None  # type: ignore

try:
    import yaml
except Exception as exc:
    raise SystemExit("缺少 PyYAML。源码运行请先安装 requirements-exe.txt。") from exc

from src.app_paths import (
    VERSION,
    apply_runtime_overrides,
    backups_dir,
    config_path,
    configure_dll_search_path,
    ensure_runtime_layout,
    icon_path,
    logs_dir,
    model_path,
    resource_root,
    user_data_dir,
)
from src.offline_license import LicenseStatus, license_path, load_license, machine_code, save_license

try:
    from tools.env_diagnostics import OFFICIAL_LINKS, collect_diagnostics, summarize
except Exception:
    OFFICIAL_LINKS = {
        "NVIDIA 驱动下载": "https://www.nvidia.com/Download/index.aspx",
        "CUDA Toolkit 下载": "https://developer.nvidia.com/cuda-downloads",
        "cuDNN 下载": "https://developer.nvidia.com/cudnn-downloads",
        "ONNX Runtime 安装说明": "https://onnxruntime.ai/docs/install/",
        "ONNX Runtime CUDA EP 要求": "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
        "Microsoft VC++ 运行库": "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist",
    }
    collect_diagnostics = None  # type: ignore
    summarize = None  # type: ignore

APP_TITLE = "VisionForge"
ensure_runtime_layout()
configure_dll_search_path()
CONFIG_PATH = config_path()
LOG_DIR = logs_dir()
BACKUP_DIR = backups_dir()
MODEL_PATH = model_path()


def load_yaml(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    ensure_runtime_layout()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("配置文件格式错误：根节点必须是字典。")
    return data


def dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120)


def save_yaml_atomic(data: Dict[str, Any], path: Path = CONFIG_PATH) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"config_{stamp}.yaml"
    if path.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)}")
    tmp.write_text(dump_yaml(data), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return backup


def get_path(data: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(data: Dict[str, Any], dotted: str, value: Any) -> None:
    cur: Dict[str, Any] = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


@dataclass(frozen=True)
class FriendlyParam:
    key: str
    group: str
    title: str
    path: str
    kind: str
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    suffix: str = ""
    description: str = ""
    lower_tip: str = ""
    higher_tip: str = ""
    risk: str = ""


FRIENDLY_PARAMS: List[FriendlyParam] = [
    # 识别
    FriendlyParam("model_conf", "识别", "识别灵敏度", "model.conf", "float", 0.12, 0.45, 0.01, "", "决定模型输出目标的最低可信度。", "更容易识别远处或短暂出现的目标，但误识别风险上升。", "误识别更少，但弱目标可能被漏掉。", "建议 0.18～0.24。"),
    FriendlyParam("max_candidates", "识别", "候选目标上限", "model.max_candidates", "int", 80, 360, 10, "个", "每帧最多保留多少个候选目标。", "复杂画面更不容易漏候选，但处理压力上升。", "更轻量，但复杂场景可能漏候选。", "建议 160～260。"),
    FriendlyParam("require_gpu", "识别", "强制使用 GPU", "model.require_gpu", "bool", description="启动时必须启用 GPU 推理，避免静默回退 CPU。", lower_tip="允许 CPU 兼容运行，但速度较慢。", higher_tip="GPU 未打通时直接提示修复。", risk="正式发行建议开启。"),
    FriendlyParam("prefer_tensorrt", "识别", "优先 TensorRT 加速", "model.prefer_tensorrt", "bool", description="本机支持 TensorRT 时优先使用更高性能的推理后端。", lower_tip="只使用 CUDA Provider。", higher_tip="优先 TensorRT，失败再使用 CUDA。", risk="没有安装 TensorRT 时会自动使用 CUDA。"),
    FriendlyParam("head_conf", "识别", "头部最低可信度", "detection_filter.min_head_conf", "float", 0.16, 0.55, 0.01, "", "进入筛选前头部目标的最低可信度。", "更容易召回弱目标。", "更严格、更少误识别。", "过低会增加噪声。"),
    FriendlyParam("body_conf", "识别", "身体最低可信度", "detection_filter.min_body_conf", "float", 0.16, 0.60, 0.01, "", "身体框参与配对的最低可信度。", "远距离身体框更容易保留。", "身体框更可信但可能漏目标。", "建议不低于 0.22。"),
    FriendlyParam("paired_head", "识别", "配对头部可信度", "detection_filter.paired_head_min_conf", "float", 0.16, 0.55, 0.01, "", "头部与身体配对成功时允许的头部最低可信度。", "更快召回真实目标。", "更保守，误识别更少。", "配对目标可比单独头部更宽松。"),
    FriendlyParam("head_only", "识别", "单独头部可信度", "detection_filter.head_only_min_conf", "float", 0.75, 0.98, 0.01, "", "没有身体配对时，单独头部必须达到的可信度。", "更容易接受单独头部，但误识别风险上升。", "更严格，地图小点更难进入。", "不建议低于 0.82。"),
    FriendlyParam("small_pair_far_h", "识别", "小目标远距离头部可信度", "detection_filter.small_pair_far_min_head_conf", "float", 0.40, 0.85, 0.01, "", "小目标距离中心较远时，头部配对的最低可信度。", "更容易识别远处小目标。", "更少误识别。", "远目标识别弱时调低。"),
    FriendlyParam("small_pair_far_b", "识别", "小目标远距离身体可信度", "detection_filter.small_pair_far_min_body_conf", "float", 0.35, 0.85, 0.01, "", "小目标距离中心较远时，身体配对的最低可信度。", "远处真实目标更容易通过。", "过滤更严格。", "建议 0.46～0.62。"),
    FriendlyParam("body_height", "识别", "身体最小高度", "detection_filter.min_body_height_px", "float", 12, 42, 1, "px", "身体框至少要有多高。", "更容易保留远处小目标。", "过滤更多小噪声。", "如果真实远目标识别不到，可适度调低。"),
    FriendlyParam("small_body_height", "识别", "小目标身体高度", "detection_filter.small_min_body_height_px", "float", 12, 42, 1, "px", "小目标身体框至少要有多高。", "更容易召回远处小目标。", "更少小噪声。", "建议 18～24。"),
    FriendlyParam("body_aspect", "识别", "身体宽高容忍", "detection_filter.max_body_aspect", "float", 0.80, 1.60, 0.01, "", "身体框过宽时会被过滤，这里控制容忍度。", "更宽松，特殊姿态更容易通过。", "更严格，扁宽误识别更少。", "建议 1.15～1.35。"),
    FriendlyParam("quick_enter_conf", "识别", "出现即响应可信度", "control.reactive_fast_enter_min_conf", "float", 0.20, 0.90, 0.01, "", "目标刚出现时进入移动的最低可信度。", "响应更快，但误识别风险上升。", "更稳，但进入控制更慢。", "建议 0.30～0.58。"),
    FriendlyParam("quick_enter_body", "识别", "出现即响应身体可信度", "control.reactive_fast_enter_min_body_conf", "float", 0.20, 0.90, 0.01, "", "快速进入通道对身体框的最低可信度要求。", "更容易快速响应。", "更稳。", "与识别灵敏度联动调整。"),
    FriendlyParam("quick_enter_range", "识别", "快速响应范围", "control.reactive_fast_enter_center_dist_px", "int", 80, 260, 5, "px", "距离中心多远以内允许快速进入。", "更大范围内响应更快。", "只在近中心目标上快速响应。", "范围越大越要留意误识别。"),

    # 目标锁定
    FriendlyParam("switch_frames", "目标锁定", "切换确认", "target_lock.switch_confirm_frames", "int", 1, 6, 1, "帧", "多目标切换时需要连续确认多少帧。", "切换更快。", "更稳定，不容易来回跳。", "多目标乱跳时调高。"),
    FriendlyParam("missing_switch", "目标锁定", "丢失切换确认", "target_lock.missing_switch_confirm_frames", "int", 1, 6, 1, "帧", "当前目标短暂丢失后，切到新目标需要确认多少帧。", "更快接新目标。", "更保守。", "目标突然出现但响应慢时调低。"),
    FriendlyParam("hold_lost", "目标锁定", "短时丢失保持", "target_lock.hold_lost_frames", "int", 0, 20, 1, "帧", "目标短时没被检测到时，保留锁定多少帧。", "更少断续。", "更快释放旧目标。", "过高会拖旧目标。"),
    FriendlyParam("lock_velocity", "目标锁定", "锁定速度上限", "target_lock.max_lock_velocity_px_s", "int", 1000, 5000, 50, "px/s", "目标锁定估计允许的最大速度。", "高速移动目标更不容易丢。", "异常跳变更容易被拦住。", "高速目标跟不上时调高。"),
    FriendlyParam("jump_accept", "目标锁定", "同目标跳变接受", "control.same_lock_jump_accept_px", "int", 60, 260, 5, "px", "同一个锁定目标发生位置跳变时，允许直接接受的距离。", "更不容易卡顿断开。", "更保守。", "卡一卡时可适度调高。"),
    FriendlyParam("jump_accept_on", "目标锁定", "启用同目标跳变接受", "control.same_lock_jump_accept_enabled", "bool", description="同一目标短时位置跳变时，不立即断开控制。", lower_tip="更保守，但可能一卡一卡。", higher_tip="更连续。", risk="建议开启。"),

    # 跟随移动
    FriendlyParam("speed", "跟随移动", "移动速度", "control.sensitivity_scaler", "float", 0.50, 1.50, 0.01, "×", "整体移动强度。", "更慢、更稳。", "更快拉向目标。", "建议 0.85～1.10。"),
    FriendlyParam("max_move", "跟随移动", "单次最大移动", "control.max_move", "int", 6, 40, 1, "px", "每次提交允许的最大移动量。", "动作更细。", "大距离更快。", "过高会粗糙。"),
    FriendlyParam("max_step", "跟随移动", "单步最大值", "control.max_step", "int", 6, 40, 1, "px", "单次输出的另一个上限。", "更细腻。", "更快。", "通常接近单次最大移动。"),
    FriendlyParam("deadzone", "跟随移动", "中心稳定区", "control.deadzone", "float", 0.0, 8.0, 0.1, "px", "误差很小时停止修正的范围。", "更精细但可能抖。", "更稳但可能停在目标旁。", "建议 1.5～4.5。"),
    FriendlyParam("fine_deadzone", "跟随移动", "精细稳定区", "control.fine_deadzone", "float", 0.0, 6.0, 0.1, "px", "更靠近中心时的细微停止范围。", "更积极。", "更安静。", "建议 0.8～2.6。"),
    FriendlyParam("smooth", "跟随移动", "锁定平滑度", "control.locked_smooth_alpha", "float", 0.40, 0.90, 0.01, "", "锁定目标移动时的平滑程度。", "跟手更快。", "更顺滑但略慢。", "建议 0.60～0.75。"),
    FriendlyParam("slew", "跟随移动", "锁定移动限速", "control.locked_slew_px_per_frame", "float", 8, 38, 1, "px/帧", "锁定目标每帧允许追踪的最大变化量。", "更柔和。", "更跟手。", "卡顿时可适当调高。"),
    FriendlyParam("residual", "跟随移动", "残差补偿", "control.residual_error_fraction", "float", 0.30, 0.90, 0.01, "", "把未完成误差平滑补进去的比例。", "更保守。", "更快追平误差。", "过高可能粗糙。"),
    FriendlyParam("natural_alpha", "跟随移动", "自然移动平滑", "control.natural_motion_alpha", "float", 0.30, 0.85, 0.01, "", "控制移动增量变化的平滑程度。", "更灵敏。", "更平滑。", "建议 0.50～0.70。"),
    FriendlyParam("near_damping", "跟随移动", "近中心减速范围", "control.near_center_damping_px", "float", 5, 60, 1, "px", "接近中心后开始减速的范围。", "更晚减速。", "更早减速、更稳。", "停顿明显时不要过大。"),
    FriendlyParam("near_scale", "跟随移动", "近中心减速强度", "control.near_center_damping_scale", "float", 0.02, 0.30, 0.01, "", "接近中心后保留多少移动强度。", "中心附近更慢。", "中心附近更积极。", "太低会感觉卡。"),

    # 稳定抑制
    FriendlyParam("settle_on", "稳定抑制", "启用稳定锁", "control.settle_lock_enabled", "bool", description="靠近中心后进入稳定状态，减少抖动。", lower_tip="更跟手但可能抖。", higher_tip="更稳。", risk="如果靠近目标后不动，可临时关闭测试。"),
    FriendlyParam("settle_enter", "稳定抑制", "稳定进入范围", "control.settle_enter_px", "float", 1, 12, 0.1, "px", "误差小于该值后进入稳定状态。", "更难进入稳定，更积极。", "更容易稳定。", "过大可能提前停。"),
    FriendlyParam("settle_exit", "稳定抑制", "稳定退出范围", "control.settle_exit_px", "float", 4, 40, 0.5, "px", "误差超过该值后退出稳定状态。", "更容易重新移动。", "更稳但可能滞后。", "建议大于稳定进入范围。"),
    FriendlyParam("stale_min", "稳定抑制", "短时断检容忍", "control.stale_target_min_seconds", "float", 0.02, 0.16, 0.005, "s", "目标短暂不更新时仍允许使用的最短时间。", "更快判定过期。", "更不容易断续。", "过高会用旧目标。"),
    FriendlyParam("stale_max", "稳定抑制", "最大断检容忍", "control.stale_target_max_seconds", "float", 0.05, 0.30, 0.005, "s", "目标不更新时可容忍的最长时间。", "更保守。", "更连续。", "建议 0.10～0.18。"),
    FriendlyParam("soft_hold", "稳定抑制", "无目标软保持", "control.no_target_soft_hold_enabled", "bool", description="目标短暂消失时保持很短时间的连续感。", lower_tip="更干净。", higher_tip="更少断续。", risk="建议开启，但保持时间不要太长。"),
    FriendlyParam("soft_hold_ms", "稳定抑制", "软保持时间", "control.no_target_soft_hold_ms", "float", 0, 120, 5, "ms", "短时无目标时保持移动轮廓的时间。", "更快释放。", "更连续。", "过长可能拖影。"),

    # 自动触发
    FriendlyParam("fire_enabled", "自动触发", "启用自动触发", "control.fire_enabled", "bool", description="目标稳定接近中心时是否允许自动触发。", lower_tip="关闭后完全不触发。", higher_tip="开启后仍受半径、冷却、稳定性限制。", risk="发行版建议默认关闭，由用户自行开启。"),
    FriendlyParam("fire_radius", "自动触发", "触发半径", "control.fire_radius", "float", 2.0, 16.0, 0.1, "px", "目标离中心多近时允许触发。", "更精准。", "更容易触发。", "建议 5～8。"),
    FriendlyParam("fire_exit", "自动触发", "离开半径", "control.fire_exit_radius", "float", 2.0, 22.0, 0.1, "px", "离开该范围后自动触发状态才复位。", "复位更快。", "滞回更稳。", "必须大于触发半径。"),
    FriendlyParam("fire_rearm", "自动触发", "重新触发半径", "control.fire_rearm_radius", "float", 3.0, 28.0, 0.1, "px", "重复触发前需要离开多远再重新进入。", "更容易再次触发。", "更克制。", "建议大于离开半径。"),
    FriendlyParam("fire_cooldown", "自动触发", "触发冷却", "control.fire_cooldown_ms", "int", 60, 600, 5, "ms", "两次触发的最短间隔。", "更频繁。", "更克制。", "建议 130～220ms。"),
    FriendlyParam("fire_stable", "自动触发", "稳定确认", "control.fire_stable_frames", "int", 1, 6, 1, "帧", "目标进入半径后需要稳定多少帧才触发。", "更快。", "更稳。", "建议 2～3。"),
    FriendlyParam("fire_conf", "自动触发", "触发最低可信度", "control.fire_min_conf", "float", 0.20, 0.80, 0.01, "", "自动触发需要的最低目标可信度。", "更容易触发。", "更稳。", "建议 0.40～0.55。"),
    FriendlyParam("fire_zero", "自动触发", "只在稳定时触发", "control.fire_require_zero_motion", "bool", description="只有移动债务足够小、目标足够稳定时才允许触发。", lower_tip="更容易触发。", higher_tip="更稳。", risk="建议开启。"),
    FriendlyParam("fire_repeat", "自动触发", "半径内重复触发", "control.fire_repeat_while_in_radius", "bool", description="目标一直在半径内时是否允许重复触发。", lower_tip="更克制。", higher_tip="更频繁。", risk="普通用户建议关闭。"),
    FriendlyParam("fire_held", "自动触发", "允许短时保持目标触发", "control.fire_allow_held_target", "bool", description="目标短时断检但仍被系统保持时，是否允许自动触发。", lower_tip="更安全。", higher_tip="断检时也能触发，但风险更高。", risk="建议关闭。"),
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "稳妥推荐": {
        "model.conf": 0.20,
        "model.max_candidates": 180,
        "control.sensitivity_scaler": 0.99,
        "control.deadzone": 2.0,
        "control.fine_deadzone": 1.0,
        "control.fire_enabled": False,
        "control.fire_radius": 6.5,
        "control.fire_cooldown_ms": 155,
        "control.fire_stable_frames": 2,
        "control.fire_require_zero_motion": True,
    },
    "更快响应": {
        "model.conf": 0.18,
        "model.max_candidates": 240,
        "control.reactive_fast_enter_min_conf": 0.30,
        "control.reactive_fast_enter_min_body_conf": 0.30,
        "control.sensitivity_scaler": 1.04,
        "control.max_move": 22,
    },
    "更少误检": {
        "model.conf": 0.24,
        "detection_filter.head_only_min_conf": 0.92,
        "detection_filter.small_head_only_min_conf": 0.94,
        "control.reactive_fast_enter_min_conf": 0.58,
        "control.fire_min_conf": 0.50,
    },
    "自动触发稳妥": {
        "control.fire_enabled": True,
        "control.fire_radius": 6.5,
        "control.fire_exit_radius": 8.5,
        "control.fire_rearm_radius": 11.0,
        "control.fire_cooldown_ms": 180,
        "control.fire_repeat_while_in_radius": False,
        "control.fire_require_zero_motion": True,
        "control.fire_stable_frames": 2,
        "control.fire_allow_held_target": False,
    },
}


class VisionForgeApp((ctk.CTk if ctk else tk.Tk)):  # type: ignore[misc]
    def __init__(self) -> None:
        if ctk:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("VisionForge")
        self.geometry("1320x840")
        self.minsize(1120, 720)
        self.proc: Optional[subprocess.Popen[str]] = None
        self.thread_mode = False
        self.rt_thread: Optional[threading.Thread] = None
        self.rt_stop_event: Optional[threading.Event] = None
        self.proc_queue: "queue.Queue[str]" = queue.Queue()
        self.cfg: Dict[str, Any] = {}
        self.license_status: LicenseStatus = load_license()
        self.machine_code_value = machine_code()
        self.param_widgets: Dict[str, Dict[str, Any]] = {}
        self.current_group = "识别"
        self.current_page = "home"
        self.diag_text_cache = ""
        self._slider_hover_widget: Any = None
        self._wheel_guard_sliders: List[Any] = []
        self._last_diag_items: Any = None
        self._last_diag_ts: float = 0.0
        self._restart_in_progress = False
        self._init_colors()
        self._setup_window_icon()
        self._setup_ttk()
        self._build_ui()
        self.reload_config(show_error=False)
        self.refresh_license_status(show_popup=False)
        self.after(350, self._poll_proc_queue)
        self.after(2500, self._refresh_latest_log_light)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_colors(self) -> None:
        self.bg = "#07111f"
        self.side = "#08111f"
        self.surface = "#0f172a"
        self.surface2 = "#111827"
        self.fg = "#f8fafc"
        self.muted = "#94a3b8"
        self.accent = "#38bdf8"
        self.green = "#22c55e"
        self.warn = "#f59e0b"
        self.red = "#ef4444"
        try:
            self.configure(bg=self.bg)
        except Exception:
            pass

    def _setup_window_icon(self) -> None:
        try:
            ico = icon_path()
            if ico.exists():
                self.iconbitmap(str(ico))
        except Exception:
            pass

    def _setup_ttk(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Horizontal.TScale", background=self.surface, troughcolor="#1f2937")

    def F(self, parent: Any, color: Optional[str] = None, radius: int = 18, **kw: Any) -> Any:
        if ctk:
            return ctk.CTkFrame(parent, fg_color=color or self.bg, corner_radius=radius, **kw)
        return tk.Frame(parent, bg=color or self.bg, **kw)

    def L(self, parent: Any, text: str = "", color: Optional[str] = None, font: Any = None, **kw: Any) -> Any:
        if ctk:
            return ctk.CTkLabel(parent, text=text, text_color=color or self.fg, font=font or ("Microsoft YaHei UI", 11), **kw)
        return tk.Label(parent, text=text, fg=color or self.fg, bg=kw.pop("fg_color", self.bg), font=font or ("Microsoft YaHei UI", 11), **kw)

    def B(self, parent: Any, text: str, command: Any, **kw: Any) -> Any:
        if ctk:
            kw.setdefault("corner_radius", 12)
            kw.setdefault("height", 38)
            return ctk.CTkButton(parent, text=text, command=command, **kw)
        kw.pop("fg_color", None)
        kw.pop("text_color", None)
        kw.pop("corner_radius", None)
        # CustomTkinter uses pixel height; tkinter uses text rows. Avoid giant fallback buttons.
        if isinstance(kw.get("height"), int) and kw.get("height", 0) > 5:
            kw.pop("height", None)
        return tk.Button(parent, text=text, command=command, **kw)

    def T(self, parent: Any, **kw: Any) -> Any:
        if ctk:
            kw.setdefault("corner_radius", 14)
            return ctk.CTkTextbox(parent, **kw)
        return tk.Text(parent, bg="#020617", fg="#d1d5db", insertbackground="#ffffff", **kw)

    def textbox_set(self, box: Any, text: str) -> None:
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text)
            box.configure(state="disabled")
        except Exception:
            pass

    def textbox_append(self, box: Any, text: str) -> None:
        try:
            box.configure(state="normal")
            box.insert("end", text)
            box.see("end")
            box.configure(state="disabled")
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = self.F(self, self.side, radius=0, width=230)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.content = self.F(self, self.bg, radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self._build_sidebar()
        self._build_header()
        self.page_host = self.F(self.content, self.bg)
        self.page_host.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 24))
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)
        self.pages: Dict[str, Any] = {}
        for name in ["home", "license", "run", "tune", "env", "release"]:
            f = self.F(self.page_host, self.bg)
            f.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = f
        self._build_home_page()
        self._build_license_page()
        self._build_run_page()
        self._build_tune_page()
        self._build_env_page()
        self._build_release_page()
        self.show_page("home")

    def _build_sidebar(self) -> None:
        self.L(self.sidebar, "VisionForge", "#ffffff", ("Microsoft YaHei UI", 24, "bold"), fg_color=self.side).pack(anchor="w", padx=24, pady=(30, 2))
        self.L(self.sidebar, "视觉运行助手", self.muted, ("Microsoft YaHei UI", 11), fg_color=self.side).pack(anchor="w", padx=24, pady=(0, 24))
        self.nav_buttons: Dict[str, Any] = {}
        for key, text in [("home", "总览"), ("license", "授权"), ("run", "启动"), ("tune", "调节"), ("env", "环境"), ("release", "更新")]:
            btn = self.B(self.sidebar, text, lambda k=key: self.show_page(k), anchor="w", fg_color="#111827" if ctk else None)
            btn.pack(fill="x", padx=18, pady=5)
            self.nav_buttons[key] = btn
        self.L(self.sidebar, f"{VERSION}\n本机数据目录已自动维护", self.muted, ("Microsoft YaHei UI", 9), fg_color=self.side, justify="left", wraplength=185).pack(side="bottom", anchor="w", padx=24, pady=24)

    def _build_header(self) -> None:
        h = self.F(self.content, self.bg)
        h.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 14))
        h.grid_columnconfigure(0, weight=1)
        self.page_title = self.L(h, "总览", "#ffffff", ("Microsoft YaHei UI", 25, "bold"), fg_color=self.bg)
        self.page_title.grid(row=0, column=0, sticky="w")
        self.page_subtitle = self.L(h, "关键状态和常用操作。", self.muted, ("Microsoft YaHei UI", 11), fg_color=self.bg)
        self.page_subtitle.grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.header_status = self.L(h, "未运行", self.muted, ("Microsoft YaHei UI", 13, "bold"), fg_color=self.bg)
        self.header_status.grid(row=0, column=1, sticky="e")

    def show_page(self, name: str) -> None:
        self.current_page = name
        title_map = {"home": "总览", "license": "授权", "run": "启动", "tune": "可视化调参", "env": "环境助手", "release": "版本更新"}
        sub_map = {
            "home": "关键状态和常用操作。",
            "license": "输入卡密即可激活。",
            "run": "一键启动或停止 VisionForge。",
            "tune": "面向普通用户的中文滑块和开关。",
            "env": "用清晰卡片显示本机环境状态。",
            "release": "本次版本修复内容和发行说明。",
        }
        self.page_title.configure(text=title_map.get(name, name))
        self.page_subtitle.configure(text=sub_map.get(name, ""))
        for key, btn in self.nav_buttons.items():
            if ctk:
                btn.configure(fg_color="#075985" if key == name else "#111827")
        self.pages[name].tkraise()

    def _card(self, parent: Any, row: int, col: int, title: str, var: tk.StringVar) -> None:
        f = self.F(parent, self.surface2, radius=22)
        f.grid(row=row, column=col, sticky="nsew", padx=9, pady=9)
        self.L(f, title, self.muted, ("Microsoft YaHei UI", 11), fg_color=self.surface2).pack(anchor="w", padx=18, pady=(17, 5))
        self.L(f, textvariable=var, color="#ffffff", font=("Microsoft YaHei UI", 15, "bold"), fg_color=self.surface2, wraplength=210, justify="left").pack(anchor="w", padx=18, pady=(0, 18))

    def _build_home_page(self) -> None:
        f = self.pages["home"]
        for i in range(4):
            f.grid_columnconfigure(i, weight=1)
        self.home_vars = {k: tk.StringVar(value="-") for k in ["license", "model", "run", "fire"]}
        self._card(f, 0, 0, "授权状态", self.home_vars["license"])
        self._card(f, 0, 1, "模型状态", self.home_vars["model"])
        self._card(f, 0, 2, "运行状态", self.home_vars["run"])
        self._card(f, 0, 3, "自动触发", self.home_vars["fire"])
        actions = self.F(f, self.surface, radius=22)
        actions.grid(row=1, column=0, columnspan=4, sticky="ew", padx=9, pady=12)
        self.B(actions, "启动 VisionForge", self.start_realtime, width=160).pack(side="left", padx=(18, 8), pady=18)
        self.B(actions, "停止", self.stop_realtime, width=110).pack(side="left", padx=8, pady=18)
        self.B(actions, "环境检查", lambda: [self.show_page("env"), self.run_diagnostics()], width=130).pack(side="left", padx=8, pady=18)
        self.B(actions, "打开数据目录", lambda: self.open_path(user_data_dir()), width=140).pack(side="left", padx=8, pady=18)
        self.home_panel = self.F(f, self.surface2, radius=22)
        self.home_panel.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=9, pady=9)
        self.home_panel.grid_columnconfigure(0, weight=1)
        self.home_notice_var = tk.StringVar(value="")
        self.L(self.home_panel, "使用流程", "#ffffff", ("Microsoft YaHei UI", 18, "bold"), fg_color=self.surface2).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 8))
        self.L(self.home_panel, textvariable=self.home_notice_var, color="#e2e8f0", font=("Microsoft YaHei UI", 13), fg_color=self.surface2, wraplength=1050, justify="left").grid(row=1, column=0, sticky="w", padx=22, pady=(0, 18))
        self.L(self.home_panel, "本机数据目录", "#ffffff", ("Microsoft YaHei UI", 15, "bold"), fg_color=self.surface2).grid(row=2, column=0, sticky="w", padx=22, pady=(8, 4))
        self.L(self.home_panel, str(user_data_dir()), "#67e8f9", ("Microsoft YaHei UI", 13, "bold"), fg_color=self.surface2, wraplength=1050, justify="left").grid(row=3, column=0, sticky="w", padx=22, pady=(0, 22))
        f.grid_rowconfigure(2, weight=1)

    def _build_license_page(self) -> None:
        f = self.pages["license"]
        f.grid_columnconfigure(0, weight=3)
        f.grid_columnconfigure(1, weight=2)
        f.grid_rowconfigure(1, weight=1)
        top = self.F(f, self.surface, radius=22)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        top.grid_columnconfigure(0, weight=1)
        self.license_summary_var = tk.StringVar(value="-")
        self.L(top, textvariable=self.license_summary_var, color="#ffffff", font=("Microsoft YaHei UI", 18, "bold"), fg_color=self.surface).grid(row=0, column=0, sticky="w", padx=22, pady=(18, 4))
        self.machine_var = tk.StringVar(value="")
        self.L(top, textvariable=self.machine_var, color="#67e8f9", font=("Microsoft YaHei UI", 12), fg_color=self.surface).grid(row=1, column=0, sticky="w", padx=22, pady=(0, 18))
        self.B(top, "复制机器码", self.copy_machine_code, width=150, height=42).grid(row=0, column=1, rowspan=2, padx=22)

        left = self.F(f, self.surface, radius=22)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)
        self.L(left, "输入卡密", "#ffffff", ("Microsoft YaHei UI", 16, "bold"), fg_color=self.surface).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 8))
        self.license_input = self.T(left, height=140, font=("Microsoft YaHei UI", 12), wrap="word")
        self.license_input.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 16))
        btnrow = self.F(left, self.surface)
        btnrow.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 22))
        self.B(btnrow, "激活", self.activate_license, width=120, height=40).pack(side="left", padx=(0, 10))
        self.B(btnrow, "刷新状态", lambda: self.refresh_license_status(True), width=120, height=40).pack(side="left", padx=10)

        right = self.F(f, self.surface2, radius=22)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        self.L(right, "如何激活", "#ffffff", ("Microsoft YaHei UI", 18, "bold"), fg_color=self.surface2).pack(anchor="w", padx=24, pady=(24, 10))
        self.license_hint = self.L(right, "", "#e2e8f0", ("Microsoft YaHei UI", 13), fg_color=self.surface2, justify="left", wraplength=430)
        self.license_hint.pack(anchor="w", padx=24, pady=(0, 18))
        self.L(right, "本机数据目录", "#ffffff", ("Microsoft YaHei UI", 15, "bold"), fg_color=self.surface2).pack(anchor="w", padx=24, pady=(12, 6))
        self.L(right, str(user_data_dir()), "#67e8f9", ("Microsoft YaHei UI", 12), fg_color=self.surface2, justify="left", wraplength=430).pack(anchor="w", padx=24, pady=(0, 14))
        self.B(right, "打开数据目录", lambda: self.open_path(user_data_dir()), width=150).pack(anchor="w", padx=24, pady=(0, 22))

    def _build_run_page(self) -> None:
        f = self.pages["run"]
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(3, weight=1)
        panel = self.F(f, self.surface, radius=24)
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        panel.grid_columnconfigure(0, weight=1)
        self.run_state_var = tk.StringVar(value="未启动")
        self.run_message_var = tk.StringVar(value="准备就绪。点击右侧按钮即可启动。")
        self.L(panel, textvariable=self.run_state_var, color="#ffffff", font=("Microsoft YaHei UI", 21, "bold"), fg_color=self.surface).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 4))
        self.L(panel, textvariable=self.run_message_var, color="#cbd5e1", font=("Microsoft YaHei UI", 13), fg_color=self.surface).grid(row=1, column=0, sticky="w", padx=22, pady=(0, 20))
        self.B(panel, "启动", self.start_realtime, width=132, height=44).grid(row=0, column=1, rowspan=2, padx=(8, 22), pady=20)
        self.B(panel, "停止", self.stop_realtime, width=132, height=44).grid(row=0, column=2, rowspan=2, padx=(8, 22), pady=20)

        tips = self.F(f, self.surface2, radius=22)
        tips.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        tips.grid_columnconfigure(0, weight=1)
        self.L(tips, "运行提示", "#ffffff", ("Microsoft YaHei UI", 16, "bold"), fg_color=self.surface2).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))
        self.L(tips, "界面只显示必要状态。详细技术日志会自动写入本机日志目录，不直接展示给普通用户。", "#f8fafc", ("Microsoft YaHei UI", 13), fg_color=self.surface2, wraplength=980, justify="left").grid(row=1, column=0, sticky="w", padx=20, pady=(0, 16))

        data = self.F(f, self.surface2, radius=22)
        data.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        data.grid_columnconfigure(0, weight=1)
        self.L(data, "本机数据位置", "#ffffff", ("Microsoft YaHei UI", 15, "bold"), fg_color=self.surface2).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 2))
        self.L(data, str(user_data_dir()), "#67e8f9", ("Microsoft YaHei UI", 13), fg_color=self.surface2, wraplength=960, justify="left").grid(row=1, column=0, sticky="w", padx=20, pady=(0, 16))
        self.B(data, "打开目录", lambda: self.open_path(user_data_dir()), width=130).grid(row=0, column=1, rowspan=2, padx=20, pady=18)
        self.run_output = None

    def _build_tune_page(self) -> None:
        f = self.pages["tune"]
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        top = self.F(f, self.surface, radius=22)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for name in ["识别", "目标锁定", "跟随移动", "稳定抑制", "自动触发"]:
            self.B(top, name, lambda g=name: self.show_param_group(g), width=110).pack(side="left", padx=(18 if name == "识别" else 8, 8), pady=16)
        self.B(top, "保存并备份", self.save_config, width=135).pack(side="right", padx=(8, 18), pady=16)
        self.B(top, "恢复默认", self.restore_default_config, width=110).pack(side="right", padx=8, pady=16)
        self.preset_var = tk.StringVar(value="稳妥推荐")
        if ctk:
            preset = ctk.CTkOptionMenu(top, values=list(PRESETS.keys()), variable=self.preset_var, command=lambda _v: self.apply_preset())
            preset.pack(side="right", padx=8, pady=16)
        else:
            preset = ttk.Combobox(top, values=list(PRESETS.keys()), textvariable=self.preset_var, state="readonly", width=14)
            preset.pack(side="right", padx=8, pady=16)
            preset.bind("<<ComboboxSelected>>", lambda _e: self.apply_preset())
        if ctk:
            self.param_area = ctk.CTkScrollableFrame(f, fg_color=self.bg, corner_radius=0)
        else:
            self.param_area = self.F(f, self.bg)
        self.param_area.grid(row=1, column=0, sticky="nsew")

    def _build_env_page(self) -> None:
        f = self.pages["env"]
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        top = self.F(f, self.surface, radius=24)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        top.grid_columnconfigure(0, weight=1)
        self.env_title_var = tk.StringVar(value="点击开始检查，VisionForge 会自动判断本机运行环境。")
        self.L(top, textvariable=self.env_title_var, color="#ffffff", font=("Microsoft YaHei UI", 17, "bold"), fg_color=self.surface).grid(row=0, column=0, sticky="w", padx=22, pady=(18, 4))
        self.L(top, "检查结果会用卡片展示。详细技术日志只保存在本机日志目录。", color="#cbd5e1", font=("Microsoft YaHei UI", 12), fg_color=self.surface).grid(row=1, column=0, sticky="w", padx=22, pady=(0, 18))
        self.B(top, "开始检查", self.run_diagnostics, width=130, height=42).grid(row=0, column=1, rowspan=2, padx=8, pady=18)
        self.B(top, "复制检查结果", self.copy_diagnostics, width=140, height=42).grid(row=0, column=2, rowspan=2, padx=8, pady=18)
        self.B(top, "打开数据目录", lambda: self.open_path(user_data_dir()), width=150, height=42).grid(row=0, column=3, rowspan=2, padx=(8, 22), pady=18)
        if ctk:
            self.env_area = ctk.CTkScrollableFrame(f, fg_color=self.bg, corner_radius=0)
        else:
            self.env_area = self.F(f, self.bg)
        self.env_area.grid(row=1, column=0, sticky="nsew")
        self._show_env_placeholder()

    def _show_env_placeholder(self) -> None:
        for child in self.env_area.winfo_children():
            child.destroy()
        card = self.F(self.env_area, self.surface2, radius=24)
        card.grid(row=0, column=0, sticky="ew", padx=6, pady=8)
        self.env_area.grid_columnconfigure(0, weight=1)
        self.L(card, "环境状态", "#ffffff", ("Microsoft YaHei UI", 18, "bold"), fg_color=self.surface2).pack(anchor="w", padx=22, pady=(22, 6))
        self.L(card, "点击“开始检查”。如果发现依赖问题，会显示清晰的修复步骤和下载入口。", "#e2e8f0", ("Microsoft YaHei UI", 13), fg_color=self.surface2, wraplength=980, justify="left").pack(anchor="w", padx=22, pady=(0, 22))

    def _build_release_page(self) -> None:
        f = self.pages["release"]
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(0, weight=1)
        text = (
            "VisionForge V17.8.32 更新简介\n\n"
            "1. 修复 GPU 环境检测误判：增强 CUDA/cuDNN/TensorRT 与打包运行库搜索。\n"
            "2. 修复 ONNX Runtime 导入前未注入 DLL 搜索路径导致 GPU Provider 回退的问题。\n"
            "3. 修复运行日志未稳定写入本机日志目录的问题，启动失败也会留下诊断文件。\n"
            "4. 修复环境页滚轮不灵敏：只阻止调参滑块被滚轮误改，不再影响普通页面滚动。\n"
            "5. 优化启动状态提示，运行成功、运行中、停止和失败均使用清晰中文状态。\n"
            "6. 继续保留参数热重载：运行中保存后自动重启运行核心。\n"
            "7. 环境助手继续使用中文卡片与下载入口，不展示开发者控制台输出。\n"
            "8. 保护型构建脚本更新为只构建正式保护型 EXE。"
        )
        box = self.T(f, font=("Microsoft YaHei UI", 13), wrap="word")
        box.grid(row=0, column=0, sticky="nsew")
        self.textbox_set(box, text)

    def show_param_group(self, group: str) -> None:
        self.current_group = group
        for child in self.param_area.winfo_children():
            child.destroy()
        self.param_widgets.clear()
        row = 0
        for spec in FRIENDLY_PARAMS:
            if spec.group != group:
                continue
            self._add_param_row(self.param_area, spec, row)
            row += 1

    def _add_param_row(self, parent: Any, spec: FriendlyParam, row: int) -> None:
        current = get_path(self.cfg, spec.path, False if spec.kind == "bool" else spec.minimum)
        card = self.F(parent, self.surface, radius=20)
        card.grid(row=row, column=0, sticky="ew", padx=6, pady=8)
        parent.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        self.L(card, spec.title, "#ffffff", ("Microsoft YaHei UI", 16, "bold"), fg_color=self.surface).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 3))
        self.L(card, spec.description, "#e2e8f0", ("Microsoft YaHei UI", 12), fg_color=self.surface, wraplength=680, justify="left").grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))
        value_var = tk.StringVar()
        if spec.kind == "bool":
            bool_var = tk.BooleanVar(value=bool(current))
            if ctk:
                switch = ctk.CTkSwitch(card, text="开启" if bool(current) else "关闭", variable=bool_var, command=lambda s=spec, v=bool_var: self._on_bool_param(s, v), font=("Microsoft YaHei UI", 13, "bold"))
                switch.grid(row=0, column=1, rowspan=2, sticky="e", padx=20, pady=18)
            else:
                switch = tk.Checkbutton(card, text="开启", variable=bool_var, command=lambda s=spec, v=bool_var: self._on_bool_param(s, v), bg=self.surface, fg=self.fg, selectcolor=self.surface)
                switch.grid(row=0, column=1, rowspan=2, sticky="e", padx=20, pady=18)
            value_var.set("已开启" if bool(current) else "已关闭")
            self.param_widgets[spec.key] = {"spec": spec, "var": bool_var, "value_var": value_var, "widget": switch}
        else:
            value_var.set(self._format_value(current, spec))
            if ctk:
                slider = ctk.CTkSlider(card, from_=spec.minimum, to=spec.maximum, number_of_steps=max(1, int(round((spec.maximum - spec.minimum) / spec.step))), command=lambda v, s=spec: self._on_slider_param(s, v))
                slider.set(float(current))
            else:
                slider = ttk.Scale(card, from_=spec.minimum, to=spec.maximum, orient="horizontal", command=lambda v, s=spec: self._on_slider_param(s, float(v)))
                slider.set(float(current))
            slider.grid(row=0, column=1, sticky="ew", padx=(10, 20), pady=(18, 4))
            self._bind_slider_wheel_guard(slider)
            self.L(card, textvariable=value_var, color="#67e8f9", font=("Microsoft YaHei UI", 15, "bold"), fg_color=self.surface).grid(row=1, column=1, sticky="e", padx=20, pady=(0, 12))
            self.param_widgets[spec.key] = {"spec": spec, "var": None, "value_var": value_var, "widget": slider}
        tip = f"调低：{spec.lower_tip}\n调高：{spec.higher_tip}\n提示：{spec.risk}"
        self.L(card, tip, "#facc15", ("Microsoft YaHei UI", 12), fg_color=self.surface, wraplength=1080, justify="left").grid(row=2, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 18))

    def _bind_slider_wheel_guard(self, widget: Any) -> None:
        """Prevent mouse wheel from changing sliders, while preserving page scroll.

        The earlier v31 build installed a broad wheel guard that interfered with
        other pages. v32 only captures wheel events that originate from a slider or
        one of its internal CustomTkinter children. The event is redirected to the
        parameter page scroll instead of changing the slider value.
        """
        def block(event: Any) -> str:
            self._scroll_widget_area(getattr(self, "param_area", None), event)
            return "break"
        if not hasattr(self, "_wheel_guard_sliders"):
            self._wheel_guard_sliders = []
        self._wheel_guard_sliders.append(widget)
        targets = [widget, getattr(widget, "_canvas", None), getattr(widget, "_button", None), getattr(widget, "_slider", None), getattr(widget, "_progressbar", None)]
        try:
            targets.extend(list(widget.winfo_children()))
        except Exception:
            pass
        for target in targets:
            if target is None:
                continue
            try:
                target.bind("<MouseWheel>", block, add="+")
                target.bind("<Button-4>", block, add="+")
                target.bind("<Button-5>", block, add="+")
            except Exception:
                pass
        if not getattr(self, "_global_wheel_bound", False):
            try:
                self.bind_all("<MouseWheel>", self._global_wheel_handler, add="+")
                self.bind_all("<Button-4>", self._global_wheel_handler, add="+")
                self.bind_all("<Button-5>", self._global_wheel_handler, add="+")
                self._global_wheel_bound = True
            except Exception:
                pass

    def _is_descendant_widget(self, child: Any, parent: Any) -> bool:
        try:
            w = child
            while w is not None:
                if w == parent:
                    return True
                w = getattr(w, "master", None)
        except Exception:
            return False
        return False

    def _global_wheel_handler(self, event: Any) -> Optional[str]:
        """Keep scrolling usable on every page; only slider widgets are protected."""
        widget = getattr(event, "widget", None)
        try:
            for slider in getattr(self, "_wheel_guard_sliders", []):
                if widget == slider or self._is_descendant_widget(widget, slider):
                    self._scroll_widget_area(getattr(self, "param_area", None), event)
                    return "break"
            if getattr(self, "current_page", "") == "env":
                area = getattr(self, "env_area", None)
                if area is not None and (widget == area or self._is_descendant_widget(widget, area)):
                    self._scroll_widget_area(area, event)
                    return "break"
            if getattr(self, "current_page", "") == "tune":
                area = getattr(self, "param_area", None)
                if area is not None and (widget == area or self._is_descendant_widget(widget, area)):
                    self._scroll_widget_area(area, event)
                    return None
        except Exception:
            pass
        return None

    def _scroll_widget_area(self, area: Any, event: Any) -> None:
        if area is None:
            return
        try:
            delta = getattr(event, "delta", 0)
            if delta:
                units = -1 if delta > 0 else 1
            else:
                units = -1 if getattr(event, "num", 0) == 4 else 1
            canvas = getattr(area, "_parent_canvas", None) or getattr(area, "_canvas", None) or area
            if hasattr(canvas, "yview_scroll"):
                canvas.yview_scroll(units * 4, "units")
            elif hasattr(area, "_parent_canvas") and hasattr(area._parent_canvas, "yview_scroll"):
                area._parent_canvas.yview_scroll(units * 4, "units")
        except Exception:
            pass

    def _scroll_param_page(self, event: Any) -> None:
        self._scroll_widget_area(getattr(self, "param_area", None), event)

    def _format_value(self, value: Any, spec: FriendlyParam) -> str:
        try:
            if spec.kind == "int":
                return f"{int(round(float(value)))}{spec.suffix}"
            return f"{float(value):.2f}{spec.suffix}"
        except Exception:
            return str(value)

    def _on_bool_param(self, spec: FriendlyParam, var: tk.BooleanVar) -> None:
        value = bool(var.get())
        set_path(self.cfg, spec.path, value)
        item = self.param_widgets.get(spec.key, {})
        if "value_var" in item:
            item["value_var"].set("已开启" if value else "已关闭")
        widget = item.get("widget")
        if ctk and widget is not None:
            try:
                widget.configure(text="开启" if value else "关闭")
            except Exception:
                pass
        self._update_cards()

    def _on_slider_param(self, spec: FriendlyParam, raw: float) -> None:
        if spec.kind == "int":
            value: Any = int(round(raw / spec.step) * spec.step)
        else:
            value = round(round(raw / spec.step) * spec.step, 4)
        set_path(self.cfg, spec.path, value)
        item = self.param_widgets.get(spec.key, {})
        if "value_var" in item:
            item["value_var"].set(self._format_value(value, spec))
        self._update_cards()

    def reload_config(self, show_error: bool = True) -> None:
        try:
            self.cfg = load_yaml(CONFIG_PATH)
            self._update_cards()
            self._update_home_text()
            self.show_param_group(self.current_group)
        except Exception as exc:
            if show_error:
                messagebox.showerror("配置加载失败", self._friendly_error(exc))

    def _update_cards(self) -> None:
        if not hasattr(self, "home_vars"):
            return
        c = self.cfg.get("control", {}) or {}
        m = self.cfg.get("model", {}) or {}
        self.home_vars["license"].set(self._license_short())
        self.home_vars["model"].set("已内置" if MODEL_PATH.exists() else "模型缺失")
        running = self.proc is not None and self.proc.poll() is None
        self.home_vars["run"].set("运行中" if running else "未运行")
        self.home_vars["fire"].set("已开启" if bool(c.get("fire_enabled")) else "已关闭")
        if hasattr(self, "run_state_var"):
            self.run_state_var.set("已启动" if running else "未启动")
        if hasattr(self, "header_status"):
            self.header_status.configure(text="运行中" if running else "未运行", text_color=self.green if running else self.muted)

    def _license_display_text(self) -> str:
        if not self.license_status.valid:
            return "未授权"
        if self.license_status.days_left is None:
            return "已授权：永久"
        return f"已授权：剩余 {self.license_status.days_left} 天"

    def _license_short(self) -> str:
        if not self.license_status.valid:
            return "未授权"
        if self.license_status.days_left is None:
            return "永久"
        return f"剩余 {self.license_status.days_left} 天"

    def _update_home_text(self) -> None:
        text = (
            "1. 先完成授权。\n"
            "2. 打开环境助手，确认 GPU 推理为正常。\n"
            "3. 使用可视化调节页调整识别、跟随移动、稳定抑制和自动触发。\n"
            "4. 点击“启动 VisionForge”。保存参数后会自动热重载，无需手动重启。"
        )
        if hasattr(self, "home_notice_var"):
            self.home_notice_var.set(text)

    def refresh_license_status(self, show_popup: bool = False) -> None:
        self.license_status = load_license()
        if hasattr(self, "license_summary_var"):
            self.license_summary_var.set(self._license_display_text())
        if hasattr(self, "machine_var"):
            self.machine_var.set(f"机器码：{self.machine_code_value}")
        if hasattr(self, "license_hint"):
            msg = (
                "1. 点击右上角“复制机器码”。\n"
                "2. 把机器码发给作者。\n"
                "3. 收到卡密后粘贴到左侧输入框。\n"
                "4. 点击“激活”即可完成授权。"
            )
            self.license_hint.configure(text=msg)
        self._update_cards()
        if show_popup:
            messagebox.showinfo("授权状态", self._license_display_text())

    def activate_license(self) -> None:
        try:
            text = self.license_input.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("卡密为空", "请先粘贴卡密。")
                return
            status = save_license(text)
            self.license_status = status
            self.refresh_license_status(show_popup=False)
            if status.valid:
                messagebox.showinfo("激活成功", self._license_display_text())
                self.show_page("home")
            else:
                messagebox.showerror("激活失败", status.reason)
        except Exception as exc:
            messagebox.showerror("激活失败", self._friendly_error(exc))

    def copy_machine_code(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.machine_code_value)
        messagebox.showinfo("已复制", "机器码已复制。")

    def save_config(self) -> None:
        try:
            self.cfg["version"] = VERSION
            was_running = self._is_running()
            backup = save_yaml_atomic(self.cfg, CONFIG_PATH)
            self.reload_config(show_error=False)
            if was_running:
                self.set_run_message("参数已保存，正在自动热重载运行核心...", running=True)
                self._restart_realtime_after_save()
                messagebox.showinfo("已保存", "参数已保存，运行核心已自动重载。")
            else:
                messagebox.showinfo("已保存", f"参数已保存。\n备份：{backup.name}")
        except Exception as exc:
            messagebox.showerror("保存失败", self._friendly_error(exc))

    def _is_running(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        if self.thread_mode and self.rt_thread is not None and self.rt_thread.is_alive():
            return True
        return False

    def set_run_message(self, text: str, running: Optional[bool] = None) -> None:
        if hasattr(self, "run_message_var"):
            self.run_message_var.set(text)
        if hasattr(self, "run_state_var") and running is not None:
            self.run_state_var.set("已启动" if running else "未启动")

    def _restart_realtime_after_save(self) -> None:
        if self._restart_in_progress:
            return
        self._restart_in_progress = True
        def worker() -> None:
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=4)
                    except Exception:
                        try:
                            self.proc.kill()
                        except Exception:
                            pass
                if self.thread_mode and self.rt_stop_event is not None:
                    self.rt_stop_event.set()
                    if self.rt_thread is not None:
                        self.rt_thread.join(timeout=5)
                self.proc = None
                self.thread_mode = False
                self.rt_thread = None
                self.rt_stop_event = None
                time.sleep(0.35)
                self.after(0, self.start_realtime)
            finally:
                self._restart_in_progress = False
        threading.Thread(target=worker, daemon=True).start()

    def restore_default_config(self) -> None:
        if not messagebox.askyesno("恢复默认", "确认恢复到当前版本推荐默认参数？旧配置会先备份。"):
            return
        try:
            from src.app_paths import default_config_path
            src = default_config_path()
            data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError("默认配置格式错误")
            save_yaml_atomic(data, CONFIG_PATH)
            self.reload_config(show_error=True)
        except Exception as exc:
            messagebox.showerror("恢复失败", self._friendly_error(exc))

    def apply_preset(self) -> None:
        name = self.preset_var.get()
        for path, value in PRESETS.get(name, {}).items():
            set_path(self.cfg, path, value)
        self.show_param_group(self.current_group)
        self._update_cards()

    def _realtime_command(self) -> Optional[List[str]]:
        args = ["--run-realtime", "--config", str(CONFIG_PATH), "--source", "screen", "--control", "on", "--visual", "off", "--profile", "on", "--threaded-capture", "on"]
        if getattr(sys, "frozen", False):
            candidates = [Path(sys.executable), Path(sys.argv[0])]
            for p in candidates:
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if rp.exists() and rp.is_file():
                    return [str(rp)] + args
            return None
        return [sys.executable, str(Path(__file__).resolve())] + args

    def start_realtime(self) -> None:
        self.show_page("run")
        self.refresh_license_status(show_popup=False)
        if not self.license_status.valid:
            self.show_page("license")
            messagebox.showerror("未授权", "请先激活卡密。")
            return
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("已在运行", "VisionForge 已经启动。")
            return
        try:
            ensure_runtime_layout()
            if not CONFIG_PATH.exists():
                raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")
            # Run realtime inside the GUI process instead of launching the one-file EXE again.
            # This avoids repeated onefile extraction, PyInstaller temporary-directory warnings,
            # and duplicated screen-capture startup. Developer logs still go to LOG_DIR.
            self._write_gui_event("用户点击启动；准备启动运行核心。")
            self.rt_stop_event = threading.Event()
            self.thread_mode = True
            self.proc = None
            self.rt_thread = threading.Thread(target=self._run_realtime_thread_fallback, daemon=True)
            self.rt_thread.start()
            self.set_run_message("✅ 启动成功，正在使用 GPU 推理。", running=True)
        except Exception as exc:
            self._write_gui_error("start_realtime", exc)
            messagebox.showerror("启动失败", "VisionForge 启动失败。请先打开“环境检查”，根据提示修复依赖；详细错误已写入日志目录。")
            self.set_run_message("启动失败。请打开环境助手查看修复建议。", running=False)
        self._update_cards()

    def _run_realtime_thread_fallback(self) -> None:
        try:
            self._write_gui_event("运行核心线程已创建。")
            rc = run_realtime_from_gui(["--run-realtime", "--config", str(CONFIG_PATH), "--source", "screen", "--control", "on", "--visual", "off", "--profile", "on", "--threaded-capture", "on"], stop_event=self.rt_stop_event)
            self._write_gui_event(f"运行核心退出，退出码={rc}。")
            self.proc_queue.put("VisionForge 已停止。\n" if rc == 0 else "运行未能启动，请打开环境助手查看修复建议。\n")
        except Exception as exc:
            self._write_gui_error("thread_fallback", exc)
            self.proc_queue.put("运行失败。详细错误已保存到日志目录。\n")
        finally:
            self.thread_mode = False

    def _read_proc_output(self) -> None:
        assert self.proc is not None
        captured: List[str] = []
        if self.proc.stdout:
            for line in self.proc.stdout:
                # Keep developer output out of the GUI. Full logs stay in the logs directory.
                captured.append(line)
        rc = self.proc.wait()
        if captured:
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                p = LOG_DIR / f"runtime_output_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                p.write_text("".join(captured), encoding="utf-8")
            except Exception:
                pass
        if rc == 0:
            self.proc_queue.put("VisionForge 已停止。\n")
        else:
            self.proc_queue.put("运行未能启动，请打开环境助手查看修复建议。\n")

    def _poll_proc_queue(self) -> None:
        try:
            while True:
                self.set_run_message(self.proc_queue.get_nowait().strip(), running=self._is_running())
        except queue.Empty:
            pass
        self._update_cards()
        self.after(350, self._poll_proc_queue)

    def stop_realtime(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.set_run_message("正在停止...", running=True)
            except Exception as exc:
                messagebox.showerror("停止失败", self._friendly_error(exc))
        elif self.thread_mode and self.rt_stop_event is not None:
            self.rt_stop_event.set()
            self.set_run_message("正在停止...", running=True)
        else:
            self.set_run_message("当前未运行。", running=False)
        self._update_cards()

    def run_diagnostics(self) -> None:
        self.env_title_var.set("正在检查，请稍候...")
        for child in self.env_area.winfo_children():
            child.destroy()
        loading = self.F(self.env_area, self.surface2, radius=24)
        loading.grid(row=0, column=0, sticky="ew", padx=6, pady=8)
        self.env_area.grid_columnconfigure(0, weight=1)
        self.L(loading, "正在读取本机环境", "#ffffff", ("Microsoft YaHei UI", 18, "bold"), fg_color=self.surface2).pack(anchor="w", padx=22, pady=(22, 6))
        self.L(loading, "正在检查显卡、运行库、ONNX 推理组件和模型文件。", "#e2e8f0", ("Microsoft YaHei UI", 13), fg_color=self.surface2).pack(anchor="w", padx=22, pady=(0, 22))
        threading.Thread(target=self._run_diagnostics_worker, daemon=True).start()

    def _run_diagnostics_worker(self) -> None:
        try:
            if collect_diagnostics is None:
                raise RuntimeError("诊断模块未加载")
            items = collect_diagnostics()
            detail_text = summarize(items) if summarize else ""
            self.after(0, lambda: self._show_diagnostics_items(items, detail_text))
        except Exception as exc:
            self._write_gui_error("diagnostics", exc)
            self.after(0, lambda: self._show_diagnostics_error())

    def _status_color(self, status: str) -> str:
        return {"OK": self.green, "WARN": self.warn, "ERROR": self.red, "INFO": self.accent}.get(status, self.muted)

    def _status_cn(self, status: str) -> str:
        return {"OK": "正常", "WARN": "需注意", "ERROR": "需修复", "INFO": "信息"}.get(status, status)

    def _show_diagnostics_error(self) -> None:
        self.env_title_var.set("环境检查未完成")
        for child in self.env_area.winfo_children():
            child.destroy()
        card = self.F(self.env_area, self.surface2, radius=24)
        card.grid(row=0, column=0, sticky="ew", padx=6, pady=8)
        self.L(card, "检查失败", self.red, ("Microsoft YaHei UI", 18, "bold"), fg_color=self.surface2).pack(anchor="w", padx=22, pady=(22, 6))
        self.L(card, f"检查过程遇到问题，详细信息已保存到：{LOG_DIR}", "#e2e8f0", ("Microsoft YaHei UI", 13), fg_color=self.surface2, wraplength=1000, justify="left").pack(anchor="w", padx=22, pady=(0, 22))

    def _show_diagnostics_items(self, items: Any, detail_text: str) -> None:
        self.diag_text_cache = detail_text
        for child in self.env_area.winfo_children():
            child.destroy()
        errors = sum(1 for i in items if getattr(i, "status", "") == "ERROR")
        warns = sum(1 for i in items if getattr(i, "status", "") == "WARN")
        if errors:
            self.env_title_var.set("发现需要修复的项目")
        elif warns:
            self.env_title_var.set("环境基本可用，有项目需要注意")
        else:
            self.env_title_var.set("环境检查通过")
        preferred = ["显卡", "GPU 推理", "TensorRT 加速", "CUDA 运行库", "cuDNN", "cuBLAS", "Microsoft VC++ 运行库", "ONNX Runtime", "屏幕采集组件", "系统信息组件", "模型文件"]
        by_name = {getattr(i, "name", ""): i for i in items}
        ordered = [by_name[n] for n in preferred if n in by_name] + [i for i in items if getattr(i, "name", "") not in preferred and getattr(i, "status", "") in {"ERROR", "WARN"}]
        row = 0
        for item in ordered:
            self._add_env_card(item, row)
            row += 1
        if not ordered:
            self._show_env_placeholder()

    def _add_env_card(self, item: Any, row: int) -> None:
        card = self.F(self.env_area, self.surface2, radius=22)
        card.grid(row=row, column=0, sticky="ew", padx=6, pady=8)
        self.env_area.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)
        status = getattr(item, "status", "INFO")
        name = getattr(item, "name", "环境项目")
        hint = getattr(item, "hint", "") or getattr(item, "detail", "") or ""
        color = self._status_color(status)
        self.L(card, f"{name}：{self._status_cn(status)}", color, ("Microsoft YaHei UI", 17, "bold"), fg_color=self.surface2).grid(row=0, column=0, sticky="w", padx=22, pady=(18, 4))
        self.L(card, hint, "#f8fafc", ("Microsoft YaHei UI", 14), fg_color=self.surface2, wraplength=920, justify="left").grid(row=1, column=0, sticky="w", padx=22, pady=(0, 16))
        link_url = getattr(item, "link_url", "")
        link_label = getattr(item, "link_label", "") or "打开官方修复入口"
        if link_url and status in {"WARN", "ERROR"}:
            self.B(card, link_label, lambda url=link_url: webbrowser.open(url), width=150, height=36).grid(row=0, column=1, rowspan=2, padx=22, pady=18)

    def copy_diagnostics(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.diag_text_cache or "尚未生成环境检查摘要。")
        messagebox.showinfo("已复制", "检查结果已复制。")

    def _refresh_latest_log_light(self) -> None:
        # Intentionally not showing raw logs on the main UI. They remain available in AppData for diagnosis.
        self.after(3000, self._refresh_latest_log_light)

    def open_path(self, p: Path) -> None:
        try:
            if p.suffix == "":
                p.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as exc:
            messagebox.showerror("打开失败", self._friendly_error(exc))

    def _friendly_error(self, exc: BaseException) -> str:
        if isinstance(exc, FileNotFoundError):
            return "系统找不到需要的文件。请先在“环境检查”页面检查依赖；如果是首次运行，请关闭后重新打开程序。"
        return repr(exc)

    def _write_gui_event(self, message: str) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = LOG_DIR / f"gui_event_{_dt.datetime.now().strftime('%Y%m%d')}.txt"
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception:
            pass

    def _write_gui_error(self, label: str, exc: Any) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = LOG_DIR / f"gui_error_{label}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            path.write_text(str(exc) + "\n\n" + traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass

    def on_close(self) -> None:
        if self._is_running():
            if messagebox.askyesno("退出", "VisionForge 仍在运行，是否停止并退出？"):
                try:
                    if self.proc and self.proc.poll() is None:
                        self.proc.terminate()
                    if self.rt_stop_event is not None:
                        self.rt_stop_event.set()
                except Exception:
                    pass
            else:
                return
        self.destroy()


def run_realtime_from_gui(argv: Optional[List[str]] = None, stop_event: Any = None) -> int:
    # DLL path must be configured before importing main -> onnx_yolo_detector -> onnxruntime.
    configure_dll_search_path()
    import main as realtime_main
    args = list(argv if argv is not None else sys.argv[1:])
    cleaned = [a for a in args if a != "--run-realtime"]
    sys.argv = ["main.py"] + cleaned
    try:
        setattr(realtime_main, "EXTERNAL_STOP_EVENT", stop_event)
    except Exception:
        pass
    try:
        return int(realtime_main.main())
    finally:
        try:
            setattr(realtime_main, "EXTERNAL_STOP_EVENT", None)
        except Exception:
            pass
        try:
            from src.log_utils import close_logging
            close_logging()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--run-realtime", action="store_true", help="internal realtime mode")
    parser.add_argument("--self-test", action="store_true", help="startup smoke test")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--source", default="screen")
    parser.add_argument("--control", default="on")
    parser.add_argument("--visual", default="off")
    parser.add_argument("--profile", default="on")
    parser.add_argument("--threaded-capture", default="on")
    known, _unknown = parser.parse_known_args()
    if known.run_realtime:
        return run_realtime_from_gui(sys.argv[1:])
    if known.self_test:
        ensure_runtime_layout()
        cfg = load_yaml(CONFIG_PATH)
        if not isinstance(cfg, dict):
            raise RuntimeError("config self-test failed")
        st = load_license()
        print(f"VISIONFORGE_SELF_TEST_OK version={VERSION} config={CONFIG_PATH} license_valid={st.valid}")
        return 0
    app = VisionForgeApp()
    app.mainloop()
    return 0


def _write_crash_report(exc: BaseException) -> Path:
    try:
        d = user_data_dir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"crash_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        p.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
        return p
    except Exception:
        return Path.cwd() / "crash_startup.txt"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        report = _write_crash_report(exc)
        try:
            messagebox.showerror("VisionForge 启动失败", f"程序启动失败，已写入错误报告：\n{report}\n\n{exc!r}")
        except Exception:
            pass
        raise
