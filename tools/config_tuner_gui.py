#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local browser-based configuration tuner for the realtime ONNX runtime project.

No framework is required. The server uses Python stdlib http.server and PyYAML,
which is already part of the project requirements.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("缺少 PyYAML。请先运行 pip install -r requirements.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config.default_v17_8_24.yaml"
BACKUP_DIR = PROJECT_ROOT / "config_backups"
LOG_DIR = PROJECT_ROOT / "logs"


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}_{int(time.time() * 1000)}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    data = yaml.safe_load(read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是字典: {path}")
    return data


def dump_yaml(data: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )


def get_path(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(data: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    cur: MutableMapping[str, Any] = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def flatten(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def coerce_value(raw: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on", "y", "是", "开"}
        return bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        if raw == "":
            return 0
        return int(float(raw))
    if isinstance(current, float):
        if raw == "":
            return 0.0
        return float(raw)
    if current is None:
        return raw
    return str(raw) if isinstance(current, str) else raw


@dataclass(frozen=True)
class ParamSpec:
    path: str
    title: str
    group: str
    kind: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""
    description: str = ""
    when: str = ""
    bigger: str = ""
    smaller: str = ""
    danger: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "path": self.path,
            "title": self.title,
            "group": self.group,
            "kind": self.kind,
            "unit": self.unit,
            "description": self.description,
            "when": self.when,
            "bigger": self.bigger,
            "smaller": self.smaller,
            "danger": self.danger,
        }
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.step is not None:
            d["step"] = self.step
        return d


def _p(*args: Any, **kwargs: Any) -> ParamSpec:
    return ParamSpec(*args, **kwargs)


PARAM_SPECS: List[ParamSpec] = [
    _p("model.conf", "模型置信度阈值", "检测召回", "number", 0.10, 0.50, 0.01, "", "ONNX 输出框进入后处理的最低置信度。", "目标一出现识别不到，先小幅降低；误检增多，升高。", "召回下降，误检减少。", "召回提高，误检增加。", "critical"),
    _p("model.iou", "NMS IoU", "检测召回", "number", 0.30, 0.90, 0.01, "", "同类重叠框合并阈值。", "同一个目标框很多或框互相抢时调整。", "保留更多相邻框，可能重复。", "更强合并，可能吃掉近距离双目标。"),
    _p("model.max_candidates", "候选框上限", "检测召回", "number", 50, 600, 10, "个", "后处理最多保留多少候选框。", "多人场景漏目标时加大；CPU 后处理慢时降低。", "多人召回更好，后处理略慢。", "更快，但可能漏框。"),
    _p("selection.head_conf", "选择器 head 阈值", "检测召回", "number", 0.10, 0.60, 0.01, "", "目标选择阶段接受 head 的最低置信度。", "检测框存在但 raw target 为空时降低。", "更稳但更容易空。", "更容易选中目标，也更容易选中弱框。"),
    _p("selection.body_conf", "选择器 body 阈值", "检测召回", "number", 0.10, 0.70, 0.01, "", "目标选择阶段接受 body 的最低置信度。", "body 框弱导致无法配对时降低。", "更稳，弱 body 被过滤。", "更容易配对，误配风险增加。"),
    _p("detection_filter.min_head_conf", "head 基础阈值", "检测召回", "number", 0.10, 0.60, 0.01, "", "几何过滤前 head 的基础置信度下限。", "中远距离头部漏检时降低。", "误检减少，召回降低。", "召回提升，光点误检风险上升。", "critical"),
    _p("detection_filter.min_body_conf", "body 基础阈值", "检测召回", "number", 0.10, 0.70, 0.01, "", "几何过滤前 body 的基础置信度下限。", "有 head 但配不到 body 时降低。", "更保守。", "更容易组成 paired target。"),
    _p("detection_filter.paired_head_min_conf", "配对 head 阈值", "检测召回", "number", 0.10, 0.60, 0.01, "", "有 body 配对时 head 可以使用的较低阈值。", "真实目标有 body 但 head 偏弱时降低。", "假配对减少。", "真实小目标更容易进来。"),
    _p("detection_filter.small_paired_head_min_conf", "小目标配对 head 阈值", "检测召回", "number", 0.10, 0.60, 0.01, "", "小 head 有 body 配对时的最低 head 置信度。", "中远距离人物识别不及时，降低。", "更稳。", "更快召回小目标。", "critical"),
    _p("detection_filter.head_only_min_conf", "孤立 head 阈值", "防误检", "number", 0.50, 0.99, 0.01, "", "没有 body 配对时 head-only 进入候选的阈值。", "不建议轻易降低；地图光点误检多时升高。", "孤立 head 更难进入，防误检更强。", "孤立 head 更容易进入，误检明显增加。", "danger"),
    _p("detection_filter.small_head_only_min_conf", "小孤立 head 阈值", "防误检", "number", 0.50, 0.99, 0.01, "", "小尺寸 head-only 的置信度门槛。", "小光点、小地图结构误检时升高。", "防误检更强。", "小目标召回增加，但风险较高。", "danger"),
    _p("detection_filter.min_body_height_px", "body 最小高度", "几何过滤", "number", 8, 80, 1, "px", "普通 paired target 的 body 最小高度。", "真实中距离人物被过滤时降低；地图短竖线误检时升高。", "误检减少，远小目标可能被杀。", "中远距离召回更好，短结构误检增加。", "critical"),
    _p("detection_filter.small_min_body_height_px", "小目标 body 最小高度", "几何过滤", "number", 8, 80, 1, "px", "小目标 paired target 的 body 最小高度。", "中远距离/露半身目标识别不到时降低。", "更稳但更容易漏小目标。", "召回更强，误检风险增加。", "critical"),
    _p("detection_filter.max_body_aspect", "body 最大宽高比", "几何过滤", "number", 0.50, 2.50, 0.01, "w/h", "body 框允许的最大宽高比。", "横向半身/姿态目标被过滤时调大；宽扁假框多时调小。", "更宽容。", "更严格。"),
    _p("detection_filter.small_max_body_aspect", "小目标 body 最大宽高比", "几何过滤", "number", 0.50, 2.50, 0.01, "w/h", "小目标 body 框允许的最大宽高比。", "小目标横向姿态被过滤时调大。", "小目标召回增加，宽扁误检增加。", "防宽扁误检更强。"),
    _p("detection_filter.small_pair_far_center_px", "小目标远离中心阈值", "防误检", "number", 50, 260, 1, "px", "超过此距离的小目标会触发更高置信度要求。", "边缘真实目标进不来时调大；边缘误检多时调小。", "远处小目标更容易通过。", "远处误检更容易被挡。"),
    _p("detection_filter.small_pair_far_min_head_conf", "远处小目标 head 要求", "防误检", "number", 0.30, 0.95, 0.01, "", "远离中心的小 paired target 对 head 的置信度要求。", "远处真实目标漏检时降低。", "防误检更强。", "召回更强。"),
    _p("detection_filter.small_pair_far_min_body_conf", "远处小目标 body 要求", "防误检", "number", 0.20, 0.95, 0.01, "", "远离中心的小 paired target 对 body 的置信度要求。", "远处真实目标有 body 但置信度低时降低。", "更稳。", "更易通过。"),
    _p("target_lock.allow_switch_while_locked", "锁定后允许切换", "多目标锁定", "bool", unit="", description="已有锁定目标时是否允许更好目标挑战当前锁。", when="多目标场景迟钝时打开；频繁左右跳时关闭。", bigger="开启后更灵活。", smaller="关闭后更黏住当前目标。"),
    _p("target_lock.switch_confirm_frames", "切换确认帧", "多目标锁定", "number", 1, 8, 1, "帧", "新目标挑战旧目标时需要连续确认的帧数。", "多目标切换慢时降低；左右乱跳时升高。", "更稳但慢。", "更快但可能抖/跳。", "critical"),
    _p("target_lock.switch_center_advantage_px", "切换中心优势", "多目标锁定", "number", 0, 160, 1, "px", "新目标需要比旧目标更靠近中心多少像素才更容易切换。", "准星附近目标不被优先选中时降低。", "更保守，不轻易切。", "更容易切到中心附近目标。"),
    _p("target_lock.lost_switch_after_frames", "丢失后切换等待", "多目标锁定", "number", 0, 10, 1, "帧", "当前锁丢失后等待多少帧才允许切到新目标。", "目标快速出现/遮挡时降低。", "更稳但迟钝。", "更快接新目标。"),
    _p("target_lock.missing_switch_confirm_frames", "丢失切换确认帧", "多目标锁定", "number", 1, 8, 1, "帧", "旧目标丢失后，新目标连续确认帧数。", "快速切换慢时降低；误切时升高。", "更稳。", "更快。"),
    _p("target_lock.max_lock_velocity_px_s", "锁定最大速度", "多目标锁定", "number", 500, 6000, 50, "px/s", "目标锁内部允许的最大速度。", "目标横移快、跟不上时调大。", "能跟更快目标，但误跳容忍更高。", "更稳但跟踪钝。"),
    _p("target_lock.velocity_smoothing", "速度平滑", "多目标锁定", "number", 0.30, 0.98, 0.01, "", "锁定速度估计的平滑系数。", "快速目标滞后时降低；轨迹抖动时升高。", "更稳但滞后。", "更灵敏但更抖。"),
    _p("tracking.ema_alpha", "跟踪 EMA 权重", "卡尔曼/跟踪", "number", 0.10, 0.90, 0.01, "", "观测值对平滑目标点的影响权重。", "跟踪滞后时调大；抖动时调小。", "响应更快但抖。", "更平滑但慢。"),
    _p("tracking.kalman_process_noise", "卡尔曼过程噪声", "卡尔曼/跟踪", "number", 0.005, 0.50, 0.005, "", "模型相信目标运动会变化的程度。", "快速横移跟不上时调大；预测乱跳时调小。", "更灵敏，预测更敢变。", "更稳，响应更慢。"),
    _p("tracking.kalman_measurement_noise", "卡尔曼测量噪声", "卡尔曼/跟踪", "number", 0.01, 0.50, 0.005, "", "模型认为检测框测量有多不可靠。", "检测框抖动时调大；响应慢时调小。", "更平滑但慢。", "更相信检测，响应快但抖。"),
    _p("tracking.kalman_max_velocity_px_s", "卡尔曼最大速度", "卡尔曼/跟踪", "number", 500, 7000, 50, "px/s", "卡尔曼预测允许的速度上限。", "高速移动目标被限制时调大。", "更能跟高速目标。", "更稳。"),
    _p("control.pid_enabled", "启用 PID", "移动/PID", "bool", description="是否使用 PID 控制误差到移动量的转换。", when="需要更系统地调节速度、过冲和稳态残差时打开。", bigger="开启 PID。", smaller="关闭 PID。"),
    _p("control.pid_kp", "PID P 比例", "移动/PID", "number", 0.10, 3.00, 0.01, "", "当前误差直接转换为移动的主增益。", "整体拉不动/慢时调大；过冲/抖动时调小。", "更快更猛，过冲风险增加。", "更稳更慢。", "critical"),
    _p("control.pid_ki", "PID I 积分", "移动/PID", "number", 0.0, 0.05, 0.001, "", "累计小误差，用于消除长期残差。", "总差一点点拉不到位时小幅调大；中心抖动时调低。", "更容易消残差，也更容易抖。", "更稳，但残差可能留着。"),
    _p("control.pid_kd", "PID D 微分", "移动/PID", "number", 0.0, 0.05, 0.001, "", "根据误差变化率进行阻尼。", "过冲明显时调大；快速目标跟不上时调小。", "刹车更强，可能拖慢。", "响应更直接，过冲风险增。"),
    _p("control.fire_enabled", "启用自动点击", "Fire/点击", "bool", description="启用后，只有在目标通过控制门、误差进入 fire_radius、且移动输出已经归零时才触发左键点击。", when="需要在稳定对准后由程序触发点击时开启；默认关闭。", bigger="开启点击门控。", smaller="关闭点击门控。", danger="danger"),
    _p("control.fire_radius", "点击半径", "Fire/点击", "number", 1, 18, 0.5, "px", "目标误差小于该半径时才允许点击。", "到点后不触发时调大；误触发时调小。", "更容易触发。", "更严格。", "critical"),
    _p("control.fire_exit_radius", "点击退出半径", "Fire/点击", "number", 1, 30, 0.5, "px", "点击半径的滞回退出值，防止边缘来回抖动导致反复重置。", "边缘抖动重复触发时调大。", "滞回更强，不易重复重置。", "更容易重新进入。"),
    _p("control.fire_rearm_radius", "重新装填半径", "Fire/点击", "number", 1, 40, 0.5, "px", "one-shot 模式下，目标误差超过该半径后才允许下一次点击。", "同一目标被连续误点时调大；需要更快二次触发时调小。", "重复点击更少。", "更容易再次触发。"),
    _p("control.fire_cooldown_ms", "点击冷却", "Fire/点击", "number", 40, 1000, 5, "ms", "两次点击之间的最短间隔。", "点击过密时调大；触发太慢时调小。", "点击更稀疏。", "点击更频繁。", "critical"),
    _p("control.fire_min_conf", "点击最低置信度", "Fire/点击", "number", 0.10, 0.99, 0.01, "", "允许点击的最低目标置信度。", "误点击弱框时调高；明显对准但不触发时调低。", "更稳。", "更容易触发。", "critical"),
    _p("control.fire_stable_frames", "点击稳定帧", "Fire/点击", "number", 1, 8, 1, "帧", "目标进入点击半径后需要经过多少个检测发布帧才允许点击。", "误触发时调大；触发慢时调小。", "更稳但慢。", "更快但风险更高。", "critical"),
    _p("control.fire_max_target_age_ms", "点击目标最大年龄", "Fire/点击", "number", 20, 250, 5, "ms", "只允许使用多新的目标检测结果触发点击。", "检测偶发间隔导致不触发时调大；旧目标误触发时调小。", "更容忍短暂检测间隔。", "更严格，必须新鲜检测。"),
    _p("control.fire_allow_held_target", "允许 held 目标点击", "Fire/点击", "bool", description="是否允许用短时保持/预测目标触发点击。", when="一般保持关闭；如果检测偶发断帧导致对准后不触发才考虑开启。", bigger="允许 held/predicted 目标点击，风险更高。", smaller="只允许真实新鲜检测触发。", danger="danger"),
    _p("control.fire_repeat_while_in_radius", "半径内重复点击", "Fire/点击", "bool", description="开启后目标留在点击半径内会按冷却时间重复点击；关闭则一次进入半径只点一次。", when="默认关闭；需要持续重复点击时开启。", bigger="可重复点击。", smaller="one-shot，更安全。", danger="danger"),
    _p("control.fire_reset_on_active_release", "松开热键重置点击门", "Fire/点击", "bool", description="松开 active_key 后重置 fire one-shot 状态。", when="通常保持开启。", bigger="下次按键可重新触发。", smaller="状态更黏，通常不推荐。"),
    _p("control.fire_log_events", "记录点击事件", "Fire/点击", "bool", description="点击触发时写入日志。", when="排查 fire_enabled 是否触发时开启；正常使用可关闭。", bigger="日志更详细。", smaller="日志更干净。"),
    _p("control.fire_require_zero_motion", "要求移动归零", "Fire/点击", "bool", description="只允许在本轮移动输出为 0 时触发点击。", when="通常保持开启；如果关闭，点击可能与移动同帧竞争。", bigger="必须先停止移动再点击。", smaller="更激进，但风险更高。", danger="danger"),
    _p("control.fire_max_motion_debt_px", "最大剩余移动债", "Fire/点击", "number", 0, 8, 0.1, "px", "剩余 residual/pending 移动债小于该值时才允许点击。", "明明已经对准但不触发时小幅调大；点击过早时调小。", "更容易触发。", "更严格，必须完全排空移动。", "critical"),
    _p("control.fire_min_time_after_move_ms", "移动后等待时间", "Fire/点击", "number", 0, 120, 1, "ms", "最后一次非零移动后至少等待多久才允许点击。", "移动未完全停稳就点击时调大；触发慢时调小。", "更稳但慢。", "更快但风险更高。", "critical"),
    _p("control.fire_stable_error_delta_px", "点击稳定误差变化", "Fire/点击", "number", 0.2, 12, 0.1, "px", "连续稳定帧之间误差变化必须小于该值。", "目标轻微抖动导致不触发时调大；误点跳变目标时调小。", "更宽容。", "更严格。", "critical"),
    _p("control.fire_block_during_settle_release", "阻断 settle 释放期点击", "Fire/点击", "bool", description="settle 锁正在释放/过渡时禁止点击。", when="通常保持开启，避免中心附近抖动边缘误触发。", bigger="更稳。", smaller="更快但风险更高。"),
    _p("control.fire_repeat_requires_fresh_detection", "重复点击要求新检测", "Fire/点击", "bool", description="开启后，即使允许半径内重复点击，也不能用同一个检测帧反复点击。", when="通常保持开启，避免 motor loop 在同一帧检测上重复触发。", bigger="更稳，更依赖新检测。", smaller="更激进，可能重复消耗同一检测。", danger="critical"),
    _p("control.fire_min_repeat_seq_delta", "重复点击最小检测间隔", "Fire/点击", "number", 1, 12, 1, "帧", "两次点击之间至少间隔多少个检测发布序号。", "重复太密时调大；触发太慢时调小。", "重复更慢更稳。", "重复更快。"),
    _p("control.fire_held_target_min_conf", "held 点击最低置信度", "Fire/点击", "number", 0.10, 0.99, 0.01, "", "只有显式允许 held/predicted 目标点击时才使用的更高置信度门槛。", "held 目标误触发时调高。", "更稳。", "更容易用 held 目标触发。", danger="critical"),
    _p("control.fire_held_target_max_age_ms", "held 点击最大年龄", "Fire/点击", "number", 10, 180, 5, "ms", "held/predicted 目标允许点击的最大年龄。", "检测短断帧但已对准不触发时小幅调大。", "更容忍短断检。", "更严格。"),
    _p("control.fire_block_on_stale_gate", "阻断陈旧门控点击", "Fire/点击", "bool", description="目标已经被实时循环判定为陈旧时不允许 fire。", when="通常保持开启。", bigger="更稳。", smaller="更激进，不推荐。"),
    _p("control.sensitivity_scaler", "整体速度倍率", "移动/PID", "number", 0.30, 1.80, 0.01, "×", "控制输出的整体倍率。", "只想整体快一点/慢一点时调它。", "整体更快。", "整体更慢更稳。", "critical"),
    _p("control.max_move", "单次最大移动", "移动/PID", "number", 4, 60, 1, "px", "每次提交给控制设备的移动上限。", "大偏差拉不动时调大；瞬间猛拉/过冲时调小。", "大幅移动更快。", "更柔和。"),
    _p("control.max_step", "单步最大值", "移动/PID", "number", 4, 60, 1, "px", "内部移动分配的单步限制。", "速度不够时小幅调大。", "速度提高，风险增。", "更稳更慢。"),
    _p("control.locked_slew_px_per_frame", "锁定点最大追随步长", "移动/PID", "number", 4, 80, 1, "px/帧", "平滑锁定点每帧最多追随 raw target 的距离。", "目标横移快但锁点滞后时调大。", "跟随更快，可能抖。", "更稳但拖后。"),
    _p("control.locked_smooth_alpha", "锁定点平滑权重", "移动/PID", "number", 0.10, 0.95, 0.01, "", "锁定点平滑时新观测的权重。", "目标快速移动滞后时调大；抖动时调小。", "更灵敏。", "更平滑。"),
    _p("control.deadzone", "死区", "防抖/稳定", "number", 0.5, 12.0, 0.1, "px", "误差小于该值时不主动移动。", "中心附近抖动时调大；到点前停住时调小。", "防抖增强，但精细贴合差。", "更精细，但可能抖。", "critical"),
    _p("control.fine_deadzone", "精细死区", "防抖/稳定", "number", 0.5, 10.0, 0.1, "px", "更靠近中心时的精细移动死区。", "细碎抖动时调大。", "更稳。", "更精细。"),
    _p("control.near_center_damping_px", "近中心阻尼半径", "防抖/稳定", "number", 4, 80, 1, "px", "进入该半径后开始压低移动量。", "拉回后还晃时调大。", "更早减速，更稳。", "靠近目标时仍较快。"),
    _p("control.near_center_damping_scale", "近中心阻尼强度", "防抖/稳定", "number", 0.01, 0.50, 0.005, "", "中心附近最终移动缩放。", "中心抖动时调小；中心附近太慢时调大。", "中心附近更快。", "中心附近更稳。"),
    _p("control.residual_error_fraction", "残差注入比例", "防抖/稳定", "number", 0.10, 1.00, 0.01, "", "每次把误差转入残差池的比例。", "响应慢时调大；过冲/回拉后抖时调小。", "更快但容易过冲。", "更稳但慢。"),
    _p("control.max_residual_total", "残差池上限", "防抖/稳定", "number", 4, 80, 1, "px", "残差累计的最大量。", "大偏差拉不完时调大；到点后继续动时调小。", "更能追大误差。", "减少拖尾。"),
    _p("control.no_target_soft_hold_enabled", "短断检软保持", "移动连续性", "bool", unit="", description="检测短暂掉一帧时，不立刻清空目标和残差，只继续排空已经确认过的运动。", when="单目标拉动时出现走一段、停一下、再走一段时打开。", bigger="开启后连续性更好。", smaller="关闭后更保守但可能卡顿。", danger="critical"),
    _p("control.no_target_soft_hold_ms", "短断检保持时间", "移动连续性", "number", 0, 180, 5, "ms", "允许短暂断检继续排空旧残差的时间窗口。", "检测偶发丢帧导致移动卡顿时调大；误跟旧目标时调小。", "更能抗一帧/两帧断检，移动更连续。", "更快清空，防旧目标更强但容易顿。", "critical"),
    _p("control.continuous_motion_profile_hold", "连续运动曲线保持", "移动连续性", "bool", unit="", description="当某个 HID tick 没有输出整数位移但仍有残差债务时，不立刻重置运动曲线。", when="体感有明显启停、脉冲感时打开。", bigger="开启后下一次非零包衔接更顺。", smaller="关闭后每次零输出都会重新起步。"),
    _p("control.continuous_motion_profile_hold_ms", "运动曲线保持时间", "移动连续性", "number", 0, 120, 5, "ms", "连续运动曲线在零输出间隙中保留多久。", "拉动过程中有微停顿时调大；靠近中心粘滞时调小。", "更连续但更黏。", "更快归零但更容易一顿一顿。"),
    _p("control.continuous_motion_profile_decay", "曲线保持衰减", "移动连续性", "number", 0.50, 0.98, 0.01, "", "零输出间隙中运动曲线的保留比例。", "启停感明显时调大；尾部拖泥带水时调小。", "衔接更顺。", "归零更快。"),
    _p("control.reactive_fast_enter_enabled", "快速进入通道", "快速响应", "bool", description="可信 paired target 可跳过部分普通确认。", when="目标突然出现但响应慢时打开；误检突然动时关闭。", bigger="开启快速通道。", smaller="关闭快速通道。", danger="critical"),
    _p("control.reactive_fast_enter_min_conf", "快速通道 head 阈值", "快速响应", "number", 0.30, 0.95, 0.01, "", "快速通道要求的 head 置信度。", "真实目标出现慢时降低；误动时升高。", "更稳但慢。", "更快但误检风险增。", "critical"),
    _p("control.reactive_fast_enter_min_body_conf", "快速通道 body 阈值", "快速响应", "number", 0.20, 0.95, 0.01, "", "快速通道要求的 body 置信度。", "body 置信度偏低但是真目标时降低。", "更稳。", "更快进入。"),
    _p("control.reactive_fast_enter_center_dist_px", "快速通道距离", "快速响应", "number", 30, 300, 1, "px", "距离中心多少以内的目标允许走快速通道。", "目标刚出现但不在准星附近也要快时调大。", "响应范围更大，误动风险增。", "更安全但范围小。"),
    _p("control.reactive_fast_enter_confirm_frames", "快速通道确认帧", "快速响应", "number", 1, 6, 1, "帧", "快速通道也需要连续确认的帧数。", "还是慢就调到 1；误动就调到 2~3。", "更稳但慢。", "更快。", "critical"),
    _p("control.instant_enter_enabled", "瞬时进入", "快速响应", "bool", description="高置信度近中心目标直接进入控制。", when="近距离突然出现目标要立刻动时打开；误动时关闭。", bigger="开启。", smaller="关闭。", danger="danger"),
    _p("control.instant_enter_min_conf", "瞬时进入阈值", "快速响应", "number", 0.50, 0.99, 0.01, "", "瞬时进入要求的 head 置信度。", "真目标近距离还慢时降低；误动时升高。", "更稳但更慢。", "更快但更危险。", "danger"),
]


PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "title": "平衡推荐",
        "description": "保持当前版本总体风格：召回、误检和抖动折中。",
        "patch": {
            "model.conf": 0.18,
            "detection_filter.min_head_conf": 0.22,
            "detection_filter.min_body_conf": 0.24,
            "detection_filter.min_body_height_px": 18.0,
            "detection_filter.small_min_body_height_px": 18.0,
            "control.reactive_fast_enter_enabled": True,
            "control.reactive_fast_enter_min_conf": 0.58,
            "control.reactive_fast_enter_min_body_conf": 0.48,
            "control.reactive_fast_enter_confirm_frames": 1,
            "control.sensitivity_scaler": 0.98,
            "control.pid_kp": 1.0,
            "control.deadzone": 4.4,
        },
    },
    "fast_recall": {
        "title": "更快识别/更强召回",
        "description": "目标一出现更容易被选中，适合觉得检测慢、漏目标。误检风险会提高。",
        "patch": {
            "model.conf": 0.16,
            "model.max_candidates": 360,
            "selection.head_conf": 0.19,
            "selection.body_conf": 0.25,
            "detection_filter.min_head_conf": 0.20,
            "detection_filter.min_body_conf": 0.22,
            "detection_filter.paired_head_min_conf": 0.18,
            "detection_filter.small_paired_head_min_conf": 0.18,
            "detection_filter.min_body_height_px": 16.0,
            "detection_filter.small_min_body_height_px": 16.0,
            "detection_filter.small_pair_far_min_head_conf": 0.52,
            "detection_filter.small_pair_far_min_body_conf": 0.40,
            "control.reactive_fast_enter_enabled": True,
            "control.reactive_fast_enter_min_conf": 0.52,
            "control.reactive_fast_enter_min_body_conf": 0.42,
            "control.reactive_fast_enter_center_dist_px": 245.0,
            "control.reactive_fast_enter_confirm_frames": 1,
            "control.small_target_confirmed_frames": 1,
            "control.suspicious_target_confirmed_frames": 2,
        },
    },
    "anti_false": {
        "title": "防误检增强",
        "description": "降低地图结构、小光点、短 body 误检。会牺牲中远距离召回。",
        "patch": {
            "model.conf": 0.24,
            "detection_filter.min_head_conf": 0.28,
            "detection_filter.min_body_conf": 0.30,
            "detection_filter.head_only_min_conf": 0.94,
            "detection_filter.small_head_only_min_conf": 0.94,
            "detection_filter.min_body_height_px": 24.0,
            "detection_filter.small_min_body_height_px": 28.0,
            "detection_filter.small_pair_far_min_head_conf": 0.72,
            "detection_filter.small_pair_far_min_body_conf": 0.62,
            "control.reactive_fast_enter_min_conf": 0.70,
            "control.reactive_fast_enter_min_body_conf": 0.62,
            "control.reactive_fast_enter_confirm_frames": 2,
            "control.instant_enter_enabled": False,
            "control.small_target_confirmed_frames": 5,
            "control.suspicious_target_confirmed_frames": 6,
        },
    },
    "smooth_continuity": {
        "title": "消除一顿一顿",
        "description": "针对单目标拉动时走一段、停一下、再走一段的启停感。",
        "patch": {
            "control.no_target_soft_hold_enabled": True,
            "control.no_target_soft_hold_ms": 80.0,
            "control.continuous_motion_profile_hold": True,
            "control.continuous_motion_profile_hold_ms": 55.0,
            "control.continuous_motion_profile_decay": 0.86,
            "control.residual_injection_min_ticks": 2,
            "control.residual_injection_max_ticks": 16,
            "control.residual_injection_interval_fraction": 0.62,
            "control.natural_motion_alpha": 0.66,
            "control.natural_motion_max_delta": 4.5,
            "control.natural_motion_close_delta": 1.35,
            "control.stale_target_min_seconds": 0.070,
            "control.stale_target_max_seconds": 0.180,
        },
    },
    "anti_jitter": {
        "title": "减少抖动",
        "description": "降低拉回目标后的中心附近抖动和残差拖尾。整体响应会变柔。",
        "patch": {
            "control.deadzone": 5.2,
            "control.fine_deadzone": 3.2,
            "control.near_center_damping_px": 40.0,
            "control.near_center_damping_scale": 0.055,
            "control.residual_error_fraction": 0.55,
            "control.drain_error_fraction": 0.55,
            "control.max_residual_total": 24,
            "control.pid_kp": 0.86,
            "control.pid_ki": 0.003,
            "control.pid_kd": 0.008,
            "tracking.ema_alpha": 0.46,
            "tracking.kalman_measurement_noise": 0.13,
        },
    },
    "speed_up": {
        "title": "移动速度小幅提高",
        "description": "不大幅放开检测，只提高移动跟随速度。",
        "patch": {
            "control.sensitivity_scaler": 1.06,
            "control.max_move": 24,
            "control.max_step": 24,
            "control.locked_slew_px_per_frame": 36.0,
            "control.locked_smooth_alpha": 0.78,
            "control.pid_kp": 1.08,
            "control.residual_error_fraction": 0.70,
            "control.drain_error_fraction": 0.70,
        },
    },
    "fast_target_motion": {
        "title": "高速目标跟随",
        "description": "增强横移/突然出现目标的追踪。适合觉得锁点滞后。",
        "patch": {
            "target_lock.max_lock_velocity_px_s": 3800,
            "target_lock.max_velocity_update_jump_px": 80,
            "target_lock.velocity_smoothing": 0.62,
            "target_lock.switch_confirm_frames": 1,
            "target_lock.missing_switch_confirm_frames": 1,
            "target_lock.predict_lost_frames": 4,
            "target_lock.predict_lost_ms": 70,
            "target_lock.prediction_ms": 28,
            "tracking.ema_alpha": 0.64,
            "tracking.kalman_process_noise": 0.16,
            "tracking.kalman_measurement_noise": 0.07,
            "tracking.kalman_max_velocity_px_s": 4800,
            "control.reactive_fast_enter_confirm_frames": 1,
            "control.locked_slew_px_per_frame": 38.0,
        },
    },
}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ONNX Runtime 调参面板</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #11182c;
      --panel2: #151f38;
      --line: #27324d;
      --text: #eef3ff;
      --muted: #9aa8c7;
      --accent: #7dd3fc;
      --accent2: #a7f3d0;
      --warn: #fde68a;
      --danger: #fca5a5;
      --ok: #86efac;
      --shadow: 0 18px 45px rgba(0,0,0,.35);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", system-ui, -apple-system, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top left, #172554 0, #0b1020 38%, #050812 100%);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(18px);
      background: rgba(9, 14, 28, .86);
      border-bottom: 1px solid var(--line);
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 22px;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: .2px; }
    .subtitle { color: var(--muted); font-size: 13px; margin-top: 4px; }
    button, input, select, textarea {
      font-family: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #1d2946;
      color: var(--text);
      padding: 9px 12px;
      cursor: pointer;
      transition: .15s ease;
      white-space: nowrap;
    }
    button:hover { transform: translateY(-1px); border-color: #3b82f6; }
    button.primary { background: linear-gradient(135deg, #2563eb, #0ea5e9); border-color: transparent; }
    button.success { background: linear-gradient(135deg, #059669, #10b981); border-color: transparent; }
    button.danger { background: #3b1d2a; border-color: #7f1d1d; color: #fecaca; }
    button.ghost { background: transparent; }
    .wrap { display: grid; grid-template-columns: 310px 1fr; min-height: calc(100vh - 70px); }
    aside {
      border-right: 1px solid var(--line);
      padding: 18px;
      background: rgba(11, 16, 32, .55);
    }
    main { padding: 18px 22px 40px; }
    .card {
      background: linear-gradient(180deg, rgba(21,31,56,.96), rgba(17,24,44,.96));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 16px;
    }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .stack { display: grid; gap: 10px; }
    .search {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #0b1223;
      color: var(--text);
      padding: 11px 12px;
      outline: none;
    }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .pill {
      display: inline-flex; align-items: center; gap: 6px;
      border: 1px solid var(--line); border-radius: 999px;
      padding: 4px 9px; font-size: 12px; color: var(--muted);
      background: rgba(255,255,255,.03);
    }
    .preset { width: 100%; text-align: left; padding: 11px 12px; }
    .preset b { display: block; font-size: 14px; margin-bottom: 3px; }
    .preset span { display: block; font-size: 12px; color: var(--muted); line-height: 1.35; white-space: normal; }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .tab { padding: 8px 10px; border-radius: 999px; font-size: 13px; }
    .tab.active { background: #1e40af; border-color: #60a5fa; color: white; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 14px; }
    .param {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(9, 14, 28, .55);
    }
    .param.changed { border-color: #38bdf8; box-shadow: 0 0 0 1px rgba(56,189,248,.15) inset; }
    .param .head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .param h3 { margin: 0 0 4px; font-size: 15px; }
    .path { font-family: Consolas, monospace; color: #93c5fd; font-size: 12px; word-break: break-all; }
    .param .desc { color: var(--muted); font-size: 13px; line-height: 1.55; margin-top: 9px; }
    .effect { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    .effect div { border: 1px solid var(--line); border-radius: 10px; padding: 8px; font-size: 12px; color: var(--muted); line-height: 1.45; }
    .effect b { color: var(--text); }
    .control-row { display: grid; grid-template-columns: 1fr 100px; gap: 10px; align-items: center; margin-top: 12px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    input[type="number"], input[type="text"], select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #0b1223;
      color: var(--text);
      padding: 8px 9px;
      outline: none;
    }
    input[type="checkbox"] { transform: scale(1.2); accent-color: var(--accent2); }
    .tag-danger { color: #fecaca; border-color: #7f1d1d; }
    .tag-critical { color: #fde68a; border-color: #854d0e; }
    .stat { display: grid; grid-template-columns: 1fr auto; gap: 8px; font-size: 13px; color: var(--muted); }
    .stat b { color: var(--text); }
    #toast {
      position: fixed; right: 18px; bottom: 18px; z-index: 99;
      max-width: 520px; padding: 12px 14px; border-radius: 12px;
      background: #0f172a; border: 1px solid var(--line); box-shadow: var(--shadow);
      display: none; color: var(--text); line-height: 1.5;
    }
    textarea {
      width: 100%; min-height: 420px; resize: vertical;
      border: 1px solid var(--line); border-radius: 12px;
      background: #060a13; color: #dbeafe;
      padding: 12px; font-family: Consolas, monospace; font-size: 13px; line-height: 1.5;
      outline: none;
    }
    .hidden { display: none !important; }
    .logbox { white-space: pre-wrap; font-family: Consolas, monospace; font-size: 12px; line-height: 1.55; color: #dbeafe; background: #060a13; border: 1px solid var(--line); border-radius: 12px; padding: 12px; max-height: 300px; overflow:auto; }
    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header>
  <div class="topbar">
    <div>
      <h1>ONNX Runtime 可视化调参面板</h1>
      <div class="subtitle" id="subtitle">读取中...</div>
    </div>
    <div class="row">
      <button class="ghost" onclick="reloadAll()">重新读取</button>
      <button onclick="downloadConfig()">导出 config.yaml</button>
      <button class="success" onclick="saveConfig()">保存并备份</button>
    </div>
  </div>
</header>
<div class="wrap">
  <aside>
    <div class="card stack">
      <input class="search" id="search" placeholder="搜索参数：例如 conf、PID、抖动、召回..." oninput="renderParams()" />
      <div class="row">
        <span class="pill" id="changedCount">0 项改动</span>
        <span class="pill" id="paramCount">0 个参数</span>
      </div>
      <div class="hint">保存时会先自动备份原始 <b>config.yaml</b>。主程序不需要开 GUI；GUI 只是改配置文件。</div>
    </div>

    <div class="card stack" id="presets"></div>

    <div class="card stack">
      <button onclick="analyzeLatestLog()">分析最新日志并给建议</button>
      <div class="hint">只读取 logs 目录下最新 run_*.txt，不会启动主程序。</div>
      <div id="logSummary" class="logbox hidden"></div>
    </div>
  </aside>

  <main>
    <div class="tabs" id="tabs"></div>
    <section id="paramView" class="grid"></section>

    <section id="yamlView" class="card hidden">
      <h2 style="margin-top:0">高级：直接编辑 YAML</h2>
      <p class="hint">适合修改页面没有列出的参数。保存前会做 YAML 解析检查；格式错误不会覆盖原配置。</p>
      <textarea id="yamlText"></textarea>
      <div class="row" style="margin-top:12px">
        <button onclick="loadYamlText()">重新载入 YAML</button>
        <button class="success" onclick="saveRawYaml()">保存 YAML 并备份</button>
      </div>
    </section>
  </main>
</div>
<div id="toast"></div>
<script>
let state = {config:{}, original:{}, flat:{}, originalFlat:{}, schema:[], presets:{}, activeGroup:'全部', yaml:''};

const $ = (id) => document.getElementById(id);
function clone(x){ return JSON.parse(JSON.stringify(x)); }
function showToast(msg, ms=2600){ const t=$('toast'); t.innerHTML=msg; t.style.display='block'; clearTimeout(window.__toastTimer); window.__toastTimer=setTimeout(()=>t.style.display='none', ms); }
function getPath(obj, path){ return path.split('.').reduce((a,k)=>a && Object.prototype.hasOwnProperty.call(a,k) ? a[k] : undefined, obj); }
function setPath(obj, path, val){ const ps=path.split('.'); let cur=obj; for(let i=0;i<ps.length-1;i++){ if(typeof cur[ps[i]]!=='object' || cur[ps[i]]===null) cur[ps[i]]={}; cur=cur[ps[i]]; } cur[ps[ps.length-1]]=val; }
function flatten(obj, prefix=''){ let out={}; Object.entries(obj||{}).forEach(([k,v])=>{ const p=prefix?prefix+'.'+k:k; if(v && typeof v==='object' && !Array.isArray(v)) Object.assign(out, flatten(v,p)); else out[p]=v; }); return out; }
function changedPaths(){ const f=flatten(state.config); const o=state.originalFlat; return Object.keys(f).filter(k => JSON.stringify(f[k]) !== JSON.stringify(o[k])); }
function fmt(v){ if(typeof v==='number') return Number.isInteger(v)? String(v) : String(Math.round(v*10000)/10000); if(typeof v==='boolean') return v?'true':'false'; return String(v ?? ''); }

async function api(path, options={}){
  const res = await fetch(path, options);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch(e){ throw new Error(text || res.statusText); }
  if(!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
  return data;
}

async function reloadAll(){
  try{
    const data = await api('/api/config');
    state.config = data.config;
    state.original = clone(data.config);
    state.flat = flatten(data.config);
    state.originalFlat = flatten(data.config);
    state.schema = data.schema;
    state.presets = data.presets;
    state.yaml = data.yaml;
    $('subtitle').textContent = `${data.config_path} ｜ version=${data.config.version || 'unknown'}`;
    renderPresets(); renderTabs(); renderParams();
    loadYamlText();
    showToast('配置已读取');
  }catch(e){ showToast('读取失败：'+e.message, 6000); }
}

function renderPresets(){
  const box=$('presets'); box.innerHTML='<h2 style="margin:0;font-size:16px">一键预设</h2>';
  Object.entries(state.presets).forEach(([key,p])=>{
    const btn=document.createElement('button'); btn.className='preset';
    btn.innerHTML=`<b>${p.title}</b><span>${p.description}</span>`;
    btn.onclick=()=>applyPresetLocal(key);
    box.appendChild(btn);
  });
  const reset=document.createElement('button'); reset.className='preset danger'; reset.innerHTML='<b>恢复出厂默认配置</b><span>使用 config.default_v17_8_19.yaml 覆盖当前内存配置，保存后生效。</span>'; reset.onclick=restoreDefaultLocal; box.appendChild(reset);
}

function groups(){
  const gs=['全部']; state.schema.forEach(s=>{ if(!gs.includes(s.group)) gs.push(s.group); }); gs.push('全部参数','YAML 高级'); return gs;
}
function renderTabs(){
  const tabs=$('tabs'); tabs.innerHTML='';
  groups().forEach(g=>{
    const b=document.createElement('button'); b.className='tab'+(state.activeGroup===g?' active':''); b.textContent=g;
    b.onclick=()=>{ state.activeGroup=g; renderTabs(); renderParams(); };
    tabs.appendChild(b);
  });
}

function renderParams(){
  const yamlMode = state.activeGroup === 'YAML 高级';
  $('yamlView').classList.toggle('hidden', !yamlMode);
  $('paramView').classList.toggle('hidden', yamlMode);
  const changed = changedPaths(); $('changedCount').textContent=`${changed.length} 项改动`;
  const search = $('search').value.trim().toLowerCase();
  let list;
  if(state.activeGroup === '全部参数'){
    const known = new Map(state.schema.map(s=>[s.path,s]));
    list = Object.entries(flatten(state.config)).map(([path,value]) => known.get(path) || {path,title:path,group:'全部参数',kind: typeof value === 'boolean' ? 'bool' : (typeof value === 'number' ? 'number' : 'text'), description:'当前配置文件中的参数。该项没有专门说明，修改前建议先备份。', when:'不确定含义时不要大幅调整。', bigger:'取决于参数语义。', smaller:'取决于参数语义。', danger:'normal'});
  } else {
    list = state.schema.filter(s => state.activeGroup==='全部' || s.group===state.activeGroup);
  }
  list = list.filter(s => {
    const blob = `${s.path} ${s.title} ${s.group} ${s.description} ${s.when}`.toLowerCase();
    return !search || blob.includes(search);
  });
  $('paramCount').textContent=`${list.length} 个参数`;
  const view=$('paramView'); view.innerHTML='';
  list.forEach(s=>view.appendChild(paramCard(s, changed.includes(s.path))));
}

function paramCard(s, changed){
  const v=getPath(state.config, s.path);
  const card=document.createElement('div'); card.className='param'+(changed?' changed':'');
  const tag=s.danger==='danger' ? '<span class="pill tag-danger">高风险</span>' : (s.danger==='critical'?'<span class="pill tag-critical">关键</span>':'');
  let control='';
  if(s.kind==='bool' || typeof v==='boolean'){
    control=`<div class="control-row" style="grid-template-columns:auto 1fr"><input type="checkbox" ${v?'checked':''} onchange="updateValue('${s.path}', this.checked)"><div class="hint">当前：${fmt(v)}</div></div>`;
  } else if(typeof v==='number' || s.kind==='number'){
    const min = s.min ?? Math.min(0, v*0.5); const max = s.max ?? Math.max(1, v*1.5+1); const step = s.step ?? 0.01;
    control=`<div class="control-row"><input type="range" min="${min}" max="${max}" step="${step}" value="${v}" oninput="updateValue('${s.path}', Number(this.value)); this.nextElementSibling.value=this.value"><input type="number" min="${min}" max="${max}" step="${step}" value="${v}" oninput="updateValue('${s.path}', Number(this.value)); this.previousElementSibling.value=this.value"></div>`;
  } else {
    control=`<div class="control-row" style="grid-template-columns:1fr"><input type="text" value="${String(v ?? '').replaceAll('&','&amp;').replaceAll('"','&quot;')}" oninput="updateValue('${s.path}', this.value)"></div>`;
  }
  card.innerHTML=`
    <div class="head"><div><h3>${s.title} ${s.unit?`<span class="hint">(${s.unit})</span>`:''}</h3><div class="path">${s.path}</div></div>${tag}</div>
    ${control}
    <div class="desc"><b>作用：</b>${s.description || '未提供说明。'}<br><b>什么时候调：</b>${s.when || '需要理解该参数后再调整。'}</div>
    <div class="effect"><div><b>调大：</b>${s.bigger || '取决于参数语义。'}</div><div><b>调小：</b>${s.smaller || '取决于参数语义。'}</div></div>`;
  return card;
}

function updateValue(path, val){
  setPath(state.config, path, val);
  renderParams();
}

async function saveConfig(){
  try{
    const data = await api('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({config:state.config})});
    state.original=clone(state.config); state.originalFlat=flatten(state.original); state.yaml=data.yaml;
    loadYamlText(); renderParams();
    showToast(`已保存。备份：${data.backup || '无'}`);
  }catch(e){ showToast('保存失败：'+e.message, 8000); }
}

async function applyPresetLocal(key){
  const p=state.presets[key]; if(!p) return;
  Object.entries(p.patch || {}).forEach(([path,val])=>setPath(state.config,path,val));
  renderParams();
  showToast(`已应用预设：${p.title}。点击“保存并备份”后才写入文件。`);
}

async function restoreDefaultLocal(){
  try{
    const data = await api('/api/default_config');
    state.config=data.config;
    renderParams();
    showToast('已载入出厂默认配置。点击“保存并备份”后才写入文件。');
  }catch(e){ showToast('恢复失败：'+e.message, 6000); }
}

function loadYamlText(){ $('yamlText').value = state.yaml || ''; }
async function saveRawYaml(){
  try{
    const data=await api('/api/raw_yaml', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({yaml:$('yamlText').value})});
    state.config=data.config; state.original=clone(data.config); state.originalFlat=flatten(data.config); state.yaml=data.yaml;
    renderParams(); renderTabs();
    showToast(`YAML 已保存。备份：${data.backup || '无'}`);
  }catch(e){ showToast('YAML 保存失败：'+e.message, 8000); }
}
function downloadConfig(){ window.open('/api/export','_blank'); }

async function analyzeLatestLog(){
  try{
    const data=await api('/api/analyze_latest_log');
    const box=$('logSummary'); box.classList.remove('hidden');
    box.textContent = data.summary;
    showToast('日志分析完成');
  }catch(e){ showToast('日志分析失败：'+e.message, 6000); }
}

reloadAll();
</script>
</body>
</html>"""


class TunerState:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.lock = threading.RLock()
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    def backup_current(self) -> Optional[str]:
        if not self.config_path.exists():
            return None
        backup = BACKUP_DIR / f"config_{_now_stamp()}.yaml"
        backup.write_text(self.config_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        return str(backup.relative_to(PROJECT_ROOT))

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            cfg = load_yaml(self.config_path)
            return {
                "ok": True,
                "config_path": str(self.config_path.relative_to(PROJECT_ROOT)) if self.config_path.is_relative_to(PROJECT_ROOT) else str(self.config_path),
                "config": cfg,
                "yaml": dump_yaml(cfg),
                "schema": [p.to_dict() for p in PARAM_SPECS],
                "presets": PRESETS,
            }

    def save_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(cfg, dict):
            raise ValueError("config 必须是对象")
        with self.lock:
            backup = self.backup_current()
            set_path(cfg, "version", "v17.8.19_config_tuner_gui")
            text = dump_yaml(cfg)
            # Parse once before writing.
            parsed = yaml.safe_load(text)
            if not isinstance(parsed, dict):
                raise ValueError("YAML 序列化结果无效")
            write_text_atomic(self.config_path, text)
            return {"ok": True, "backup": backup, "yaml": text, "config": parsed}

    def save_raw_yaml(self, raw: str) -> Dict[str, Any]:
        if not isinstance(raw, str):
            raise ValueError("yaml 必须是字符串")
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict):
            raise ValueError("YAML 根节点必须是对象")
        with self.lock:
            backup = self.backup_current()
            set_path(parsed, "version", "v17.8.19_config_tuner_gui")
            text = dump_yaml(parsed)
            write_text_atomic(self.config_path, text)
            return {"ok": True, "backup": backup, "yaml": text, "config": parsed}


def latest_log_file() -> Optional[Path]:
    if not LOG_DIR.exists():
        return None
    files = sorted(LOG_DIR.glob("run_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def analyze_log(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    status = [ln for ln in lines if " STATUS frame=" in ln]
    events = [ln for ln in lines if " EVENT " in ln]
    control_allowed = sum("control_allowed_change" in ln and "to=True" in ln for ln in events)
    movement_ready = sum("movement_ready_change" in ln and "to=True" in ln for ln in events)
    active_down = sum("active_key=DOWN" in ln for ln in events)
    raw_head = sum("raw_source_change" in ln and "to=head" in ln for ln in events)
    raw_none = sum("raw_source_change" in ln and "to=none" in ln for ln in events)
    fps_values: List[float] = []
    stale_values: List[int] = []
    filter_rej: List[Tuple[int, int]] = []
    for ln in status:
        m = re.search(r"fps=([0-9.]+)", ln)
        if m:
            try: fps_values.append(float(m.group(1)))
            except ValueError: pass
        m = re.search(r"stale_skips=([0-9]+)", ln)
        if m:
            stale_values.append(int(m.group(1)))
        m = re.search(r"filter=\{'in': ([0-9]+), 'out': ([0-9]+), 'head_rej': ([0-9]+), 'body_rej': ([0-9]+)\}", ln)
        if m:
            filter_rej.append((int(m.group(3)), int(m.group(4))))
    avg_fps = sum(fps_values)/len(fps_values) if fps_values else 0.0
    max_stale = max(stale_values) if stale_values else 0
    head_rej = sum(x for x, _ in filter_rej)
    body_rej = sum(y for _, y in filter_rej)
    suggestions: List[str] = []
    if avg_fps and avg_fps < 70:
        suggestions.append("平均 FPS 偏低：优先检查 GPU provider、关闭可视化、降低 max_candidates 或 ROI。")
    if max_stale > 180:
        suggestions.append("stale_skips 偏高：采集/推理链路有旧帧堆积，建议保持 threaded_capture=true、drop_stale_frames=true。")
    if raw_head > 0 and movement_ready == 0:
        suggestions.append("有 raw head 但 movement_ready 很少：控制门控过严，可试“更快识别/更强召回”预设。")
    if head_rej > raw_head * 3 and head_rej > 30:
        suggestions.append("head 被过滤较多：如果确认是真目标漏检，可降低 detection_filter.min_head_conf / small_paired_head_min_conf。")
    if movement_ready > 0 and control_allowed == 0:
        suggestions.append("movement_ready 有但 control_allowed 少：检查 active_key 是否按下，或控制基础校验是否过严。")
    if not suggestions:
        suggestions.append("没有发现明显异常。按体感选择预设微调即可。")
    return "\n".join([
        f"日志：{path.name}",
        f"行数：{len(lines)}",
        f"STATUS 数：{len(status)}",
        f"EVENT 数：{len(events)}",
        f"平均 FPS：{avg_fps:.1f}" if avg_fps else "平均 FPS：未解析到",
        f"active_key DOWN：{active_down}",
        f"raw head 出现次数：{raw_head}",
        f"raw none 次数：{raw_none}",
        f"movement_ready=True 次数：{movement_ready}",
        f"control_allowed=True 次数：{control_allowed}",
        f"累计 head_rej/body_rej：{head_rej}/{body_rej}",
        f"最大 stale_skips：{max_stale}",
        "",
        "建议：",
        *[f"- {s}" for s in suggestions],
    ])


class Handler(BaseHTTPRequestHandler):
    state: TunerState

    server_version = "ConfigTunerGUI/17.8.19"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[GUI] " + (fmt % args) + "\n")

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj: Mapping[str, Any], status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

    def error_json(self, msg: str, status: int = 400) -> None:
        self.send_json({"ok": False, "error": msg}, status)

    def read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/" or parsed.path == "/index.html":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/config":
                self.send_json(self.state.snapshot())
            elif parsed.path == "/api/default_config":
                cfg = load_yaml(DEFAULT_PROFILE_PATH)
                self.send_json({"ok": True, "config": cfg, "yaml": dump_yaml(cfg)})
            elif parsed.path == "/api/export":
                data = self.state.config_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-yaml; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=config.yaml")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/api/analyze_latest_log":
                path = latest_log_file()
                if not path:
                    self.error_json("logs 目录没有 run_*.txt", 404)
                    return
                self.send_json({"ok": True, "summary": analyze_log(path)})
            else:
                self.error_json(f"未知接口: {parsed.path}", 404)
        except Exception as exc:
            self.error_json(f"{exc}\n{traceback.format_exc()}", 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            data = self.read_json()
            if parsed.path == "/api/config":
                cfg = data.get("config")
                if not isinstance(cfg, dict):
                    self.error_json("缺少 config 对象")
                    return
                self.send_json(self.state.save_config(cfg))
            elif parsed.path == "/api/raw_yaml":
                raw = data.get("yaml")
                self.send_json(self.state.save_raw_yaml(raw))
            else:
                self.error_json(f"未知接口: {parsed.path}", 404)
        except Exception as exc:
            self.error_json(f"{exc}\n{traceback.format_exc()}", 500)


def find_free_port(host: str, port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return port
        except OSError:
            pass
    for p in range(port + 1, port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="启动本地浏览器调参 GUI")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", default=8765, type=int, help="监听端口，默认 8765")
    parser.add_argument("--open-browser", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise SystemExit(f"配置文件不存在: {config_path}")

    port = find_free_port(args.host, int(args.port))
    Handler.state = TunerState(config_path)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    url = f"http://{args.host}:{port}/"
    print("=" * 72)
    print("ONNX Runtime 可视化调参面板")
    print(f"配置文件: {config_path}")
    print(f"访问地址: {url}")
    print("关闭方法: 在此窗口按 Ctrl+C")
    print("=" * 72)

    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭调参面板。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
