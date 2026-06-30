from __future__ import annotations

import argparse
import atexit
import math
import os
import platform
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

EXTERNAL_STOP_EVENT = None  # Set by the native GUI when realtime runs inside the same process.

import cv2
import numpy as np
import yaml

from src.log_utils import close_logging, get_log_path, init_logging, log, log_block, log_exception, log_kv
from src.onnx_yolo_detector import OnnxYoloDetector
from src.runtime_controller import ControlConfig, RuntimeController
from src.control_gate import ConfirmedHeadGate, ControlGateConfig
from src.screen_capture import ScreenCapture, load_image_bgr
from src.frame_pipeline import LatestFrameReader
from src.profiler import RollingProfiler
from src.target_selector import TargetSelector
from src.target_lock import TargetLockConfig, TargetLockManager
from src.target_validation import MovementValidationConfig, validate_movement_target
from src.detection_filter import (
    DetectionGeometryFilterConfig,
    filter_detections_by_geometry,
)
from src.tracker import EmaPointTracker, LegacyPointTracker
from src.types import DetectionBox, TargetResult
from src.app_paths import apply_runtime_overrides, configure_dll_search_path, ensure_runtime_layout, VERSION


def load_cfg(path: str | Path) -> Dict[str, Any]:
    ensure_runtime_layout()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件根节点必须是字典: {path}")
    return apply_runtime_overrides(cfg)


def _yaml_dump(data: Any) -> str:
    try:
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    except Exception as e:
        return f"<failed to dump config: {e}>"


def log_session_header(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Write a detailed run header to the file log.

    Console output is intentionally quiet in v16; this file is the source of
    truth for later diagnosis.
    """
    log(f"SESSION_START {VERSION}", "SUCCESS")
    log_kv("ENV", {
        "cwd": os.getcwd(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": " ".join(sys.argv),
        "log_path": get_log_path() or "none",
    })
    log_kv("ARGS", vars(args))
    core_sections = {
        "model": cfg.get("model", {}),
        "capture": cfg.get("capture", {}),
        "runtime": cfg.get("runtime", {}),
        "visual": cfg.get("visual", {}),
        "selection": cfg.get("selection", {}),
        "detection_filter": cfg.get("detection_filter", {}),
        "target_lock": cfg.get("target_lock", {}),
        "tracking": cfg.get("tracking", {}),
        "control": cfg.get("control", {}),
        "logging": cfg.get("logging", {}),
    }
    for name, section in core_sections.items():
        log_kv(f"CONFIG_{name.upper()}", section if isinstance(section, dict) else {"value": section})
    if bool((cfg.get("logging", {}) or {}).get("include_config_snapshot", True)):
        log_block("CONFIG_SNAPSHOT", _yaml_dump(cfg), "INFO")


def _target_short(t: TargetResult, center: tuple[float, float]) -> str:
    if not t or not t.found:
        return "none"
    dist = math.hypot(float(t.x) - float(center[0]), float(t.y) - float(center[1]))
    head = t.head_box
    body = t.body_box
    head_s = "none" if head is None else f"{head.conf:.3f}@({head.x1:.1f},{head.y1:.1f},{head.x2:.1f},{head.y2:.1f})"
    body_s = "none" if body is None else f"{body.conf:.3f}@({body.x1:.1f},{body.y1:.1f},{body.x2:.1f},{body.y2:.1f})"
    return f"{t.source}:conf={t.confidence:.3f},xy=({t.x:.1f},{t.y:.1f}),dist={dist:.1f},head={head_s},body={body_s},reason={t.reason}"


def _clone_target(t: TargetResult, reason_suffix: str = "") -> TargetResult:
    if not t or not t.found:
        return TargetResult(False, reason=(getattr(t, "reason", "") if t else ""))
    reason = str(t.reason or "")
    if reason_suffix:
        reason = f"{reason}; {reason_suffix}" if reason else reason_suffix
    return TargetResult(
        found=True,
        x=float(t.x),
        y=float(t.y),
        source=str(t.source),
        confidence=float(t.confidence),
        reason=reason,
        head_box=t.head_box,
        body_box=t.body_box,
    )


def _correct_control_point(
    raw_target: TargetResult,
    tracked_target: TargetResult,
    cfg: Dict[str, Any],
    center: tuple[float, float],
) -> TargetResult:
    """Bound tracker lag before the point reaches geometry/gate/controller.

    V17.8.1 used Kalman-smoothed x/y directly. In the logs, the reported control
    point was often 15-70 px away from the current head box center while the same
    log line still printed the current head bbox. That is tracker lag, not a true
    target center. It explains the observed final stop on the left/right side of
    the head and the slow later correction.

    This function keeps tracker smoothing, but the control point may not lag too
    far behind the current detector head center. The current head bbox remains the
    final geometric anchor; the tracker is only allowed to denoise within a small
    radius around it.
    """
    trk_cfg = cfg.get("tracking", {}) or {}
    if not bool(trk_cfg.get("control_lag_clamp_enabled", True)):
        return tracked_target
    if not (raw_target and raw_target.found and tracked_target and tracked_target.found):
        return tracked_target
    if raw_target.source != "head" or raw_target.head_box is None:
        return tracked_target
    # Do not snap prediction-only targets to a stale head box.
    reason_text = str(raw_target.reason or "")
    if "prediction" in reason_text or "held" in reason_text or "missing" in reason_text:
        return tracked_target

    raw_x, raw_y = raw_target.head_box.center
    out = _clone_target(tracked_target)
    lag_x = float(out.x) - float(raw_x)
    lag_y = float(out.y) - float(raw_y)
    lag = math.hypot(lag_x, lag_y)
    max_lag = float(trk_cfg.get("max_control_lag_px", 5.0))
    if raw_target.body_box is not None and float(raw_target.confidence) >= float(trk_cfg.get("high_conf_lag_conf", 0.60)):
        max_lag = float(trk_cfg.get("max_control_lag_px_high_conf", 2.5))
    if "new target lock" in reason_text and bool(trk_cfg.get("snap_new_lock_to_raw", True)):
        max_lag = min(max_lag, float(trk_cfg.get("new_lock_max_lag_px", 1.5)))

    if lag > max_lag > 0.0:
        scale = max_lag / max(lag, 1e-6)
        out.x = float(raw_x) + lag_x * scale
        out.y = float(raw_y) + lag_y * scale
        out.reason = f"{out.reason}; control_lag_clamped raw_head_center=({raw_x:.1f},{raw_y:.1f}) lag={lag:.1f}->{max_lag:.1f}"
    return out

def _box_summary(boxes: list[DetectionBox]) -> Dict[str, Any]:
    heads = [b for b in boxes if b.cls_name == "head" or b.cls_id == 1]
    bodies = [b for b in boxes if b.cls_name == "body" or b.cls_id == 0]
    return {
        "boxes": len(boxes),
        "heads": len(heads),
        "bodies": len(bodies),
        "max_head_conf": round(max((b.conf for b in heads), default=0.0), 4),
        "max_body_conf": round(max((b.conf for b in bodies), default=0.0), 4),
        "min_head_area": round(min((b.area for b in heads), default=0.0), 2),
        "max_head_area": round(max((b.area for b in heads), default=0.0), 2),
    }


def draw_result(frame: np.ndarray, boxes: list[DetectionBox], target: TargetResult, center: tuple[float, float]) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        color = (0, 220, 0) if b.cls_id == 0 else (0, 165, 255)
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.putText(out, f"{b.cls_name} {b.conf:.2f}", (int(b.x1), max(15, int(b.y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cx, cy = int(center[0]), int(center[1])
    cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
    if target.found:
        tx, ty = int(target.x), int(target.y)
        cv2.circle(out, (tx, ty), 5, (0, 0, 255), -1)
        cv2.line(out, (cx, cy), (tx, ty), (0, 0, 255), 1)
        cv2.putText(out, f"target {target.source} {target.confidence:.2f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "target none", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    return out


def make_detector(cfg: Dict[str, Any]) -> OnnxYoloDetector:
    m = cfg.get("model", {})
    cls_cfg = m.get("classes", {})
    class_names = {int(cls_cfg.get("body", 0)): "body", int(cls_cfg.get("head", 1)): "head"}
    return OnnxYoloDetector(
        model_path=m.get("path", "vendor_models/valorant_320_v11n.onnx"),
        imgsz=int(m.get("imgsz", 320)),
        conf=float(m.get("conf", 0.25)),
        iou=float(m.get("iou", 0.70)),
        class_names=class_names,
        providers=m.get("providers", ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
        max_candidates=int(m.get("max_candidates", 300)),
        require_gpu=bool(m.get("require_gpu", True)),
    )


def make_selector(cfg: Dict[str, Any], center: tuple[float, float]) -> TargetSelector:
    m = cfg.get("model", {})
    s = cfg.get("selection", {})
    cls_cfg = m.get("classes", {})
    prefer_center = center if bool(s.get("prefer_screen_center", True)) else None
    return TargetSelector(
        body_class_id=int(cls_cfg.get("body", 0)),
        head_class_id=int(cls_cfg.get("head", 1)),
        head_conf=float(s.get("head_conf", m.get("conf", 0.25))),
        body_conf=float(s.get("body_conf", m.get("conf", 0.25))),
        body_fallback_y_ratio=float(s.get("body_fallback_y_ratio", 0.18)),
        prefer_center=prefer_center,
        prefer_head=bool(s.get("prefer_head", True)),
        fallback_to_body=bool(s.get("fallback_to_body", True)),
    )


def make_detection_filter_cfg(cfg: Dict[str, Any]) -> DetectionGeometryFilterConfig:
    """Build the pre-selection geometry filter used to suppress map false positives."""
    m = cfg.get("model", {})
    cls_cfg = m.get("classes", {})
    f = cfg.get("detection_filter", {})
    return DetectionGeometryFilterConfig(
        enabled=bool(f.get("enabled", True)),
        body_class_id=int(cls_cfg.get("body", 0)),
        head_class_id=int(cls_cfg.get("head", 1)),
        min_head_conf=float(f.get("min_head_conf", 0.20)),
        min_body_conf=float(f.get("min_body_conf", 0.30)),
        paired_head_min_conf=float(f.get("paired_head_min_conf", 0.23)),
        head_only_min_conf=float(f.get("head_only_min_conf", 0.72)),
        head_only_center_max_px=float(f.get("head_only_center_max_px", 145.0)),
        keep_unpaired_bodies=bool(f.get("keep_unpaired_bodies", True)),
        min_head_area_px=float(f.get("min_head_area_px", 4.0)),
        max_head_area_px=float(f.get("max_head_area_px", 2600.0)),
        min_head_width_px=float(f.get("min_head_width_px", 2.0)),
        min_head_height_px=float(f.get("min_head_height_px", 2.0)),
        max_head_width_px=float(f.get("max_head_width_px", 96.0)),
        max_head_height_px=float(f.get("max_head_height_px", 96.0)),
        min_head_aspect=float(f.get("min_head_aspect", 0.35)),
        max_head_aspect=float(f.get("max_head_aspect", 2.35)),
        min_body_area_px=float(f.get("min_body_area_px", 14.0)),
        max_body_area_ratio=float(f.get("max_body_area_ratio", 0.82)),
        min_body_width_px=float(f.get("min_body_width_px", 8.0)),
        min_body_height_px=float(f.get("min_body_height_px", 28.0)),
        small_min_body_width_px=float(f.get("small_min_body_width_px", 10.0)),
        small_min_body_height_px=float(f.get("small_min_body_height_px", 34.0)),
        min_body_aspect=float(f.get("min_body_aspect", 0.12)),
        max_body_aspect=float(f.get("max_body_aspect", 1.05)),
        small_max_body_aspect=float(f.get("small_max_body_aspect", 0.82)),
        border_reject_enabled=bool(f.get("border_reject_enabled", True)),
        border_margin_px=float(f.get("border_margin_px", 2.0)),
        border_reject_center_min_px=float(f.get("border_reject_center_min_px", 105.0)),
        border_reject_head_conf=float(f.get("border_reject_head_conf", 0.88)),
        border_reject_body_conf=float(f.get("border_reject_body_conf", 0.88)),
        require_head_near_body=bool(f.get("require_head_near_body", True)),
        body_expand_w_ratio=float(f.get("body_expand_w_ratio", 0.30)),
        body_expand_h_ratio=float(f.get("body_expand_h_ratio", 0.12)),
        head_upper_body_ratio_max=float(f.get("head_upper_body_ratio_max", 0.64)),
        min_head_body_area_ratio=float(f.get("min_head_body_area_ratio", 0.0015)),
        max_head_body_area_ratio=float(f.get("max_head_body_area_ratio", 0.58)),
        min_head_body_width_ratio=float(f.get("min_head_body_width_ratio", 0.020)),
        max_head_body_width_ratio=float(f.get("max_head_body_width_ratio", 0.92)),
        min_head_body_height_ratio=float(f.get("min_head_body_height_ratio", 0.020)),
        max_head_body_height_ratio=float(f.get("max_head_body_height_ratio", 0.82)),
        require_body_extends_below_head=bool(f.get("require_body_extends_below_head", True)),
        min_body_pixels_below_head=float(f.get("min_body_pixels_below_head", 3.0)),
        small_head_area_px=float(f.get("small_head_area_px", 96.0)),
        small_head_max_dim_px=float(f.get("small_head_max_dim_px", 18.0)),
        small_head_min_conf=float(f.get("small_head_min_conf", 0.20)),
        small_paired_head_min_conf=float(f.get("small_paired_head_min_conf", 0.18)),
        small_head_only_min_conf=float(f.get("small_head_only_min_conf", 0.32)),
        small_head_only_center_max_px=float(f.get("small_head_only_center_max_px", 230.0)),
        small_body_min_conf=float(f.get("small_body_min_conf", 0.18)),
        small_relaxed_pair=bool(f.get("small_relaxed_pair", True)),
        small_body_expand_w_ratio=float(f.get("small_body_expand_w_ratio", 0.58)),
        small_body_expand_h_ratio=float(f.get("small_body_expand_h_ratio", 0.28)),
        small_head_upper_body_ratio_max=float(f.get("small_head_upper_body_ratio_max", 0.78)),
        small_min_head_body_area_ratio=float(f.get("small_min_head_body_area_ratio", 0.00035)),
        small_max_head_body_area_ratio=float(f.get("small_max_head_body_area_ratio", 0.78)),
        small_min_head_body_width_ratio=float(f.get("small_min_head_body_width_ratio", 0.006)),
        small_max_head_body_width_ratio=float(f.get("small_max_head_body_width_ratio", 1.15)),
        small_min_head_body_height_ratio=float(f.get("small_min_head_body_height_ratio", 0.006)),
        small_max_head_body_height_ratio=float(f.get("small_max_head_body_height_ratio", 1.05)),
        small_min_body_pixels_below_head=float(f.get("small_min_body_pixels_below_head", 1.0)),
        small_pair_far_center_px=float(f.get("small_pair_far_center_px", 90.0)),
        small_pair_far_min_head_conf=float(f.get("small_pair_far_min_head_conf", 0.82)),
        small_pair_far_min_body_conf=float(f.get("small_pair_far_min_body_conf", 0.84)),
        small_pair_short_body_px=float(f.get("small_pair_short_body_px", 72.0)),
        small_pair_short_min_head_conf=float(f.get("small_pair_short_min_head_conf", 0.84)),
        small_pair_short_min_body_conf=float(f.get("small_pair_short_min_body_conf", 0.84)),
    )


def make_target_lock(cfg: Dict[str, Any]) -> TargetLockManager:
    t = cfg.get("target_lock", {})
    return TargetLockManager(TargetLockConfig(
        enabled=bool(t.get("enabled", True)),
        hold_lost_frames=int(t.get("hold_lost_frames", 5)),
        hold_lost_seconds=float(t.get("hold_lost_seconds", 0.16)),
        match_max_distance_px=float(t.get("match_max_distance_px", 85.0)),
        hard_match_max_distance_px=float(t.get("hard_match_max_distance_px", 150.0)),
        body_iou_match_max_distance_px=float(t.get("body_iou_match_max_distance_px", 120.0)),
        min_lock_head_conf=float(t.get("min_lock_head_conf", 0.30)),
        head_without_body_lock_conf=float(t.get("head_without_body_lock_conf", 0.86)),
        head_iou_min=float(t.get("head_iou_min", 0.015)),
        body_iou_min=float(t.get("body_iou_min", 0.015)),
        allow_switch_while_locked=bool(t.get("allow_switch_while_locked", False)),
        switch_center_advantage_px=float(t.get("switch_center_advantage_px", 90.0)),
        switch_conf_advantage=float(t.get("switch_conf_advantage", 0.20)),
        switch_confirm_frames=int(t.get("switch_confirm_frames", 3)),
        switch_match_px=float(t.get("switch_match_px", 42.0)),
        switch_score_advantage=float(t.get("switch_score_advantage", 0.10)),
        switch_max_center_dist_px=float(t.get("switch_max_center_dist_px", 210.0)),
        initial_center_weight=float(t.get("initial_center_weight", 1.0)),
        initial_conf_weight=float(t.get("initial_conf_weight", 1.65)),
        initial_body_conf_weight=float(t.get("initial_body_conf_weight", 0.65)),
        reset_on_active_press=bool(t.get("reset_on_active_press", True)),
        reset_on_control_off=bool(t.get("reset_on_control_off", False)),
        allow_switch_when_locked_missing=bool(t.get("allow_switch_when_locked_missing", True)),
        lost_switch_after_frames=int(t.get("lost_switch_after_frames", 2)),
        lost_switch_min_conf=float(t.get("lost_switch_min_conf", 0.40)),
        lost_switch_requires_body=bool(t.get("lost_switch_requires_body", True)),
        lost_switch_center_max_px=float(t.get("lost_switch_center_max_px", 180.0)),
        missing_switch_confirm_frames=int(t.get("missing_switch_confirm_frames", 2)),
        missing_switch_match_px=float(t.get("missing_switch_match_px", 60.0)),
        max_lock_velocity_px_s=float(t.get("max_lock_velocity_px_s", 1600.0)),
        max_velocity_update_jump_px=float(t.get("max_velocity_update_jump_px", 55.0)),
        predict_lost_target=bool(t.get("predict_lost_target", True)),
        predict_lost_frames=int(t.get("predict_lost_frames", 2)),
        predict_lost_ms=float(t.get("predict_lost_ms", 45.0)),
        predict_lost_min_conf=float(t.get("predict_lost_min_conf", 0.35)),
        prediction_ms=float(t.get("prediction_ms", 35.0)),
        velocity_smoothing=float(t.get("velocity_smoothing", 0.65)),
    ))


def make_controller(cfg: Dict[str, Any]) -> RuntimeController:
    c = cfg.get("control", {})
    return RuntimeController(ControlConfig(
        enabled=bool(c.get("enabled", False)),
        mode=str(c.get("mode", "leonardo")),
        port=str(c.get("port", "auto")),
        baud=int(c.get("baud", 115200)),
        gain_x=float(c.get("gain_x", 1.0)),
        gain_y=float(c.get("gain_y", 1.0)),
        sensitivity_scaler=float(c.get("sensitivity_scaler", 0.78)),
        sensitivity_boost_close=float(c.get("sensitivity_boost_close", 1.25)),
        close_range_threshold=float(c.get("close_range_threshold", 140.0)),
        min_kinetic_speed=float(c.get("min_kinetic_speed", 0.0)),
        pid_enabled=bool(c.get("pid_enabled", True)),
        pid_kp=float(c.get("pid_kp", 1.0)),
        pid_ki=float(c.get("pid_ki", 0.006)),
        pid_kd=float(c.get("pid_kd", 0.004)),
        pid_integral_limit=float(c.get("pid_integral_limit", 28.0)),
        pid_derivative_smoothing=float(c.get("pid_derivative_smoothing", 0.70)),
        pid_output_limit=float(c.get("pid_output_limit", 0.0)),
        pid_integral_deadband_px=float(c.get("pid_integral_deadband_px", 7.0)),
        pid_reset_jump_px=float(c.get("pid_reset_jump_px", 95.0)),
        pid_dt_min=float(c.get("pid_dt_min", 0.001)),
        pid_dt_max=float(c.get("pid_dt_max", 0.08)),
        velocity_lead_ms=float(c.get("velocity_lead_ms", 24.0)),
        velocity_lead_max_px=float(c.get("velocity_lead_max_px", 16.0)),
        error_velocity_smoothing=float(c.get("error_velocity_smoothing", 0.82)),
        velocity_reset_jump_px=float(c.get("velocity_reset_jump_px", 45.0)),
        max_error_velocity_px_s=float(c.get("max_error_velocity_px_s", 1800.0)),
        velocity_lead_error_fraction=float(c.get("velocity_lead_error_fraction", 0.35)),
        velocity_lead_min_error_px=float(c.get("velocity_lead_min_error_px", 8.0)),
        max_residual_total=float(c.get("max_residual_total", 64.0)),
        smooth_residual_injection=bool(c.get("smooth_residual_injection", True)),
        residual_injection_min_ticks=int(c.get("residual_injection_min_ticks", 4)),
        residual_injection_max_ticks=int(c.get("residual_injection_max_ticks", 28)),
        residual_injection_interval_fraction=float(c.get("residual_injection_interval_fraction", 0.85)),
        natural_motion_enabled=bool(c.get("natural_motion_enabled", True)),
        natural_motion_alpha=float(c.get("natural_motion_alpha", 0.46)),
        natural_motion_max_delta=float(c.get("natural_motion_max_delta", 2.4)),
        natural_motion_close_delta=float(c.get("natural_motion_close_delta", 0.85)),
        natural_motion_close_px=float(c.get("natural_motion_close_px", 34.0)),
        natural_motion_zero_cross_brake=bool(c.get("natural_motion_zero_cross_brake", True)),
        continuous_motion_profile_hold=bool(c.get("continuous_motion_profile_hold", True)),
        continuous_motion_profile_hold_ms=float(c.get("continuous_motion_profile_hold_ms", 45.0)),
        continuous_motion_profile_decay=float(c.get("continuous_motion_profile_decay", 0.82)),
        no_target_soft_hold_enabled=bool(c.get("no_target_soft_hold_enabled", True)),
        no_target_soft_hold_ms=float(c.get("no_target_soft_hold_ms", 65.0)),
        adaptive_stale_target=bool(c.get("adaptive_stale_target", True)),
        stale_target_min_seconds=float(c.get("stale_target_min_seconds", 0.045)),
        stale_target_max_seconds=float(c.get("stale_target_max_seconds", 0.120)),
        stale_target_interval_multiplier=float(c.get("stale_target_interval_multiplier", 3.0)),
        active_press_accept_recent_ms=float(c.get("active_press_accept_recent_ms", 22.0)),
        held_target_sensitivity_scale=float(c.get("held_target_sensitivity_scale", 0.35)),
        held_target_disable_lead=bool(c.get("held_target_disable_lead", True)),
        micro_step_threshold=float(c.get("micro_step_threshold", 0.95)),
        overshoot_guard_enabled=bool(c.get("overshoot_guard_enabled", True)),
        overshoot_error_fraction=float(c.get("overshoot_error_fraction", 0.42)),
        residual_error_fraction=float(c.get("residual_error_fraction", 0.58)),
        drain_error_fraction=float(c.get("drain_error_fraction", 0.45)),
        settle_lock_enabled=bool(c.get("settle_lock_enabled", True)),
        settle_enter_px=float(c.get("settle_enter_px", 2.2)),
        settle_enter_frames=int(c.get("settle_enter_frames", 1)),
        settle_exit_px=float(c.get("settle_exit_px", 6.0)),
        settle_hard_exit_px=float(c.get("settle_hard_exit_px", 18.0)),
        settle_release_frames=int(c.get("settle_release_frames", 2)),
        settle_min_conf=float(c.get("settle_min_conf", 0.30)),
        settle_target_radius_enabled=bool(c.get("settle_target_radius_enabled", True)),
        settle_enter_radius_fraction=float(c.get("settle_enter_radius_fraction", 0.10)),
        settle_exit_radius_fraction=float(c.get("settle_exit_radius_fraction", 0.55)),
        settle_hard_exit_radius_fraction=float(c.get("settle_hard_exit_radius_fraction", 1.15)),
        settle_exit_max_px=float(c.get("settle_exit_max_px", 18.0)),
        settle_hard_exit_max_px=float(c.get("settle_hard_exit_max_px", 34.0)),
        near_center_damping_px=float(c.get("near_center_damping_px", 12.0)),
        near_center_damping_scale=float(c.get("near_center_damping_scale", 0.45)),
        max_submit_error_px=float(c.get("max_submit_error_px", 180.0)),
        max_residual_add_per_frame=float(c.get("max_residual_add_per_frame", 24.0)),
        max_step=int(c.get("max_step", c.get("max_move", 34))),
        max_move=int(c.get("max_move", c.get("max_step", 34))),
        micro_step_min_error_px=float(c.get("micro_step_min_error_px", 8.0)),
        deadzone=float(c.get("deadzone", c.get("dead_zone", 3.4))),
        fine_deadzone=float(c.get("fine_deadzone", 1.8)),
        invert_y=bool(c.get("invert_y", False)),
        only_when_active=bool(c.get("only_when_active", True)),
        active_key=str(c.get("active_key", "shift")),
        toggle_key=str(c.get("toggle_key", "f8")),
        quit_key=str(c.get("quit_key", "f10")),
        control_loop_hz=int(c.get("control_loop_hz", 1000)),
        stale_target_seconds=float(c.get("stale_target_seconds", 0.16)),
        residual_epsilon=float(c.get("residual_epsilon", 0.10)),
        reset_residual_on_direction_change=bool(c.get("reset_residual_on_direction_change", True)),
        suppress_reverse_inside_deadzone=bool(c.get("suppress_reverse_inside_deadzone", True)),
        ego_scaler=float(c.get("ego_scaler", 2.7)),
        require_head_for_movement=bool(c.get("require_head_for_movement", True)),
        allow_body_fallback_control=bool(c.get("allow_body_fallback_control", False)),
        clear_residual_on_no_target=bool(c.get("clear_residual_on_no_target", True)),
        clear_residual_on_active_press=bool(c.get("clear_residual_on_active_press", True)),
        fire_enabled=bool(c.get("fire_enabled", False)),
        fire_radius=float(c.get("fire_radius", 4.0)),
        fire_exit_radius=float(c.get("fire_exit_radius", max(float(c.get("fire_radius", 4.0)) + 3.0, float(c.get("fire_radius", 4.0))))),
        fire_rearm_radius=float(c.get("fire_rearm_radius", max(float(c.get("fire_radius", 4.0)) + 5.0, float(c.get("fire_exit_radius", float(c.get("fire_radius", 4.0)) + 3.0))))),
        fire_cooldown_ms=float(c.get("fire_cooldown_ms", 165.0)),
        fire_min_conf=float(c.get("fire_min_conf", 0.50)),
        fire_stable_frames=int(c.get("fire_stable_frames", 2)),
        fire_max_target_age_ms=float(c.get("fire_max_target_age_ms", 90.0)),
        fire_allow_held_target=bool(c.get("fire_allow_held_target", False)),
        fire_repeat_while_in_radius=bool(c.get("fire_repeat_while_in_radius", False)),
        fire_reset_on_active_release=bool(c.get("fire_reset_on_active_release", True)),
        fire_log_events=bool(c.get("fire_log_events", False)),
        fire_require_zero_motion=bool(c.get("fire_require_zero_motion", True)),
        fire_max_motion_debt_px=float(c.get("fire_max_motion_debt_px", 0.90)),
        fire_min_time_after_move_ms=float(c.get("fire_min_time_after_move_ms", 22.0)),
        fire_stable_error_delta_px=float(c.get("fire_stable_error_delta_px", 2.8)),
        fire_block_during_settle_release=bool(c.get("fire_block_during_settle_release", True)),
        fire_repeat_requires_fresh_detection=bool(c.get("fire_repeat_requires_fresh_detection", True)),
        fire_min_repeat_seq_delta=int(c.get("fire_min_repeat_seq_delta", 1)),
        fire_held_target_min_conf=float(c.get("fire_held_target_min_conf", 0.72)),
        fire_held_target_max_age_ms=float(c.get("fire_held_target_max_age_ms", 45.0)),
        fire_block_on_stale_gate=bool(c.get("fire_block_on_stale_gate", True)),
        preconnect_driver=bool(c.get("preconnect_driver", True)),
        driver_connect_in_background=bool(c.get("driver_connect_in_background", True)),
    ))




def make_control_gate(cfg: Dict[str, Any]) -> ConfirmedHeadGate:
    c = cfg.get("control", {})
    s = cfg.get("selection", {})
    legacy_min = float(c.get("min_control_head_conf", s.get("head_conf", 0.38)))
    return ConfirmedHeadGate(ControlGateConfig(
        require_confirmed_frames=int(c.get("require_confirmed_frames", 2)),
        min_head_conf_enter=float(c.get("min_control_head_conf_enter", legacy_min)),
        min_head_conf_hold=float(c.get("min_control_head_conf_hold", min(legacy_min, 0.30))),
        high_conf_head=float(c.get("high_conf_head", 0.82)),
        high_conf_confirmed_frames=int(c.get("high_conf_confirmed_frames", 1)),
        max_target_jump_px=float(c.get("max_target_jump_px", 120.0)),
        active_press_delay_frames=int(c.get("active_press_delay_frames", 1)),
        reset_on_no_head=bool(c.get("reset_gate_on_no_head", False)),
        max_control_distance_px=float(c.get("max_control_distance_px", 0.0)),
        locked_target_grace_frames=int(c.get("locked_target_grace_frames", 3)),
        locked_target_grace_ms=float(c.get("locked_target_grace_ms", 70.0)),
        locked_target_max_drift_px=float(c.get("locked_target_max_drift_px", 40.0)),
        confirmed_memory_ms=float(c.get("confirmed_memory_ms", 250.0)),
        allow_missing_target_hold_control=bool(c.get("allow_missing_target_hold_control", True)),
        missing_target_hold_frames=int(c.get("missing_target_hold_frames", 2)),
        missing_target_hold_ms=float(c.get("missing_target_hold_ms", 35.0)),
        missing_target_hold_min_conf=float(c.get("missing_target_hold_min_conf", 0.30)),
        instant_enter_enabled=bool(c.get("instant_enter_enabled", True)),
        instant_enter_center_dist_px=float(c.get("instant_enter_center_dist_px", 135.0)),
        instant_enter_min_conf=float(c.get("instant_enter_min_conf", 0.32)),
        instant_enter_requires_body=bool(c.get("instant_enter_requires_body", True)),
        skip_active_delay_on_instant_target=bool(c.get("skip_active_delay_on_instant_target", True)),
        reactive_fast_enter_enabled=bool(c.get("reactive_fast_enter_enabled", True)),
        reactive_fast_enter_min_conf=float(c.get("reactive_fast_enter_min_conf", 0.70)),
        reactive_fast_enter_min_body_conf=float(c.get("reactive_fast_enter_min_body_conf", 0.62)),
        reactive_fast_enter_center_dist_px=float(c.get("reactive_fast_enter_center_dist_px", 155.0)),
        reactive_fast_enter_close_dist_px=float(c.get("reactive_fast_enter_close_dist_px", 95.0)),
        reactive_fast_enter_confirm_frames=int(c.get("reactive_fast_enter_confirm_frames", 2)),
        reactive_fast_enter_close_confirm_frames=int(c.get("reactive_fast_enter_close_confirm_frames", 1)),
        reactive_fast_enter_min_body_height_px=float(c.get("reactive_fast_enter_min_body_height_px", 44.0)),
        trust_locked_target_jump=bool(c.get("trust_locked_target_jump", True)),
        trusted_locked_jump_px=float(c.get("trusted_locked_jump_px", 95.0)),
        trusted_locked_min_conf=float(c.get("trusted_locked_min_conf", 0.45)),
        hold_on_locked_jump=bool(c.get("hold_on_locked_jump", True)),
        locked_jump_hold_frames=int(c.get("locked_jump_hold_frames", 4)),
        locked_jump_hold_ms=float(c.get("locked_jump_hold_ms", 90.0)),
        locked_jump_hold_min_conf=float(c.get("locked_jump_hold_min_conf", 0.25)),
        locked_jump_hold_max_px=float(c.get("locked_jump_hold_max_px", 160.0)),
        same_lock_jump_accept_enabled=bool(c.get("same_lock_jump_accept_enabled", True)),
        same_lock_jump_accept_px=float(c.get("same_lock_jump_accept_px", 190.0)),
        same_lock_jump_max_center_dist_px=float(c.get("same_lock_jump_max_center_dist_px", 260.0)),
        same_lock_jump_center_worse_tolerance_px=float(c.get("same_lock_jump_center_worse_tolerance_px", 80.0)),
        same_lock_jump_min_conf=float(c.get("same_lock_jump_min_conf", 0.42)),
        same_lock_jump_requires_body=bool(c.get("same_lock_jump_requires_body", True)),
        smooth_locked_target=bool(c.get("smooth_locked_target", True)),
        locked_jitter_px=float(c.get("locked_jitter_px", 2.0)),
        locked_jitter_radius_fraction=float(c.get("locked_jitter_radius_fraction", 0.10)),
        locked_jitter_alpha=float(c.get("locked_jitter_alpha", 0.18)),
        locked_smooth_alpha=float(c.get("locked_smooth_alpha", 0.55)),
        locked_slew_px_per_frame=float(c.get("locked_slew_px_per_frame", 9.0)),
        locked_slew_radius_fraction=float(c.get("locked_slew_radius_fraction", 0.55)),
        locked_snap_px=float(c.get("locked_snap_px", 42.0)),
        locked_snap_min_conf=float(c.get("locked_snap_min_conf", 0.86)),
        locked_rebase_enabled=bool(c.get("locked_rebase_enabled", True)),
        locked_rebase_px=float(c.get("locked_rebase_px", 44.0)),
        locked_rebase_radius_fraction=float(c.get("locked_rebase_radius_fraction", 1.25)),
        locked_rebase_min_conf=float(c.get("locked_rebase_min_conf", 0.58)),
        locked_rebase_requires_body=bool(c.get("locked_rebase_requires_body", True)),
        locked_rebase_max_jump_px=float(c.get("locked_rebase_max_jump_px", 150.0)),
        locked_rebase_max_center_dist_px=float(c.get("locked_rebase_max_center_dist_px", 190.0)),
        locked_rebase_center_worse_tolerance_px=float(c.get("locked_rebase_center_worse_tolerance_px", 18.0)),
        locked_smoothing_max_raw_lag_px=float(c.get("locked_smoothing_max_raw_lag_px", 3.0)),
        locked_smoothing_max_raw_lag_radius_fraction=float(c.get("locked_smoothing_max_raw_lag_radius_fraction", 0.18)),
        head_only_confirmed_frames=int(c.get("head_only_confirmed_frames", 2)),
        head_only_min_conf=float(c.get("head_only_min_conf", 0.45)),
        small_target_confirmed_frames=int(c.get("small_target_confirmed_frames", 7)),
        tiny_target_confirmed_frames=int(c.get("tiny_target_confirmed_frames", 10)),
        small_target_high_conf=float(c.get("small_target_high_conf", 0.86)),
        small_target_high_conf_frames=int(c.get("small_target_high_conf_frames", 4)),
        small_target_area_px=float(c.get("small_target_area_px", c.get("small_head_area_px", cfg.get("detection_filter", {}).get("small_head_area_px", 96.0)))),
        small_target_max_dim_px=float(c.get("small_target_max_dim_px", c.get("small_head_max_dim_px", cfg.get("detection_filter", {}).get("small_head_max_dim_px", 18.0)))),
        suspicious_body_height_px=float(c.get("suspicious_body_height_px", 42.0)),
        suspicious_body_aspect=float(c.get("suspicious_body_aspect", 0.82)),
        suspicious_target_confirmed_frames=int(c.get("suspicious_target_confirmed_frames", 8)),
    ))



def make_movement_validation_cfg(cfg: Dict[str, Any]) -> MovementValidationConfig:
    c = cfg.get("control", {})
    s = cfg.get("selection", {})
    return MovementValidationConfig(
        min_head_conf=float(c.get("min_control_head_conf_hold", c.get("min_control_head_conf", s.get("head_conf", 0.38)))),
        require_body_pair=bool(c.get("require_body_pair_for_movement", True)),
        allow_head_without_body_high_conf=bool(c.get("allow_head_without_body_high_conf", True)),
        head_without_body_conf=float(c.get("head_without_body_conf", 0.82)),
        head_without_body_max_center_dist=float(c.get("head_without_body_max_center_dist", 150.0)),
        require_head_center_near_body=bool(c.get("require_head_center_near_body", True)),
        body_expand_w_ratio=float(c.get("body_expand_w_ratio", 0.28)),
        body_expand_h_ratio=float(c.get("body_expand_h_ratio", 0.10)),
        head_upper_body_ratio_max=float(c.get("head_upper_body_ratio_max", 0.62)),
        min_head_area_px=float(c.get("min_head_area_px", 4.0)),
        min_body_area_px=float(c.get("min_body_area_px", 12.0)),
        min_body_width_px=float(c.get("min_body_width_px", cfg.get("detection_filter", {}).get("min_body_width_px", 8.0))),
        min_body_height_px=float(c.get("min_body_height_px", cfg.get("detection_filter", {}).get("min_body_height_px", 28.0))),
        max_body_aspect=float(c.get("max_body_aspect", cfg.get("detection_filter", {}).get("max_body_aspect", 0.95))),
        min_head_body_area_ratio=float(c.get("min_head_body_area_ratio", 0.002)),
        max_head_body_area_ratio=float(c.get("max_head_body_area_ratio", 0.55)),
        min_head_body_width_ratio=float(c.get("min_head_body_width_ratio", 0.025)),
        max_head_body_width_ratio=float(c.get("max_head_body_width_ratio", 0.85)),
        min_head_body_height_ratio=float(c.get("min_head_body_height_ratio", 0.025)),
        max_head_body_height_ratio=float(c.get("max_head_body_height_ratio", 0.75)),
        require_body_extends_below_head=bool(c.get("require_body_extends_below_head", cfg.get("detection_filter", {}).get("require_body_extends_below_head", True))),
        min_body_pixels_below_head=float(c.get("min_body_pixels_below_head", cfg.get("detection_filter", {}).get("min_body_pixels_below_head", 3.0))),
        max_control_distance_px=float(c.get("max_control_distance_px", 0.0)),
        small_head_area_px=float(c.get("small_head_area_px", cfg.get("detection_filter", {}).get("small_head_area_px", 96.0))),
        small_head_max_dim_px=float(c.get("small_head_max_dim_px", cfg.get("detection_filter", {}).get("small_head_max_dim_px", 18.0))),
        small_paired_min_head_conf=float(c.get("small_paired_min_head_conf", 0.18)),
        small_head_only_conf=float(c.get("small_head_only_conf", 0.78)),
        small_head_only_center_max_px=float(c.get("small_head_only_center_max_px", 150.0)),
        small_head_only_requires_body=bool(c.get("small_head_only_requires_body", True)),
        tiny_head_area_px=float(c.get("tiny_head_area_px", 24.0)),
        tiny_head_max_dim_px=float(c.get("tiny_head_max_dim_px", 7.0)),
        tiny_head_only_conf=float(c.get("tiny_head_only_conf", 0.90)),
        small_min_body_area_px=float(c.get("small_min_body_area_px", 6.0)),
        small_min_body_width_px=float(c.get("small_min_body_width_px", cfg.get("detection_filter", {}).get("small_min_body_width_px", 10.0))),
        small_min_body_height_px=float(c.get("small_min_body_height_px", cfg.get("detection_filter", {}).get("small_min_body_height_px", 34.0))),
        small_max_body_aspect=float(c.get("small_max_body_aspect", cfg.get("detection_filter", {}).get("small_max_body_aspect", 0.82))),
        small_body_expand_w_ratio=float(c.get("small_body_expand_w_ratio", cfg.get("detection_filter", {}).get("small_body_expand_w_ratio", 0.58))),
        small_body_expand_h_ratio=float(c.get("small_body_expand_h_ratio", cfg.get("detection_filter", {}).get("small_body_expand_h_ratio", 0.28))),
        small_head_upper_body_ratio_max=float(c.get("small_head_upper_body_ratio_max", cfg.get("detection_filter", {}).get("small_head_upper_body_ratio_max", 0.78))),
        small_min_head_body_area_ratio=float(c.get("small_min_head_body_area_ratio", cfg.get("detection_filter", {}).get("small_min_head_body_area_ratio", 0.00035))),
        small_max_head_body_area_ratio=float(c.get("small_max_head_body_area_ratio", cfg.get("detection_filter", {}).get("small_max_head_body_area_ratio", 0.78))),
        small_min_head_body_width_ratio=float(c.get("small_min_head_body_width_ratio", cfg.get("detection_filter", {}).get("small_min_head_body_width_ratio", 0.006))),
        small_max_head_body_width_ratio=float(c.get("small_max_head_body_width_ratio", cfg.get("detection_filter", {}).get("small_max_head_body_width_ratio", 1.15))),
        small_min_head_body_height_ratio=float(c.get("small_min_head_body_height_ratio", cfg.get("detection_filter", {}).get("small_min_head_body_height_ratio", 0.006))),
        small_max_head_body_height_ratio=float(c.get("small_max_head_body_height_ratio", cfg.get("detection_filter", {}).get("small_max_head_body_height_ratio", 1.05))),
        small_min_body_pixels_below_head=float(c.get("small_min_body_pixels_below_head", cfg.get("detection_filter", {}).get("small_min_body_pixels_below_head", 3.0))),
        border_reject_enabled=bool(c.get("border_reject_enabled", cfg.get("detection_filter", {}).get("border_reject_enabled", True))),
        border_margin_px=float(c.get("border_margin_px", cfg.get("detection_filter", {}).get("border_margin_px", 2.0))),
        border_reject_center_min_px=float(c.get("border_reject_center_min_px", cfg.get("detection_filter", {}).get("border_reject_center_min_px", 105.0))),
        border_reject_head_conf=float(c.get("border_reject_head_conf", cfg.get("detection_filter", {}).get("border_reject_head_conf", 0.88))),
        border_reject_body_conf=float(c.get("border_reject_body_conf", cfg.get("detection_filter", {}).get("border_reject_body_conf", 0.88))),
        normal_control_min_body_conf=float(c.get("normal_control_min_body_conf", 0.42)),
        small_control_min_body_conf=float(c.get("small_control_min_body_conf", 0.74)),
        small_control_far_center_px=float(c.get("small_control_far_center_px", 80.0)),
        small_control_far_min_head_conf=float(c.get("small_control_far_min_head_conf", 0.86)),
        small_control_far_min_body_conf=float(c.get("small_control_far_min_body_conf", 0.86)),
        small_control_short_body_px=float(c.get("small_control_short_body_px", 80.0)),
        small_control_short_min_head_conf=float(c.get("small_control_short_min_head_conf", 0.88)),
        small_control_short_min_body_conf=float(c.get("small_control_short_min_body_conf", 0.88)),
        small_control_tiny_body_px=float(c.get("small_control_tiny_body_px", 58.0)),
        small_control_tiny_body_reject=bool(c.get("small_control_tiny_body_reject", True)),
        small_control_max_body_aspect=float(c.get("small_control_max_body_aspect", 0.62)),
    )


def make_tracker(cfg: Dict[str, Any]):
    t = cfg.get("tracking", {})
    method = str(t.get("method", "legacy_kalman")).lower()
    if method in ("legacy", "legacy_kalman", "kalman"):
        return LegacyPointTracker(
            q=float(t.get("kalman_process_noise", 0.06)),
            r=float(t.get("kalman_measurement_noise", 0.12)),
            max_lost_frames=int(t.get("max_lost_frames", 4)),
            ego_scaler=float(cfg.get("control", {}).get("ego_scaler", 2.7)),
            hold_body_fallback_after_head_frames=int(t.get("hold_body_fallback_after_head_frames", 2)),
            kalman_max_velocity_px_s=float(t.get("kalman_max_velocity_px_s", 3400.0)),
            kalman_max_prediction_dt=float(t.get("kalman_max_prediction_dt", 0.12)),
            kalman_innovation_gate_px=float(t.get("kalman_innovation_gate_px", 145.0)),
            kalman_ego_covariance_boost=float(t.get("kalman_ego_covariance_boost", 0.18)),
        )
    return EmaPointTracker(
        alpha=float(t.get("ema_alpha", 0.45)),
        max_lost_frames=int(t.get("max_lost_frames", 4)),
    )




def target_allowed_for_control(
    cfg: Dict[str, Any],
    validation_cfg: MovementValidationConfig,
    raw_target: TargetResult,
    boxes: list[DetectionBox],
    center: tuple[float, float],
) -> tuple[bool, str]:
    """Low-level safety guard before the temporal gate.

    v12 only checked source=head and confidence. That allowed persistent false
    positives to pass. v13 adds body/head geometry validation before any HID
    movement is allowed.
    """
    c = cfg.get("control", {})
    if not raw_target.found:
        return False, "no boxes/raw target"
    # v17.6: a very short target-lock grace prediction may carry the previous
    # body/head geometry while the detector drops one frame. This is still passed
    # through the same MovementValidationConfig and is limited inside TargetLockManager.
    if not boxes:
        reason_text = str(raw_target.reason)
        if ("grace prediction" not in reason_text
                and "gate held missing target" not in reason_text
                and "gate held locked jump" not in reason_text
                and "control-side hold" not in reason_text):
            return False, "no boxes/raw target"
    if bool(c.get("require_head_for_movement", True)):
        ok, reason = validate_movement_target(
            raw_target,
            validation_cfg,
            center_x=float(center[0]),
            center_y=float(center[1]),
        )
        return ok, reason
    if raw_target.source == "body_fallback" and not bool(c.get("allow_body_fallback_control", False)):
        return False, "body fallback control disabled"
    return raw_target.source in ("head", "body_fallback"), f"accepted source={raw_target.source}"

def run_image(cfg: Dict[str, Any], detector: OnnxYoloDetector) -> None:
    source_cfg = cfg.get("capture", {})
    path = source_cfg.get("image_path", "samples/test.jpg")
    frame = load_image_bgr(path)
    center = (frame.shape[1] * 0.5, frame.shape[0] * 0.5)
    selector = make_selector(cfg, center)
    filter_cfg = make_detection_filter_cfg(cfg)
    target_lock = make_target_lock(cfg)
    tracker = make_tracker(cfg)
    boxes = detector.predict(frame)
    boxes, _filter_stats = filter_detections_by_geometry(boxes, filter_cfg, center=center, frame_shape=frame.shape)
    target = tracker.update(target_lock.select(boxes, selector, center, active=True), dt=0.0)
    vis = draw_result(frame, boxes, target, center)
    out = Path("outputs/test_image_result.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    log(f"图片测试完成: {out}; detections={len(boxes)}; target={target.to_dict()}", "SUCCESS")


def _maybe_draw(
    frame: np.ndarray,
    boxes: list[DetectionBox],
    target: TargetResult,
    center: tuple[float, float],
    window: str,
    *,
    show: bool,
    draw_every: int,
    frame_id: int,
    max_visual_fps: float,
    last_draw_time: float,
) -> tuple[bool, float]:
    if not show:
        return True, last_draw_time
    if draw_every > 1 and frame_id % draw_every != 0:
        return True, last_draw_time
    now = time.perf_counter()
    if max_visual_fps > 0 and (now - last_draw_time) < (1.0 / max_visual_fps):
        return True, last_draw_time
    vis = draw_result(frame, boxes, target, center)
    cv2.imshow(window, vis)
    last_draw_time = now
    if cv2.waitKey(1) & 0xFF == 27:
        return False, last_draw_time
    return True, last_draw_time


def run_screen(cfg: Dict[str, Any], detector: OnnxYoloDetector) -> None:
    cap_cfg = cfg.get("capture", {})
    vis_cfg = cfg.get("visual", {})
    track_cfg = cfg.get("tracking", {})
    runtime_cfg = cfg.get("runtime", {})
    logging_cfg = cfg.get("logging", {})

    width = int(cap_cfg.get("roi_width", 512))
    height = int(cap_cfg.get("roi_height", 512))
    target_fps = float(cap_cfg.get("target_fps", 120))
    cap = ScreenCapture(
        width,
        height,
        cap_cfg.get("backend", "dxcam"),
        target_fps=int(target_fps),
        max_reused_frames=int(cap_cfg.get("max_reused_frames", 2)),
        max_reused_frame_ms=float(cap_cfg.get("max_reused_frame_ms", 20.0)),
        restart_after_empty_frames=int(cap_cfg.get("restart_after_empty_frames", 8)),
        fallback_after_restarts=int(cap_cfg.get("fallback_after_restarts", 2)),
    )
    center = (cap.region.width * 0.5, cap.region.height * 0.5)
    global_center = cap.region.center

    selector = make_selector(cfg, center)
    filter_cfg = make_detection_filter_cfg(cfg)
    target_lock = make_target_lock(cfg)
    tracker = make_tracker(cfg)
    gate = make_control_gate(cfg)
    validation_cfg = make_movement_validation_cfg(cfg)
    controller = make_controller(cfg)

    threaded_capture = bool(runtime_cfg.get("threaded_capture", True))
    drop_stale_frames = bool(runtime_cfg.get("drop_stale_frames", True))
    infer_fps_limit = float(runtime_cfg.get("infer_fps_limit", 0.0) or 0.0)
    idle_sleep_ms = float(runtime_cfg.get("idle_sleep_ms", 1.0))

    show = bool(vis_cfg.get("show_window", True))
    window = str(vis_cfg.get("window_name", "vendor_onnx_runtime"))
    draw_every = max(1, int(vis_cfg.get("draw_every", 1)))
    max_visual_fps = float(vis_cfg.get("max_fps", 0.0) or 0.0)

    print_every = max(1, int(logging_cfg.get("print_every", 60)))
    profile_enabled = bool(logging_cfg.get("profile", True))
    profile_every = max(1, int(logging_cfg.get("profile_every", 120)))
    profiler = RollingProfiler(profile_enabled, int(logging_cfg.get("profile_window", 120)))

    frame_reader = None
    if threaded_capture:
        frame_reader = LatestFrameReader(cap, target_fps=target_fps, idle_sleep_ms=idle_sleep_ms)
        frame_reader.start()
        if not frame_reader.wait_first(timeout=2.0):
            raise RuntimeError("capture thread did not produce a first frame")
    else:
        log("capture thread disabled: using synchronous capture", "WARN")

    frame_id = 0
    processed_seq = -1
    active_prev = False
    last_gate_reason = "init"
    last_log = time.perf_counter()
    last_infer = 0.0
    last_draw_time = 0.0
    last_frame_ts = time.perf_counter()
    skipped_stale = 0

    # v16 file-log statistics. These counters intentionally stay lightweight and
    # are written only at configured intervals and at shutdown.
    run_start = time.perf_counter()
    frames_with_boxes = 0
    frames_with_head = 0
    frames_with_body = 0
    frames_control_allowed = 0
    frames_movement_ready = 0
    frames_base_ok = 0
    frames_gate_ok = 0
    frames_raw_head = 0
    frames_raw_body_fallback = 0
    frames_control_switch_on = 0
    frames_driver_ready = 0
    frames_active = 0
    max_boxes_seen = 0
    max_head_conf_seen = 0.0
    max_body_conf_seen = 0.0
    total_boxes_seen = 0
    gate_reason_counter: Counter[str] = Counter()
    base_reason_counter: Counter[str] = Counter()
    lock_reason_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    active_transition_counter: Counter[str] = Counter()
    last_active_logged = False
    last_control_ok_logged = False
    last_movement_ready_logged = False
    last_raw_source_logged = "none"
    rolling_summary_every_sec = float(logging_cfg.get("rolling_summary_every_sec", 10.0) or 0.0)
    last_rolling_summary = run_start
    log_body_fallback_events = bool(logging_cfg.get("log_body_fallback_events", False))
    log_none_raw_events = bool(logging_cfg.get("log_none_raw_events", False))
    log_summary_at_exit = bool(logging_cfg.get("log_summary_at_exit", True))
    summary_written = False
    stop_requested = False

    def write_run_summary(label: str) -> None:
        nonlocal summary_written
        elapsed = max(time.perf_counter() - run_start, 1e-6)
        avg_fps = frame_id / elapsed
        log(f"{label}_BEGIN", "INFO")
        log_kv(label, {
            "elapsed_sec": round(elapsed, 3),
            "processed_frames": frame_id,
            "avg_fps": round(avg_fps, 2),
            "frames_with_boxes": frames_with_boxes,
            "frames_with_head": frames_with_head,
            "frames_with_body": frames_with_body,
            "frames_active": frames_active,
            "frames_control_allowed": frames_control_allowed,
            "frames_movement_ready": frames_movement_ready,
            "frames_base_ok": frames_base_ok,
            "frames_gate_ok": frames_gate_ok,
            "frames_raw_head": frames_raw_head,
            "frames_raw_body_fallback": frames_raw_body_fallback,
            "frames_control_switch_on": frames_control_switch_on,
            "frames_driver_ready": frames_driver_ready,
            "total_boxes_seen": total_boxes_seen,
            "max_boxes_seen": max_boxes_seen,
            "max_head_conf_seen": round(max_head_conf_seen, 4),
            "max_body_conf_seen": round(max_body_conf_seen, 4),
            "stale_frame_skips": skipped_stale,
            "active_ratio": round(frames_active / max(frame_id, 1), 4),
            "control_allowed_ratio": round(frames_control_allowed / max(frame_id, 1), 4),
            "movement_ready_ratio": round(frames_movement_ready / max(frame_id, 1), 4),
            "driver_ready_ratio": round(frames_driver_ready / max(frame_id, 1), 4),
        })
        log_kv(f"{label}_SOURCE_COUNTS", dict(source_counter))
        log_kv(f"{label}_BASE_REASON_TOP", dict(base_reason_counter.most_common(20)))
        log_kv(f"{label}_GATE_REASON_TOP", dict(gate_reason_counter.most_common(20)))
        log_kv(f"{label}_LOCK_REASON_TOP", dict(lock_reason_counter.most_common(20)))
        log_kv(f"{label}_ACTIVE_TRANSITIONS", dict(active_transition_counter))
        if profile_enabled:
            log(f"{label}_PROFILE: " + profiler.summary([
                "loop_total",
                "capture",
                "ego",
                "det.pre_ms",
                "det.infer_ms",
                "det.post_ms",
                "infer_total",
                "select_gate",
                "control_submit",
                "draw",
            ]), "INFO")
        log(f"{label}_END", "INFO")
        if label == "RUN_SUMMARY":
            summary_written = True

    def _atexit_summary() -> None:
        if log_summary_at_exit and not summary_written:
            try:
                write_run_summary("RUN_SUMMARY_ATEXIT")
                log("SESSION_END_ATEXIT", "WARN")
            except Exception:
                pass

    atexit.register(_atexit_summary)

    def _signal_stop_handler(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        log(f"SIGNAL_RECEIVED signum={signum}; graceful shutdown requested", "WARN")

    _old_signal_handlers = {}
    for _sig in [getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)]:
        if _sig is None:
            continue
        try:
            _old_signal_handlers[_sig] = signal.getsignal(_sig)
            signal.signal(_sig, _signal_stop_handler)
        except Exception:
            pass

    warmup_frames = int(runtime_cfg.get("warmup_inference_frames", 20) or 0)
    if warmup_frames > 0:
        try:
            warm_captured = frame_reader.get_latest() if frame_reader is not None else None
            warm_frame = warm_captured.frame if warm_captured is not None else cap.read()
            log(f"ONNX_WARMUP_BEGIN frames={warmup_frames}", "INFO")
            t_warm = time.perf_counter()
            for _ in range(warmup_frames):
                detector.predict(warm_frame)
            log(f"ONNX_WARMUP_END elapsed_ms={(time.perf_counter() - t_warm) * 1000.0:.1f}", "INFO")
        except Exception as e:
            log(f"ONNX_WARMUP_FAILED: {e}", "WARN")

    log(
        "进入实时循环：threaded_capture=%s, show_window=%s, F8 切换控制；F10 退出；默认只在 active_key 按下时移动"
        % (threaded_capture, show),
        "SUCCESS",
    )

    try:
        while not stop_requested:
            if EXTERNAL_STOP_EVENT is not None and getattr(EXTERNAL_STOP_EVENT, "is_set", lambda: False)():
                log("GUI_STOP_EVENT_RECEIVED; graceful shutdown requested", "WARN")
                break
            loop_start = time.perf_counter()
            if not controller.poll_hotkeys():
                break

            if infer_fps_limit > 0 and (loop_start - last_infer) < (1.0 / infer_fps_limit):
                time.sleep(min(max(0.0, 1.0 / infer_fps_limit - (loop_start - last_infer)), 0.003))
                continue

            with profiler.stage("capture"):
                if frame_reader is not None:
                    captured = frame_reader.get_latest()
                    if captured is None:
                        time.sleep(0.001)
                        continue
                    if drop_stale_frames and captured.seq == processed_seq:
                        skipped_stale += 1
                        time.sleep(0.001)
                        continue
                    processed_seq = captured.seq
                    frame = captured.frame
                    capture_ts = captured.timestamp
                else:
                    frame = cap.read()
                    capture_ts = time.perf_counter()

            # v17.5: count the processed frame before logging per-frame events.
            # Earlier versions logged event frame ids one step behind STATUS/RUN_SUMMARY.
            frame_id += 1

            last_infer = time.perf_counter()
            dt_frame = max(0.0, capture_ts - last_frame_ts)
            last_frame_ts = capture_ts

            with profiler.stage("ego"):
                ego_dx, ego_dy = controller.consume_ego_delta()
                if (ego_dx or ego_dy) and bool(track_cfg.get("enabled", True)) and hasattr(tracker, "apply_ego_motion"):
                    tracker.apply_ego_motion(ego_dx, ego_dy)
                if (ego_dx or ego_dy) and hasattr(gate, "apply_ego_motion"):
                    gate.apply_ego_motion(ego_dx, ego_dy, float(cfg.get("control", {}).get("ego_scaler", 2.7)))

            with profiler.stage("infer_total"):
                if hasattr(detector, "predict_with_profile"):
                    boxes, det_profile = detector.predict_with_profile(frame)
                    profiler.add_many("det", det_profile)
                else:
                    boxes = detector.predict(frame)
                boxes, _filter_stats = filter_detections_by_geometry(
                    boxes, filter_cfg, center=center, frame_shape=frame.shape
                )

            with profiler.stage("select_gate"):
                active_now = controller.is_active()
                if active_now and not active_prev:
                    gate.on_active_rising()
                    target_lock.on_active_rising()
                    controller.clear_target()
                    if bool(cfg.get("control", {}).get("reset_tracker_on_active_press", True)) and hasattr(tracker, "reset"):
                        tracker.reset()
                active_prev = active_now

                if active_now != last_active_logged:
                    active_transition_counter["active_on" if active_now else "active_off"] += 1
                    log(f"EVENT active_key={'DOWN' if active_now else 'UP'} frame={frame_id}", "INFO")
                    last_active_logged = active_now

                raw_target = target_lock.select(boxes, selector, center, active=active_now)
                raw_for_log = _clone_target(raw_target)

                # v17.7 logic fix: use tracker output for the control chain.
                # v17.8.2 correction: tracker.update mutates its TargetResult in place.
                # Passing raw_target directly corrupted raw logging and, more importantly,
                # let Kalman lag become the actual aim point. Work on a clone and then
                # clamp the control point back toward the current detected head center.
                if bool(track_cfg.get("enabled", True)):
                    target = tracker.update(_clone_target(raw_target), dt=dt_frame)
                    target = _correct_control_point(raw_for_log, target, cfg, center)
                else:
                    target = raw_for_log

                # v17.8: run the temporal gate before geometry validation.
                # V17.7 validated the raw/tracked target first; when one frame was
                # missing, base_ok became False before the gate could supply its short
                # confirmed-target hold. That produced movement_ready True/False flicker
                # even though gate memory was explicitly saying "holding".
                gate_ok, gated_target, gate_reason = gate.update(
                    target,
                    active=active_now,
                    center_x=center[0],
                    center_y=center[1],
                )
                base_ok, base_reason = target_allowed_for_control(cfg, validation_cfg, gated_target, boxes, center)
                if hasattr(gate, "on_validation_result"):
                    gate.on_validation_result(bool(base_ok), gated_target)
                control_ok = bool(gate_ok and base_ok and gated_target.found)
                last_gate_reason = gate_reason if gate_ok else base_reason

                # v16 counters for later offline analysis from the .txt file.
                bs = _box_summary(boxes)
                if bs["boxes"] > 0:
                    frames_with_boxes += 1
                if bs["heads"] > 0:
                    frames_with_head += 1
                if bs["bodies"] > 0:
                    frames_with_body += 1
                driver_status_now = controller.driver_status()
                movement_ready = bool(control_ok and controller.enabled and active_now and driver_status_now == "ready")
                if base_ok:
                    frames_base_ok += 1
                if gate_ok:
                    frames_gate_ok += 1
                if raw_for_log.found and raw_for_log.source == "head":
                    frames_raw_head += 1
                if raw_for_log.found and raw_for_log.source == "body_fallback":
                    frames_raw_body_fallback += 1
                if controller.enabled:
                    frames_control_switch_on += 1
                if driver_status_now == "ready":
                    frames_driver_ready += 1
                if control_ok:
                    frames_control_allowed += 1
                if movement_ready:
                    frames_movement_ready += 1
                if active_now:
                    frames_active += 1
                max_boxes_seen = max(max_boxes_seen, int(bs["boxes"]))
                max_head_conf_seen = max(max_head_conf_seen, float(bs["max_head_conf"]))
                max_body_conf_seen = max(max_body_conf_seen, float(bs["max_body_conf"]))
                total_boxes_seen += int(bs["boxes"])
                gate_reason_counter[str(gate_reason)] += 1
                base_reason_counter[str(base_reason)] += 1
                lock_reason_counter[str(target_lock.last_reason)] += 1
                source_counter[str(raw_for_log.source if raw_for_log.found else "none")] += 1
                raw_source_now = raw_for_log.source if raw_for_log.found else "none"
                if raw_source_now != last_raw_source_logged:
                    should_log_raw_event = (
                        raw_source_now == "head"
                        or last_raw_source_logged == "head"
                        or (raw_source_now == "body_fallback" and log_body_fallback_events)
                        or (raw_source_now == "none" and log_none_raw_events)
                    )
                    if should_log_raw_event:
                        log(f"EVENT raw_source_change frame={frame_id}, from={last_raw_source_logged}, to={raw_source_now}, target={_target_short(raw_for_log, center)}", "INFO")
                    last_raw_source_logged = raw_source_now
                if control_ok != last_control_ok_logged:
                    log(f"EVENT control_allowed_change frame={frame_id}, control_allowed={control_ok}, movement_ready={movement_ready}, active={active_now}, driver={driver_status_now}, base_ok={base_ok}, base_reason={base_reason}, gate_ok={gate_ok}, gate_reason={gate_reason}, raw={_target_short(raw_for_log, center)}, gated={_target_short(gated_target, center)}", "INFO")
                    last_control_ok_logged = control_ok
                if movement_ready != last_movement_ready_logged:
                    log(f"EVENT movement_ready_change frame={frame_id}, movement_ready={movement_ready}, control_allowed={control_ok}, active={active_now}, control={'ON' if controller.enabled else 'OFF'}, driver={driver_status_now}, raw={_target_short(raw_for_log, center)}, gated={_target_short(gated_target, center)}", "INFO")
                    last_movement_ready_logged = movement_ready

                held_for_control = any(token in str(gated_target.reason) for token in (
                    "gate held missing target",
                    "gate held locked jump",
                    "locked target grace prediction",
                    "held predicted head",
                ))

                if control_ok and gated_target.found and gated_target.source == "head":
                    if "control rebase snap" in str(gated_target.reason):
                        controller.clear_residual()
                        if hasattr(controller, "_reset_settle_lock"):
                            controller._reset_settle_lock()
                    global_x = cap.region.left + gated_target.x
                    global_y = cap.region.top + gated_target.y
                    distance = math.hypot(global_x - global_center[0], global_y - global_center[1])
                    head_radius = 0.0
                    if gated_target.head_box is not None:
                        head_radius = max(float(gated_target.head_box.w), float(gated_target.head_box.h)) * 0.5
                    controller.submit(
                        global_x,
                        global_y,
                        global_center[0],
                        global_center[1],
                        distance=distance,
                        confidence=gated_target.confidence,
                        valid=True,
                        held=held_for_control,
                        target_radius=head_radius,
                    )
                else:
                    controller.clear_target(soft=True, active=active_now)

            with profiler.stage("draw"):
                keep_running, last_draw_time = _maybe_draw(
                    frame,
                    boxes,
                    target,
                    center,
                    window,
                    show=show,
                    draw_every=draw_every,
                    frame_id=frame_id,
                    max_visual_fps=max_visual_fps,
                    last_draw_time=last_draw_time,
                )
                if not keep_running:
                    break

            profiler.add("loop_total", (time.perf_counter() - loop_start) * 1000.0)

            if frame_id % print_every == 0:
                now = time.perf_counter()
                dt = now - last_log
                fps = print_every / max(dt, 1e-6)
                last_log = now
                bs = _box_summary(boxes)
                log(
                    f"STATUS frame={frame_id}, fps={fps:.1f}, active={active_now}, control={'ON' if controller.enabled else 'OFF'}, "
                    f"control_allowed={control_ok}, movement_ready={movement_ready}, driver={driver_status_now}, boxes={bs}, "
                    f"filter={{'in': {getattr(_filter_stats, 'input_boxes', 0)}, 'out': {getattr(_filter_stats, 'output_boxes', 0)}, 'head_rej': {getattr(_filter_stats, 'heads_rejected', 0)}, 'body_rej': {getattr(_filter_stats, 'bodies_rejected', 0)}}}, "
                    f"raw={_target_short(raw_for_log, center)}, "
                    f"tracked={_target_short(target, center)}, gated={_target_short(gated_target, center)}, base_ok={base_ok}, base_reason={base_reason}, "
                    f"gate_ok={gate_ok}, gate={last_gate_reason}, lock={target_lock.last_reason}, "
                    f"lock_lost={target_lock.lost_frames}, stale_skips={skipped_stale}"
                )

            if profile_enabled and frame_id % profile_every == 0:
                log(
                    "profile: " + profiler.summary([
                        "loop_total",
                        "capture",
                        "ego",
                        "det.pre_ms",
                        "det.infer_ms",
                        "det.post_ms",
                        "infer_total",
                        "select_gate",
                        "control_submit",
                        "draw",
                    ]),
                    "INFO",
                )

            if rolling_summary_every_sec > 0 and (time.perf_counter() - last_rolling_summary) >= rolling_summary_every_sec:
                write_run_summary("ROLLING_SUMMARY")
                last_rolling_summary = time.perf_counter()
    finally:
        for _sig, _old in _old_signal_handlers.items():
            try:
                signal.signal(_sig, _old)
            except Exception:
                pass
        write_run_summary("RUN_SUMMARY")
        log("SESSION_END", "INFO")
        controller.close()
        if frame_reader is not None:
            frame_reader.stop()
        else:
            cap.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        log("实时循环已退出", "WARN")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", choices=["screen", "image", "video"], default=None)
    ap.add_argument("--control", choices=["on", "off", "config"], default="config")
    ap.add_argument("--visual", choices=["on", "off", "config"], default="config")
    ap.add_argument("--profile", choices=["on", "off", "config"], default="config")
    ap.add_argument("--threaded-capture", choices=["on", "off", "config"], default="config")
    ap.add_argument("--capture-backend", choices=["config", "dxcam", "dxcam_auto", "mss"], default="config")
    ap.add_argument("--console-log", choices=["on", "off", "config"], default="config")
    ap.add_argument("--log-file", default=None, help="Reserved for future use; v17 writes timestamped logs under logging.log_dir")
    args = ap.parse_args()
    configure_dll_search_path()
    cfg = load_cfg(args.config)
    if args.console_log != "config":
        cfg.setdefault("logging", {})["console"] = (args.console_log == "on")
    init_logging(cfg)
    if args.source:
        cfg.setdefault("capture", {})["source"] = args.source
    if args.capture_backend != "config":
        cfg.setdefault("capture", {})["backend"] = args.capture_backend
        log(f"capture backend override from command line: {args.capture_backend}", "WARN")
    if args.control != "config":
        cfg.setdefault("control", {})["enabled"] = (args.control == "on")
        log(f"control override from command line: {args.control}", "WARN")
    if args.visual != "config":
        cfg.setdefault("visual", {})["show_window"] = (args.visual == "on")
        log(f"visual override from command line: {args.visual}", "WARN")
    if args.profile != "config":
        cfg.setdefault("logging", {})["profile"] = (args.profile == "on")
        log(f"profile override from command line: {args.profile}", "WARN")
    if args.threaded_capture != "config":
        cfg.setdefault("runtime", {})["threaded_capture"] = (args.threaded_capture == "on")
        log(f"threaded-capture override from command line: {args.threaded_capture}", "WARN")
    log_session_header(cfg, args)
    detector = make_detector(cfg)
    source = cfg.get("capture", {}).get("source", "screen")
    if source == "image":
        run_image(cfg, detector)
    elif source == "screen":
        run_screen(cfg, detector)
    else:
        raise NotImplementedError("video 模式预留，当前请使用 screen 或 image")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        log_exception()
        raise
    finally:
        close_logging()
