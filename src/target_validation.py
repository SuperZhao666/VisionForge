
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .types import DetectionBox, TargetResult


@dataclass(frozen=True)
class MovementValidationConfig:
    """Geometry and confidence validation before HID movement.

    This validator solves the v12 problem where a persistent false-positive head
    could pass the consecutive-frame gate. Consecutive frames alone cannot
    distinguish a real target from a stable false positive on UI/wall/weapon.

    Movement therefore requires either:
    - a plausible head+body pair; or
    - an optional very-high-confidence head-only target near the center.
    """

    min_head_conf: float = 0.38
    require_body_pair: bool = True
    allow_head_without_body_high_conf: bool = True
    head_without_body_conf: float = 0.82
    head_without_body_max_center_dist: float = 150.0

    # Body/head geometry checks. These are intentionally tolerant for small/far targets.
    require_head_center_near_body: bool = True
    body_expand_w_ratio: float = 0.28
    body_expand_h_ratio: float = 0.10
    head_upper_body_ratio_max: float = 0.62
    min_head_area_px: float = 4.0
    min_body_area_px: float = 12.0
    min_body_width_px: float = 8.0
    min_body_height_px: float = 28.0
    max_body_aspect: float = 0.95
    min_head_body_area_ratio: float = 0.002
    max_head_body_area_ratio: float = 0.55
    min_head_body_width_ratio: float = 0.025
    max_head_body_width_ratio: float = 0.85
    min_head_body_height_ratio: float = 0.025
    max_head_body_height_ratio: float = 0.75
    require_body_extends_below_head: bool = True
    min_body_pixels_below_head: float = 3.0
    max_control_distance_px: float = 0.0

    # V17.8.9: adaptive small/far target validation. Geometry is still strict for
    # normal targets, but small distant heads often have weak/missing bodies.
    small_head_area_px: float = 96.0
    small_head_max_dim_px: float = 18.0
    small_paired_min_head_conf: float = 0.18
    small_head_only_conf: float = 0.78
    small_head_only_center_max_px: float = 150.0
    # V17.8.10: micro map lights often look like tiny head-only boxes.
    # To make Shift absolutely safe, a small/far head without a matched body is
    # not allowed to drive the motor unless explicitly disabled in config.
    small_head_only_requires_body: bool = True
    tiny_head_area_px: float = 24.0
    tiny_head_max_dim_px: float = 7.0
    tiny_head_only_conf: float = 0.90
    small_min_body_area_px: float = 6.0
    small_min_body_width_px: float = 10.0
    small_min_body_height_px: float = 34.0
    small_max_body_aspect: float = 0.82
    small_body_expand_w_ratio: float = 0.58
    small_body_expand_h_ratio: float = 0.28
    small_head_upper_body_ratio_max: float = 0.78
    small_min_head_body_area_ratio: float = 0.00035
    small_max_head_body_area_ratio: float = 0.78
    small_min_head_body_width_ratio: float = 0.006
    small_max_head_body_width_ratio: float = 1.15
    small_min_head_body_height_ratio: float = 0.006
    small_max_head_body_height_ratio: float = 1.05
    small_min_body_pixels_below_head: float = 3.0

    # V17.8.11: movement-layer edge and micro-pair hard rejection.
    # Detection may still draw a box, but these candidates cannot drive HID.
    border_reject_enabled: bool = True
    border_margin_px: float = 2.0
    border_reject_center_min_px: float = 105.0
    border_reject_head_conf: float = 0.88
    border_reject_body_conf: float = 0.88

    # V17.8.12: final movement fail-safe. The detector may still draw small
    # head/body pairs, but HID movement requires stronger evidence. This is the
    # last line of defense against map lights / tiny geometry being treated as a
    # controllable human target.
    normal_control_min_body_conf: float = 0.42
    small_control_min_body_conf: float = 0.74
    small_control_far_center_px: float = 80.0
    small_control_far_min_head_conf: float = 0.86
    small_control_far_min_body_conf: float = 0.86
    small_control_short_body_px: float = 80.0
    small_control_short_min_head_conf: float = 0.88
    small_control_short_min_body_conf: float = 0.88
    small_control_tiny_body_px: float = 58.0
    small_control_tiny_body_reject: bool = True
    small_control_max_body_aspect: float = 0.62


def _expanded_contains(box: DetectionBox, x: float, y: float, expand_w: float, expand_h: float) -> bool:
    ew = box.w * expand_w
    eh = box.h * expand_h
    return (box.x1 - ew) <= x <= (box.x2 + ew) and (box.y1 - eh) <= y <= (box.y2 + eh)


def _is_small_head(head: DetectionBox, cfg: MovementValidationConfig) -> bool:
    return (
        head.area <= float(cfg.small_head_area_px)
        or max(head.w, head.h) <= float(cfg.small_head_max_dim_px)
    )


def _body_aspect(body: DetectionBox) -> float:
    return body.w / max(body.h, 1e-6)


def _touches_border(box: DetectionBox, roi_w: float, roi_h: float, margin: float) -> bool:
    return (
        box.x1 <= margin
        or box.y1 <= margin
        or box.x2 >= roi_w - margin
        or box.y2 >= roi_h - margin
    )


def validate_movement_target(
    target: TargetResult,
    cfg: MovementValidationConfig,
    *,
    center_x: float,
    center_y: float,
) -> Tuple[bool, str]:
    if not target.found:
        return False, "no target"
    if target.source != "head" or target.head_box is None:
        return False, f"movement requires head, got {target.source}"

    head = target.head_box
    body = target.body_box
    conf = float(target.confidence)
    small = _is_small_head(head, cfg)
    min_head_conf = min(float(cfg.min_head_conf), float(cfg.small_paired_min_head_conf)) if small else float(cfg.min_head_conf)
    if conf < min_head_conf:
        return False, f"head conf {conf:.2f} < {min_head_conf:.2f}"

    hx, hy = head.center
    center_dist = math.hypot(hx - center_x, hy - center_y)
    if cfg.max_control_distance_px and cfg.max_control_distance_px > 0 and center_dist > cfg.max_control_distance_px:
        return False, f"target center distance {center_dist:.1f}px > {cfg.max_control_distance_px:.1f}px"

    if head.area < float(cfg.min_head_area_px):
        return False, f"head area too small: {head.area:.1f}px"

    if body is None:
        if not cfg.allow_head_without_body_high_conf:
            return False, "no matched body"
        tiny = head.area <= float(cfg.tiny_head_area_px) or max(head.w, head.h) <= float(cfg.tiny_head_max_dim_px)
        if tiny and conf < float(cfg.tiny_head_only_conf):
            return False, f"tiny head-only rejected: area={head.area:.1f}, conf={conf:.2f} < {cfg.tiny_head_only_conf:.2f}"
        if small and bool(cfg.small_head_only_requires_body):
            return False, "small head-only rejected: no matched body"
        if small:
            needed_conf = float(cfg.small_head_only_conf)
            max_dist = float(cfg.small_head_only_center_max_px)
        else:
            needed_conf = float(cfg.head_without_body_conf)
            max_dist = float(cfg.head_without_body_max_center_dist)
        if conf < needed_conf:
            return False, f"head-only conf {conf:.2f} < {needed_conf:.2f}"
        if center_dist > max_dist:
            return False, f"head-only too far from center: {center_dist:.1f}px"
        return True, f"head-only accepted: conf={conf:.2f}, dist={center_dist:.1f}, small={small}, tiny={tiny}"

    min_body_area = float(cfg.small_min_body_area_px) if small else float(cfg.min_body_area_px)
    if body.area < min_body_area:
        return False, f"body area too small: {body.area:.1f}px"

    min_body_w = float(cfg.small_min_body_width_px) if small else float(cfg.min_body_width_px)
    min_body_h = float(cfg.small_min_body_height_px) if small else float(cfg.min_body_height_px)
    max_body_ar = float(cfg.small_max_body_aspect) if small else float(cfg.max_body_aspect)
    if body.w < min_body_w or body.h < min_body_h:
        return False, f"body box too small for movement: w={body.w:.1f}, h={body.h:.1f}, need>={min_body_w:.1f}x{min_body_h:.1f}, small={small}"
    body_ar = _body_aspect(body)
    if body_ar > max_body_ar:
        return False, f"body aspect too wide for human target: {body_ar:.2f} > {max_body_ar:.2f}, small={small}"

    min_body_conf_for_move = float(cfg.small_control_min_body_conf) if small else float(cfg.normal_control_min_body_conf)
    if float(body.conf) < min_body_conf_for_move:
        return False, f"body conf too weak for movement: {body.conf:.2f} < {min_body_conf_for_move:.2f}, small={small}"

    if small:
        if body_ar > float(cfg.small_control_max_body_aspect):
            return False, f"small body aspect too wide for control: {body_ar:.2f} > {cfg.small_control_max_body_aspect:.2f}"
        if bool(cfg.small_control_tiny_body_reject) and body.h < float(cfg.small_control_tiny_body_px):
            return False, f"small body too short for safe movement: h={body.h:.1f} < {cfg.small_control_tiny_body_px:.1f}"
        if body.h < float(cfg.small_control_short_body_px):
            if not (conf >= float(cfg.small_control_short_min_head_conf) and float(body.conf) >= float(cfg.small_control_short_min_body_conf)):
                return False, (
                    f"short small-pair rejected: body_h={body.h:.1f} < {cfg.small_control_short_body_px:.1f}, "
                    f"head_conf={conf:.2f}, body_conf={body.conf:.2f}"
                )
        if center_dist > float(cfg.small_control_far_center_px):
            if not (conf >= float(cfg.small_control_far_min_head_conf) and float(body.conf) >= float(cfg.small_control_far_min_body_conf)):
                return False, (
                    f"far small-pair rejected: dist={center_dist:.1f} > {cfg.small_control_far_center_px:.1f}, "
                    f"head_conf={conf:.2f}, body_conf={body.conf:.2f}"
                )

    # Hard reject edge-clamped boxes unless they are very confident and near the center.
    # ROI is centered, so frame dimensions are approximately 2*center.
    roi_w = float(center_x) * 2.0
    roi_h = float(center_y) * 2.0
    if bool(cfg.border_reject_enabled) and (_touches_border(head, roi_w, roi_h, float(cfg.border_margin_px)) or _touches_border(body, roi_w, roi_h, float(cfg.border_margin_px))):
        far_from_center = center_dist >= float(cfg.border_reject_center_min_px)
        not_very_strong = (float(head.conf) < float(cfg.border_reject_head_conf) or float(body.conf) < float(cfg.border_reject_body_conf))
        if far_from_center and not_very_strong:
            return False, f"edge-clamped target rejected: dist={center_dist:.1f}, head_conf={head.conf:.2f}, body_conf={body.conf:.2f}"

    if small:
        expand_w = float(cfg.small_body_expand_w_ratio)
        expand_h = float(cfg.small_body_expand_h_ratio)
        upper_ratio = float(cfg.small_head_upper_body_ratio_max)
        min_area_ratio = float(cfg.small_min_head_body_area_ratio)
        max_area_ratio = float(cfg.small_max_head_body_area_ratio)
        min_wr = float(cfg.small_min_head_body_width_ratio)
        max_wr = float(cfg.small_max_head_body_width_ratio)
        min_hr = float(cfg.small_min_head_body_height_ratio)
        max_hr = float(cfg.small_max_head_body_height_ratio)
        min_below = float(cfg.small_min_body_pixels_below_head)
    else:
        expand_w = float(cfg.body_expand_w_ratio)
        expand_h = float(cfg.body_expand_h_ratio)
        upper_ratio = float(cfg.head_upper_body_ratio_max)
        min_area_ratio = float(cfg.min_head_body_area_ratio)
        max_area_ratio = float(cfg.max_head_body_area_ratio)
        min_wr = float(cfg.min_head_body_width_ratio)
        max_wr = float(cfg.max_head_body_width_ratio)
        min_hr = float(cfg.min_head_body_height_ratio)
        max_hr = float(cfg.max_head_body_height_ratio)
        min_below = float(cfg.min_body_pixels_below_head)

    if cfg.require_head_center_near_body and not _expanded_contains(body, hx, hy, expand_w, expand_h):
        return False, "head center is not near matched body"

    rel_y = (hy - body.y1) / max(body.h, 1e-6)
    if rel_y > upper_ratio:
        return False, f"head too low in body: rel_y={rel_y:.2f}"
    if bool(cfg.require_body_extends_below_head) and (body.y2 - hy) < min_below:
        return False, f"body does not extend below head enough: below={body.y2 - hy:.1f}px < {min_below:.1f}px"

    area_ratio = head.area / max(body.area, 1e-6)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False, f"head/body area ratio invalid: {area_ratio:.3f}"

    wr = head.w / max(body.w, 1e-6)
    hr = head.h / max(body.h, 1e-6)
    if wr < min_wr or wr > max_wr:
        return False, f"head/body width ratio invalid: {wr:.3f}"
    if hr < min_hr or hr > max_hr:
        return False, f"head/body height ratio invalid: {hr:.3f}"

    return True, f"valid head-body pair: conf={conf:.2f}, rel_y={rel_y:.2f}, area_ratio={area_ratio:.3f}, small={small}"
