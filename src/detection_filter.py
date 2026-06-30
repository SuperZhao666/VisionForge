from __future__ import annotations

"""Adaptive geometry filters for suppressing map-model false positives.

V17.8.8 added a strict geometry filter. It reduced map/prop false positives, but
it could also reject real small/far targets because those targets often have:

- very small head boxes;
- weak or missing body boxes;
- unstable head/body area ratios.

V17.8.12 keeps the same idea, but adds anti-map fail-safe strictness:

- normal/large targets still need strict human geometry;
- small/far heads use a relaxed geometry profile;
- head-only small targets may pass only when confidence is adequate and they are
  near the control center.

This module only filters boxes already emitted by the ONNX detector. It does not
run any image model, so it stays cheap and deterministic.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

from .types import DetectionBox


@dataclass(frozen=True)
class DetectionGeometryFilterConfig:
    enabled: bool = True

    body_class_id: int = 0
    head_class_id: int = 1

    # Base confidence rules.
    min_head_conf: float = 0.18
    min_body_conf: float = 0.26
    paired_head_min_conf: float = 0.20
    head_only_min_conf: float = 0.66
    head_only_center_max_px: float = 175.0
    keep_unpaired_bodies: bool = True

    # Small/far target adaptive path. This is deliberately weaker than the
    # ordinary head-only rule, but bounded by center distance and size.
    small_head_area_px: float = 96.0
    small_head_max_dim_px: float = 18.0
    small_head_min_conf: float = 0.20
    small_paired_head_min_conf: float = 0.18
    small_head_only_min_conf: float = 0.32
    small_head_only_center_max_px: float = 230.0
    small_body_min_conf: float = 0.18
    small_relaxed_pair: bool = True

    # Basic head shape checks.
    min_head_area_px: float = 3.0
    max_head_area_px: float = 2600.0
    min_head_width_px: float = 1.5
    min_head_height_px: float = 1.5
    max_head_width_px: float = 96.0
    max_head_height_px: float = 96.0
    min_head_aspect: float = 0.30
    max_head_aspect: float = 2.55

    # Basic body shape checks. V17.8.11 adds minimum body width/height and
    # stricter aspect limits because map props/lights often generate a tiny
    # "body" box that technically pairs with a false head.
    min_body_area_px: float = 10.0
    max_body_area_ratio: float = 0.82
    min_body_width_px: float = 8.0
    min_body_height_px: float = 26.0
    small_min_body_width_px: float = 10.0
    small_min_body_height_px: float = 34.0
    min_body_aspect: float = 0.10
    max_body_aspect: float = 1.05
    small_max_body_aspect: float = 0.82

    # Reject crop-edge ghosts. False positives from wall/prop edges commonly
    # clamp to ROI borders. Do not let them enter target selection unless they
    # are near the center and very plausible.
    border_reject_enabled: bool = True
    border_margin_px: float = 2.0
    border_reject_center_min_px: float = 105.0
    border_reject_head_conf: float = 0.88
    border_reject_body_conf: float = 0.88

    # Strict head/body relationship checks.
    require_head_near_body: bool = True
    body_expand_w_ratio: float = 0.34
    body_expand_h_ratio: float = 0.14
    head_upper_body_ratio_max: float = 0.66
    min_head_body_area_ratio: float = 0.0010
    max_head_body_area_ratio: float = 0.60
    min_head_body_width_ratio: float = 0.015
    max_head_body_width_ratio: float = 0.95
    min_head_body_height_ratio: float = 0.015
    max_head_body_height_ratio: float = 0.85
    require_body_extends_below_head: bool = True
    min_body_pixels_below_head: float = 3.0

    # Relaxed small/far relationship checks. These only apply when the head box is
    # small by area/dimension. They prevent the filter from killing real distant
    # enemies while still rejecting large map props with impossible geometry.
    small_body_expand_w_ratio: float = 0.58
    small_body_expand_h_ratio: float = 0.28
    small_head_upper_body_ratio_max: float = 0.78
    small_min_head_body_area_ratio: float = 0.00035
    small_max_head_body_area_ratio: float = 0.78
    small_min_head_body_width_ratio: float = 0.006
    small_max_head_body_width_ratio: float = 1.15
    small_min_head_body_height_ratio: float = 0.006
    small_max_head_body_height_ratio: float = 1.05
    small_min_body_pixels_below_head: float = 1.0

    # V17.8.12: paired small-head ghosts are the remaining failure mode.
    # A wall light can create both a fake head and a fake body.  Therefore a
    # small pair is no longer trusted merely because it geometrically matches;
    # when far from center or supported by a short body, it must be very strong.
    small_pair_far_center_px: float = 90.0
    small_pair_far_min_head_conf: float = 0.82
    small_pair_far_min_body_conf: float = 0.84
    small_pair_short_body_px: float = 72.0
    small_pair_short_min_head_conf: float = 0.84
    small_pair_short_min_body_conf: float = 0.84


@dataclass(frozen=True)
class FilterStats:
    input_boxes: int = 0
    output_boxes: int = 0
    heads_in: int = 0
    bodies_in: int = 0
    heads_kept: int = 0
    bodies_kept: int = 0
    heads_rejected: int = 0
    bodies_rejected: int = 0
    small_heads_kept: int = 0
    head_only_kept: int = 0


def _aspect(box: DetectionBox) -> float:
    return box.w / max(box.h, 1e-6)


def _center_dist(box: DetectionBox, center: tuple[float, float]) -> float:
    x, y = box.center
    return math.hypot(x - center[0], y - center[1])


def _touches_border(box: DetectionBox, roi_w: float, roi_h: float, margin: float) -> bool:
    return (
        box.x1 <= margin
        or box.y1 <= margin
        or box.x2 >= roi_w - margin
        or box.y2 >= roi_h - margin
    )


def _pair_touches_border(head: DetectionBox, body: DetectionBox, roi_w: float, roi_h: float, margin: float) -> bool:
    return _touches_border(head, roi_w, roi_h, margin) or _touches_border(body, roi_w, roi_h, margin)


def _expanded_contains(box: DetectionBox, x: float, y: float, expand_w: float, expand_h: float) -> bool:
    ew = box.w * expand_w
    eh = box.h * expand_h
    return (box.x1 - ew) <= x <= (box.x2 + ew) and (box.y1 - eh) <= y <= (box.y2 + eh)


def _is_small_head(head: DetectionBox, cfg: DetectionGeometryFilterConfig) -> bool:
    return (
        head.area <= float(cfg.small_head_area_px)
        or max(head.w, head.h) <= float(cfg.small_head_max_dim_px)
    )


def _basic_head_ok(head: DetectionBox, cfg: DetectionGeometryFilterConfig) -> bool:
    # Use a lower base floor for small/far heads. The later pairing / center rules
    # decide whether the candidate is trustworthy enough.
    min_conf = float(cfg.small_head_min_conf) if _is_small_head(head, cfg) else float(cfg.min_head_conf)
    if float(head.conf) < min_conf:
        return False
    if head.area < float(cfg.min_head_area_px) or head.area > float(cfg.max_head_area_px):
        return False
    if head.w < float(cfg.min_head_width_px) or head.h < float(cfg.min_head_height_px):
        return False
    if head.w > float(cfg.max_head_width_px) or head.h > float(cfg.max_head_height_px):
        return False
    ar = _aspect(head)
    if ar < float(cfg.min_head_aspect) or ar > float(cfg.max_head_aspect):
        return False
    return True


def _basic_body_ok(body: DetectionBox, cfg: DetectionGeometryFilterConfig, roi_area: float, *, small_context: bool = False) -> bool:
    # Keep weak bodies around; they may be needed only as spatial evidence for a
    # far/small head. Movement validation can still reject impossible pairs later.
    if float(body.conf) < min(float(cfg.min_body_conf), float(cfg.small_body_min_conf)):
        return False
    if body.area < float(cfg.min_body_area_px):
        return False
    min_w = float(cfg.small_min_body_width_px) if small_context else float(cfg.min_body_width_px)
    min_h = float(cfg.small_min_body_height_px) if small_context else float(cfg.min_body_height_px)
    if body.w < min_w or body.h < min_h:
        return False
    if roi_area > 0 and body.area > roi_area * float(cfg.max_body_area_ratio):
        return False
    ar = _aspect(body)
    max_ar = float(cfg.small_max_body_aspect) if small_context else float(cfg.max_body_aspect)
    if ar < float(cfg.min_body_aspect) or ar > max_ar:
        return False
    return True


def _pair_ok(
    head: DetectionBox,
    body: DetectionBox,
    cfg: DetectionGeometryFilterConfig,
    *,
    center: tuple[float, float] | None = None,
) -> bool:
    small = _is_small_head(head, cfg) and bool(cfg.small_relaxed_pair)
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
        min_body_conf = float(cfg.small_body_min_conf)
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
        min_body_conf = float(cfg.min_body_conf)

    if float(body.conf) < min_body_conf:
        return False

    # Anti-map micro-body check. The model sometimes emits a head/body pair on
    # small lights or wall geometry. Those pairs usually have a very short body
    # height or a squat/wide body aspect. Reject them before geometric pairing.
    min_bw = float(cfg.small_min_body_width_px) if small else float(cfg.min_body_width_px)
    min_bh = float(cfg.small_min_body_height_px) if small else float(cfg.min_body_height_px)
    max_bar = float(cfg.small_max_body_aspect) if small else float(cfg.max_body_aspect)
    if body.w < min_bw or body.h < min_bh:
        return False
    if _aspect(body) > max_bar:
        return False

    hx, hy = head.center

    # V17.8.12 fail-safe: small paired detections are allowed only if their
    # support body is sufficiently human-like, or the pair has very strong
    # head/body confidence. This specifically targets map lights/props that
    # form a fake head+body pair for many consecutive frames.
    if small:
        center_dist = _center_dist(head, center) if center is not None else 0.0
        if body.h < float(cfg.small_pair_short_body_px):
            if not (
                float(head.conf) >= float(cfg.small_pair_short_min_head_conf)
                and float(body.conf) >= float(cfg.small_pair_short_min_body_conf)
            ):
                return False
        if center is not None and center_dist > float(cfg.small_pair_far_center_px):
            if not (
                float(head.conf) >= float(cfg.small_pair_far_min_head_conf)
                and float(body.conf) >= float(cfg.small_pair_far_min_body_conf)
            ):
                return False

    if cfg.require_head_near_body and not _expanded_contains(body, hx, hy, expand_w, expand_h):
        return False

    rel_y = (hy - body.y1) / max(body.h, 1e-6)
    if rel_y > upper_ratio:
        return False

    if bool(cfg.require_body_extends_below_head):
        if (body.y2 - hy) < min_below:
            return False

    area_ratio = head.area / max(body.area, 1e-6)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False

    wr = head.w / max(body.w, 1e-6)
    hr = head.h / max(body.h, 1e-6)
    if wr < min_wr or wr > max_wr:
        return False
    if hr < min_hr or hr > max_hr:
        return False
    return True


def _match_body(
    head: DetectionBox,
    bodies: Sequence[DetectionBox],
    cfg: DetectionGeometryFilterConfig,
    *,
    center: tuple[float, float] | None = None,
) -> Optional[DetectionBox]:
    matches = [b for b in bodies if _pair_ok(head, b, cfg, center=center)]
    if not matches:
        return None
    hx, hy = head.center

    def score(body: DetectionBox) -> tuple[float, float, float]:
        bx, by = body.center
        dist = math.hypot(hx - bx, hy - by)
        return (float(body.conf), -dist, body.area)

    return max(matches, key=score)


def filter_detections_by_geometry(
    boxes: Iterable[DetectionBox],
    cfg: DetectionGeometryFilterConfig,
    *,
    center: tuple[float, float],
    frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> tuple[list[DetectionBox], FilterStats]:
    all_boxes = list(boxes)
    if not bool(cfg.enabled):
        return all_boxes, FilterStats(
            input_boxes=len(all_boxes),
            output_boxes=len(all_boxes),
            heads_in=sum(1 for b in all_boxes if b.cls_id == cfg.head_class_id),
            bodies_in=sum(1 for b in all_boxes if b.cls_id == cfg.body_class_id),
            heads_kept=sum(1 for b in all_boxes if b.cls_id == cfg.head_class_id),
            bodies_kept=sum(1 for b in all_boxes if b.cls_id == cfg.body_class_id),
        )

    if frame_shape is not None and len(frame_shape) >= 2:
        roi_h, roi_w = int(frame_shape[0]), int(frame_shape[1])
        roi_area = float(max(1, roi_h * roi_w))
    else:
        max_x = max((b.x2 for b in all_boxes), default=1.0)
        max_y = max((b.y2 for b in all_boxes), default=1.0)
        roi_w, roi_h = int(max(1.0, max_x)), int(max(1.0, max_y))
        roi_area = float(max(1.0, max_x * max_y))

    heads_in = [b for b in all_boxes if b.cls_id == int(cfg.head_class_id)]
    bodies_in = [b for b in all_boxes if b.cls_id == int(cfg.body_class_id)]
    others = [b for b in all_boxes if b.cls_id not in (int(cfg.head_class_id), int(cfg.body_class_id))]

    valid_heads = [h for h in heads_in if _basic_head_ok(h, cfg)]
    valid_bodies = [b for b in bodies_in if _basic_body_ok(b, cfg, roi_area, small_context=False)]

    accepted_heads: list[DetectionBox] = []
    accepted_body_ids: set[int] = set()
    small_heads_kept = 0
    head_only_kept = 0

    for head in valid_heads:
        small = _is_small_head(head, cfg)
        body = _match_body(head, valid_bodies, cfg, center=center)
        paired_min = float(cfg.small_paired_head_min_conf) if small else float(cfg.paired_head_min_conf)
        if body is not None and float(head.conf) >= paired_min:
            # Strong border clamp rejection: edge-stuck paired boxes are a major
            # source of map-model ghosts. Allow only if the pair is extremely
            # confident or close to the center.
            if bool(cfg.border_reject_enabled) and _pair_touches_border(head, body, float(roi_w), float(roi_h), float(cfg.border_margin_px)):
                far_from_center = _center_dist(head, center) >= float(cfg.border_reject_center_min_px)
                not_very_strong = (float(head.conf) < float(cfg.border_reject_head_conf) or float(body.conf) < float(cfg.border_reject_body_conf))
                if far_from_center and not_very_strong:
                    continue
            accepted_heads.append(head)
            accepted_body_ids.add(id(body))
            if small:
                small_heads_kept += 1
            continue

        if small:
            head_only_min = float(cfg.small_head_only_min_conf)
            center_max = float(cfg.small_head_only_center_max_px)
        else:
            head_only_min = float(cfg.head_only_min_conf)
            center_max = float(cfg.head_only_center_max_px)

        if float(head.conf) >= head_only_min and _center_dist(head, center) <= center_max:
            accepted_heads.append(head)
            head_only_kept += 1
            if small:
                small_heads_kept += 1

    if bool(cfg.keep_unpaired_bodies):
        accepted_bodies = list(valid_bodies)
    else:
        accepted_bodies = [b for b in valid_bodies if id(b) in accepted_body_ids]

    accepted = accepted_bodies + accepted_heads + others
    stats = FilterStats(
        input_boxes=len(all_boxes),
        output_boxes=len(accepted),
        heads_in=len(heads_in),
        bodies_in=len(bodies_in),
        heads_kept=len(accepted_heads),
        bodies_kept=len(accepted_bodies),
        heads_rejected=max(0, len(heads_in) - len(accepted_heads)),
        bodies_rejected=max(0, len(bodies_in) - len(accepted_bodies)),
        small_heads_kept=small_heads_kept,
        head_only_kept=head_only_kept,
    )
    return accepted, stats
