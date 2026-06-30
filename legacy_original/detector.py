from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from tracker import Target


@dataclass
class DetectStats:
    capture_age_ms: float = 0.0
    preprocess_ms: float = 0.0
    contour_ms: float = 0.0
    filter_ms: float = 0.0
    head_estimate_ms: float = 0.0
    track_ms: float = 0.0
    inference_ms: float = 0.0
    debug_draw_ms: float = 0.0
    total_loop_ms: float = 0.0
    candidate_count: int = 0
    raw_contour_count: int = 0
    reject_by_size: int = 0
    reject_by_area_low: int = 0
    reject_by_area_high: int = 0
    reject_by_edge: int = 0
    reject_by_circularity: int = 0
    reject_by_extent: int = 0
    reject_by_solidity: int = 0
    reject_by_aspect: int = 0
    reject_by_static: int = 0
    reject_by_motion: int = 0
    accepted_candidates: int = 0
    head_limited: int = 0
    rescued_close: int = 0

    def to_dict(self):
        return asdict(self)


def new_detect_stats() -> DetectStats:
    return DetectStats()


def format_detect_stats(stats: Optional[DetectStats]) -> str:
    if not stats:
        return "raw=0 final=0"
    return (
        f"raw={stats.raw_contour_count} final={stats.candidate_count} "
        f"size={stats.reject_by_size} edge={stats.reject_by_edge} "
        f"area<{stats.reject_by_area_low}/>{stats.reject_by_area_high} "
        f"circ={stats.reject_by_circularity} extent={stats.reject_by_extent} "
        f"solid={stats.reject_by_solidity} aspect={stats.reject_by_aspect} "
        f"motion={stats.reject_by_motion} static={stats.reject_by_static} "
        f"head_limit={stats.head_limited} rescue={stats.rescued_close}"
    )


def build_color_mask(img: np.ndarray, cfg, source_color: str = "RGB") -> np.ndarray:
    order = (source_color or "RGB").upper()
    if order == "RGB":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif order == "BGR":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError(f"source_color must be RGB or BGR, got {source_color!r}")

    lower = np.array(cfg.color_lower, dtype=np.uint8)
    upper = np.array(cfg.color_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if cfg.morph_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(1, int(cfg.morph_kernel_width)), max(1, int(cfg.morph_kernel_height))),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.morph_iterations))
    return mask


def _clamp_float(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _split_row_bands(valid_rows, gap_tolerance: int):
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
        valid = row_count > 0
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size == 0:
        return fallback_hx, fallback_hy, debug

    bands = _split_row_bands(valid_rows, int(cfg.head_row_gap_tolerance))
    if not bands:
        return fallback_hx, fallback_hy, debug

    best_band = bands[0]
    best_score = -1.0
    for b0, b1 in bands:
        rr = np.arange(b0, b1 + 1)
        counts = row_count[rr]
        if counts.size == 0:
            continue
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
    weights *= np.exp(-float(cfg.head_vertical_decay) * (rr - b0))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raw_hx = float(np.median(centers))
        raw_hy = float(b0 + (b1 - b0) * float(cfg.head_y_band_position))
    else:
        raw_hx = float(np.sum(centers * weights) / weight_sum)
        band_y = b0 + (b1 - b0) * float(cfg.head_y_band_position)
        weighted_y = float(np.sum(rr * weights) / weight_sum)
        raw_hy = 0.65 * band_y + 0.35 * weighted_y

    offset_y = float(cfg.head_offset_y_ratio) * float(h)
    if is_wide_pose:
        offset_y *= float(cfg.head_wide_offset_scale)
    lo = min(float(cfg.head_offset_y_min), float(cfg.head_offset_y_max))
    hi = max(float(cfg.head_offset_y_min), float(cfg.head_offset_y_max))
    offset_y = _clamp_float(offset_y, lo, hi)

    final_hx = _clamp_float(raw_hx, 0, w - 1)
    final_hy = _clamp_float(raw_hy + offset_y, 0, h - 1)

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


def contour_extent_and_solidity(contour, area: float, w: int, h: int):
    rect_area = float(max(1, w * h))
    extent = float(area) / rect_area
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    return extent, solidity


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


def detect_targets(
    img,
    gray,
    cfg,
    cx,
    cy,
    morph_kernel,
    color_lower_np,
    color_upper_np,
    prev_gray=None,
    motion_kernel=None,
    source_color: str = "RGB",
):
    stats = DetectStats()
    t_pre = time.perf_counter()
    mask = build_color_mask(img, cfg, source_color=source_color)
    stats.preprocess_ms += (time.perf_counter() - t_pre) * 1000

    if cfg.motion_enabled and prev_gray is not None:
        if gray is None:
            t_gray = time.perf_counter()
            code = cv2.COLOR_RGB2GRAY if (source_color or "RGB").upper() == "RGB" else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(img, code)
            stats.preprocess_ms += (time.perf_counter() - t_gray) * 1000
        before_motion = mask.copy()
        diff = cv2.absdiff(gray, prev_gray)
        _, motion = cv2.threshold(diff, cfg.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        if motion_kernel is None:
            ksize = max(1, cfg.motion_dilate_kernel)
            motion_kernel = np.ones((ksize, ksize), np.uint8)
        motion = cv2.dilate(motion, motion_kernel, iterations=1)
        mask = cv2.bitwise_and(mask, motion)
        t_motion_contours = time.perf_counter()
        cnt_before, _ = cv2.findContours(before_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_after, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stats.contour_ms += (time.perf_counter() - t_motion_contours) * 1000
        stats.reject_by_motion = max(0, len(cnt_before) - len(cnt_after))

    t_contour = time.perf_counter()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats.contour_ms += (time.perf_counter() - t_contour) * 1000
    stats.raw_contour_count = len(cnts)

    res = []
    prelim = []
    roi_w = cfg.roi_width
    roi_h = cfg.roi_height
    edge = cfg.edge_margin

    t_filter = time.perf_counter()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            stats.reject_by_size += 1
            continue
        area = cv2.contourArea(c)
        rescue_close = _is_close_target_rescue(x, y, w, h, area, cfg, cx, cy, roi_w, roi_h)

        size_hit = w > cfg.filter_max_width or h > cfg.filter_max_height or h < cfg.filter_min_height
        if size_hit:
            stats.reject_by_size += 1
            if not rescue_close:
                continue

        edge_hit = x < edge or y < edge or (x + w) > (roi_w - edge) or (y + h) > (roi_h - edge)
        if edge_hit:
            stats.reject_by_edge += 1
            if not rescue_close:
                continue

        if area < cfg.min_contour_area:
            stats.reject_by_area_low += 1
            continue
        if area > cfg.max_contour_area:
            stats.reject_by_area_high += 1
            if not rescue_close:
                continue

        perim = cv2.arcLength(c, True)
        circ = (4 * math.pi * area / (perim * perim)) if perim > 0 else 0
        if circ > cfg.filter_circularity_max:
            stats.reject_by_circularity += 1
            continue

        extent, solidity = contour_extent_and_solidity(c, area, w, h)
        if extent < cfg.filter_rectangularity_min or extent > cfg.filter_extent_max:
            stats.reject_by_extent += 1
            continue
        if solidity > cfg.filter_solidity_max or solidity < cfg.filter_solidity_min:
            stats.reject_by_solidity += 1
            continue

        ar = h / w
        if ar < cfg.filter_aspect_min:
            stats.reject_by_aspect += 1
            continue

        if rescue_close and (size_hit or edge_hit or area > cfg.max_contour_area):
            stats.rescued_close += 1

        approx_head_x = x + w * 0.5
        approx_head_y = y + h * 0.25
        approx_dist = math.hypot(approx_head_x - cx, approx_head_y - cy)
        prelim.append((approx_dist, -float(area), x, y, w, h, float(area), float(ar), float(circ)))
    stats.filter_ms += (time.perf_counter() - t_filter) * 1000

    if len(prelim) > cfg.max_head_estimation_candidates:
        prelim.sort(key=lambda item: (item[0], item[1]))
        stats.head_limited = len(prelim) - cfg.max_head_estimation_candidates
        prelim = prelim[:cfg.max_head_estimation_candidates]

    t_head = time.perf_counter()
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
                t = (dist - cfg.dynamic_offset_near_dist) / (
                    cfg.dynamic_offset_far_dist - cfg.dynamic_offset_near_dist
                )
                oy_dyn = cfg.dynamic_offset_near_y + t * (cfg.dynamic_offset_far_y - cfg.dynamic_offset_near_y)
            fy = fy + oy_dyn
            dist = math.hypot(fx - cx, fy - cy)

        target = Target(float(fx), float(fy), float(area), dist, int(w), int(h), float(ar), float(circ), int(x), int(y))
        target.raw_head_x = float(raw_fx)
        target.raw_head_y = float(raw_fy)
        target.head_roi_x = int(head_dbg.get("roi_x", x))
        target.head_roi_y = int(head_dbg.get("roi_y", y))
        target.head_roi_w = int(head_dbg.get("roi_w", w))
        target.head_roi_h = int(head_dbg.get("roi_h", max(1, h)))
        target.head_quality = float(head_dbg.get("quality", 0.0))
        target.head_pose_hint = str(head_dbg.get("pose", "upright"))
        res.append(target)
    stats.head_estimate_ms += (time.perf_counter() - t_head) * 1000

    stats.candidate_count = len(res)
    stats.accepted_candidates = len(res)
    return res, mask, stats
