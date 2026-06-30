#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import time
from collections import deque

import cv2
import mss
import numpy as np

from config import Config
from detector import detect_targets, format_detect_stats
from fire_classifier import FireClassifier
from tracker import TargetTracker


def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def avg(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="benchmark mode: no debug window, no HID control, no training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--no-inference", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.run_mode = "benchmark"
    cfg.show_debug = False
    cfg.debug_show_mask = False
    cfg.auto_train_enabled = False
    cfg.control_enabled = False
    cfg.auto_save_hard_negatives = False

    cx, cy = cfg.roi_width // 2, cfg.roi_height // 2
    mk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(1, cfg.morph_kernel_width), max(1, cfg.morph_kernel_height)),
    )
    motion_kernel = None
    if cfg.motion_enabled:
        ksize = max(1, int(cfg.motion_dilate_kernel))
        motion_kernel = np.ones((ksize, ksize), np.uint8)

    color_lower_np = np.array(cfg.color_lower, dtype=np.uint8)
    color_upper_np = np.array(cfg.color_upper, dtype=np.uint8)
    tracker = TargetTracker(cfg)
    classifier = None
    if cfg.model_inference_in_main and not args.no_inference:
        try:
            classifier = FireClassifier(
                cfg.model_path,
                cfg.img_size,
                cfg.fire_threshold,
                cfg.model_filter_threshold,
                cfg.model_filter_consecutive,
                cfg.model_filter_cache_ttl,
                color_lower_np=color_lower_np,
                color_upper_np=color_upper_np,
            )
        except Exception as e:
            print(f"[WARN] model disabled: {e}")

    sw, sh = get_screen_size()
    left, top = (sw - cfg.roi_width) // 2, (sh - cfg.roi_height) // 2
    monitor = {"top": top, "left": left, "width": cfg.roi_width, "height": cfg.roi_height}

    totals = deque(maxlen=max(1, args.frames))
    preprocess = deque(maxlen=max(1, args.frames))
    contours = deque(maxlen=max(1, args.frames))
    filters = deque(maxlen=max(1, args.frames))
    heads = deque(maxlen=max(1, args.frames))
    tracks = deque(maxlen=max(1, args.frames))
    inferences = deque(maxlen=max(1, args.frames))
    ages = deque(maxlen=max(1, args.frames))
    prev_gray = None
    last_t = time.perf_counter()

    mss_factory = getattr(mss, "MSS", mss.mss)
    with mss_factory() as sct:
        for frame_idx in range(1, max(1, args.frames) + 1):
            loop_start = time.perf_counter()
            raw = np.array(sct.grab(monitor))
            captured_at = time.perf_counter()
            img = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

            t_gray = time.perf_counter()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_ms = (time.perf_counter() - t_gray) * 1000
            capture_age_ms = (time.perf_counter() - captured_at) * 1000

            cands, mask, stats = detect_targets(
                img,
                gray,
                cfg,
                cx,
                cy,
                mk,
                color_lower_np,
                color_upper_np,
                prev_gray,
                motion_kernel,
                source_color="RGB",
            )
            prev_gray = gray
            stats.capture_age_ms = capture_age_ms
            stats.preprocess_ms += gray_ms

            dt = min(loop_start - last_t, 0.1)
            last_t = loop_start
            t_track = time.perf_counter()
            _, _, _, _, _, _, cf, _, best = tracker.update(cands, cx, cy, dt)
            stats.track_ms = (time.perf_counter() - t_track) * 1000
            if getattr(tracker, "last_reject_reason", None) == "static":
                stats.reject_by_static += 1

            if classifier is not None and best is not None and cf:
                t_inf = time.perf_counter()
                classifier.predict_proba(img, mask, best, use_cache=True)
                stats.inference_ms = (time.perf_counter() - t_inf) * 1000

            stats.total_loop_ms = (time.perf_counter() - loop_start) * 1000
            totals.append(stats.total_loop_ms)
            preprocess.append(stats.preprocess_ms)
            contours.append(stats.contour_ms)
            filters.append(stats.filter_ms)
            heads.append(stats.head_estimate_ms)
            tracks.append(stats.track_ms)
            inferences.append(stats.inference_ms)
            ages.append(stats.capture_age_ms)

            print(
                f"[BENCH] frame={frame_idx} total_loop_ms={stats.total_loop_ms:.2f} "
                f"capture_age_ms={stats.capture_age_ms:.2f} preprocess_ms={stats.preprocess_ms:.2f} "
                f"contour_ms={stats.contour_ms:.2f} filter_ms={stats.filter_ms:.2f} "
                f"head_estimate_ms={stats.head_estimate_ms:.2f} track_ms={stats.track_ms:.2f} "
                f"inference_ms={stats.inference_ms:.2f} candidate_count={stats.candidate_count} "
                f"raw_contour_count={stats.raw_contour_count} {format_detect_stats(stats)}"
            )

    print("\n[BENCH SUMMARY]")
    print(f"frames={len(totals)} total_loop_ms_avg={avg(totals):.2f}")
    print(
        f"capture_age_ms_avg={avg(ages):.2f} preprocess_ms_avg={avg(preprocess):.2f} "
        f"contour_ms_avg={avg(contours):.2f} filter_ms_avg={avg(filters):.2f} "
        f"head_estimate_ms_avg={avg(heads):.2f} track_ms_avg={avg(tracks):.2f} "
        f"inference_ms_avg={avg(inferences):.2f}"
    )


if __name__ == "__main__":
    main()
