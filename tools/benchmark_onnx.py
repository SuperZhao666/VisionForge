from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.onnx_yolo_detector import OnnxYoloDetector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--loops", type=int, default=200)
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    det = OnnxYoloDetector(args.model, imgsz=args.imgsz)
    for _ in range(args.warmup):
        det.predict(img)
    times = []
    for _ in range(args.loops):
        t0 = time.perf_counter()
        det.predict(img)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    print({
        "loops": args.loops,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
