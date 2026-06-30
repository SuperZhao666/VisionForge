from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import yaml

from src.onnx_yolo_detector import OnnxYoloDetector
from src.target_selector import TargetSelector
from src.tracker import EmaPointTracker
from main import draw_result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--out", default="outputs/test_image_result.jpg")
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    detector = OnnxYoloDetector(args.model, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    boxes = detector.predict(img)
    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
    selector = TargetSelector(prefer_center=center)
    target = EmaPointTracker().update(selector.select(boxes))
    out = draw_result(img, boxes, target, center)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print({"detections": len(boxes), "target": target.to_dict(), "out": str(out_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
