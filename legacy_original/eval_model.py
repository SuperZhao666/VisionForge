#!/usr/bin/env python3
"""
ONNX 二分类模型离线评估脚本：敌人 vs 非敌人。

新版特点：
1. 自动读取 ONNX 输入通道数：1 通道旧 mask 模型、3 通道 RGB 模型、4 通道 RGB+mask 模型都能评估。
2. 4 通道模型会同时输入原始 RGB ROI 与 mask ROI，用于验证“能否看见纹理/颜色/边缘”。
3. 不导入 main.py，不触发管理员权限、截图、键盘监听、串口连接。
"""

import argparse
import csv
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure(pkg: str, import_name: Optional[str] = None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", MIRROR, pkg])


for _pkg, _imp in [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pyyaml", "yaml"),
    ("onnxruntime", "onnxruntime"),
]:
    ensure(_pkg, _imp)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import yaml  # noqa: E402

from config import Config  # noqa: E402
from detector import build_color_mask as build_shared_color_mask, contour_extent_and_solidity  # noqa: E402
from model_input import adapt_batch_layout, infer_input_layout, make_model_input  # noqa: E402


@dataclass
class EvalSample:
    source_path: str
    label: int
    roi_index: int
    x: int
    y: int
    w: int
    h: int
    area: float
    white_ratio: float
    tensor: np.ndarray
    roi_mask: np.ndarray
    roi_bgr: np.ndarray


@dataclass
class PredRecord:
    source_path: str
    label: int
    prob: float
    roi_index: int
    x: int
    y: int
    w: int
    h: int
    area: float
    white_ratio: float


def load_yaml(path: str) -> Dict:
    return Config.from_yaml(path)


def cfg_value(cfg: Dict, key: str, default):
    value = cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)
    return default if value is None else value


def list_images(folder: Path, max_files: int = 0) -> List[Path]:
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_files and max_files > 0:
        files = files[-max_files:]
    return files


def build_runtime_mask(img_bgr: np.ndarray, cfg: Dict) -> np.ndarray:
    return build_shared_color_mask(img_bgr, cfg, source_color="BGR")


def contour_passes_runtime_filters(c, x: int, y: int, w: int, h: int, area: float, cfg: Dict, img_w: int, img_h: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    if w > int(cfg_value(cfg, "filter_max_width", 600)):
        return False
    if h > int(cfg_value(cfg, "filter_max_height", 600)):
        return False
    if h < int(cfg_value(cfg, "filter_min_height", 2)):
        return False
    if area < float(cfg_value(cfg, "min_contour_area", 8)):
        return False
    if area > float(cfg_value(cfg, "max_contour_area", 10000)):
        return False

    edge = int(cfg_value(cfg, "edge_margin", 2))
    if x < edge or y < edge or (x + w) > (img_w - edge) or (y + h) > (img_h - edge):
        return False

    perim = cv2.arcLength(c, True)
    circ = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
    if circ > float(cfg_value(cfg, "filter_circularity_max", 1.0)):
        return False

    wh = w * h
    if wh <= 0:
        return False
    extent, solidity = contour_extent_and_solidity(c, area, w, h)
    if extent < float(cfg_value(cfg, "filter_rectangularity_min", 0.0)):
        return False
    if extent > float(cfg_value(cfg, "filter_extent_max", 1.0)):
        return False
    if solidity > float(cfg_value(cfg, "filter_solidity_max", 1.0)):
        return False
    if solidity < float(cfg_value(cfg, "filter_solidity_min", 0.0)):
        return False

    ar = h / max(1, w)
    if ar < float(cfg_value(cfg, "filter_aspect_min", 0.1)):
        return False
    return True


def extract_samples_from_image(path: Path, label: int, cfg: Dict, img_size: int, input_channels: int, use_runtime_filters: bool) -> List[EvalSample]:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return []

    mask = build_runtime_mask(img_bgr, cfg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    samples: List[EvalSample] = []
    img_h, img_w = mask.shape[:2]

    for idx, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5:
            continue
        area = float(cv2.contourArea(c))
        if use_runtime_filters and not contour_passes_runtime_filters(c, x, y, w, h, area, cfg, img_w, img_h):
            continue

        roi_mask = mask[y:y + h, x:x + w]
        if roi_mask.size <= 0:
            continue
        white = int(np.count_nonzero(roi_mask))
        if white < 10:
            continue
        white_ratio = white / float(roi_mask.size)
        roi_bgr = img_bgr[y:y + h, x:x + w].copy()

        try:
            tensor = make_model_input(roi_bgr, roi_mask, img_size, channels=input_channels, source_color="BGR")
        except Exception:
            continue

        samples.append(EvalSample(
            source_path=str(path), label=label, roi_index=idx,
            x=int(x), y=int(y), w=int(w), h=int(h), area=area,
            white_ratio=float(white_ratio), tensor=tensor,
            roi_mask=roi_mask.copy(), roi_bgr=roi_bgr,
        ))
    return samples


def load_eval_samples(dataset_dir: str, cfg: Dict, img_size: int, input_channels: int, max_files_per_class: int, use_runtime_filters: bool) -> List[EvalSample]:
    root = Path(dataset_dir)
    fire_files = list_images(root / "fire", max_files_per_class)
    no_fire_files = list_images(root / "no_fire", max_files_per_class)

    print(f"[数据] 正样本图片: {len(fire_files)} → {root / 'fire'}")
    print(f"[数据] 负样本图片: {len(no_fire_files)} → {root / 'no_fire'}")

    samples: List[EvalSample] = []
    for p in fire_files:
        samples.extend(extract_samples_from_image(p, 1, cfg, img_size, input_channels, use_runtime_filters))
    for p in no_fire_files:
        samples.extend(extract_samples_from_image(p, 0, cfg, img_size, input_channels, use_runtime_filters))
    return samples


def make_session(model_path: str) -> ort.InferenceSession:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = set(ort.get_available_providers())
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    print(f"[模型] providers={providers}")
    return ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)


def batch_predict(sess: ort.InferenceSession, samples: List[EvalSample], batch_size: int, channels_first: bool) -> List[PredRecord]:
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    records: List[PredRecord] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        x = np.stack([s.tensor for s in chunk], axis=0).astype(np.float32)
        x = adapt_batch_layout(x, channels_first)
        y = sess.run([output_name], {input_name: x})[0]
        probs = np.asarray(y, dtype=np.float32).reshape(-1)

        for s, p in zip(chunk, probs):
            records.append(PredRecord(
                source_path=s.source_path, label=int(s.label), prob=float(np.clip(p, 0.0, 1.0)),
                roi_index=s.roi_index, x=s.x, y=s.y, w=s.w, h=s.h,
                area=s.area, white_ratio=s.white_ratio,
            ))
    return records


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def metrics_at(records: List[PredRecord], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for r in records:
        pred = 1 if r.prob >= threshold else 0
        if r.label == 1 and pred == 1:
            tp += 1
        elif r.label == 0 and pred == 1:
            fp += 1
        elif r.label == 0 and pred == 0:
            tn += 1
        elif r.label == 1 and pred == 0:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
    }


def write_predictions_csv(path: Path, records: List[PredRecord]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_path", "label", "prob", "roi_index", "x", "y", "w", "h", "area", "white_ratio"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)


def write_metrics_csv(path: Path, rows: List[Dict[str, float]]):
    fields = ["threshold", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "specificity", "fpr", "fnr", "f1"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_misclassified(out_dir: Path, samples: List[EvalSample], records: List[PredRecord], threshold: float, limit: int):
    if limit <= 0:
        return
    fp_dir = out_dir / "misclassified" / "false_positive_non_enemy_as_enemy"
    fn_dir = out_dir / "misclassified" / "false_negative_enemy_as_non_enemy"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    by_key = {(s.source_path, s.roi_index, s.x, s.y, s.w, s.h): s for s in samples}
    false_pos = [r for r in records if r.label == 0 and r.prob >= threshold]
    false_neg = [r for r in records if r.label == 1 and r.prob < threshold]
    false_pos.sort(key=lambda r: r.prob, reverse=True)
    false_neg.sort(key=lambda r: r.prob)

    def save_side_by_side(r: PredRecord, folder: Path, prefix: str, rank: int):
        key = (r.source_path, r.roi_index, r.x, r.y, r.w, r.h)
        s = by_key.get(key)
        if s is None:
            return
        roi = s.roi_bgr
        mask = cv2.cvtColor(s.roi_mask, cv2.COLOR_GRAY2BGR)
        h = max(roi.shape[0], mask.shape[0], 40)

        def pad(img):
            top = (h - img.shape[0]) // 2
            bottom = h - img.shape[0] - top
            return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        side = np.hstack([pad(roi), pad(mask)])
        name = f"{prefix}_{rank:03d}_p{r.prob:.4f}_w{r.w}_h{r.h}_area{int(r.area)}.png"
        cv2.imwrite(str(folder / name), side)

    for i, r in enumerate(false_pos[:limit], 1):
        save_side_by_side(r, fp_dir, "FP", i)
    for i, r in enumerate(false_neg[:limit], 1):
        save_side_by_side(r, fn_dir, "FN", i)


def write_summary_txt(path: Path, rows: List[Dict[str, float]], cfg: Dict, records: List[PredRecord], input_channels: int):
    filter_thr = float(cfg_value(cfg, "model_filter_threshold", 0.5))
    fire_thr = float(cfg_value(cfg, "fire_threshold", 0.7))
    row_filter = min(rows, key=lambda r: abs(r["threshold"] - filter_thr)) if rows else None
    row_fire = min(rows, key=lambda r: abs(r["threshold"] - fire_thr)) if rows else None

    n_pos = sum(1 for r in records if r.label == 1)
    n_neg = sum(1 for r in records if r.label == 0)
    mode = {1: "mask-only", 3: "RGB", 4: "RGB+mask"}.get(input_channels, str(input_channels))

    with open(path, "w", encoding="utf-8") as f:
        f.write("模型离线评估摘要\n")
        f.write("=" * 40 + "\n")
        f.write(f"模型输入模式: {input_channels}ch {mode}\n")
        f.write(f"ROI总数: {len(records)}\n")
        f.write(f"敌人ROI: {n_pos}\n")
        f.write(f"非敌人ROI: {n_neg}\n\n")
        if row_filter:
            f.write(f"移动/锁定阈值 model_filter_threshold={filter_thr}\n")
            f.write(f"  TP={row_filter['tp']} FP={row_filter['fp']} TN={row_filter['tn']} FN={row_filter['fn']}\n")
            f.write(f"  非敌人误判率FPR={row_filter['fpr']:.4f}  敌人召回率Recall={row_filter['recall']:.4f}  精确率Precision={row_filter['precision']:.4f}\n\n")
        if row_fire:
            f.write(f"开火阈值 fire_threshold={fire_thr}\n")
            f.write(f"  TP={row_fire['tp']} FP={row_fire['fp']} TN={row_fire['tn']} FN={row_fire['fn']}\n")
            f.write(f"  非敌人误判率FPR={row_fire['fpr']:.4f}  敌人召回率Recall={row_fire['recall']:.4f}  精确率Precision={row_fire['precision']:.4f}\n\n")

        f.write("判定参考：\n")
        f.write("1. 如果 model_filter_threshold 下 FP 很多，说明非敌人仍会被锁定，需要继续补充 dataset/no_fire。\n")
        f.write("2. 如果 fire_threshold 下 FP 很多，说明误检仍可能进入开火级，必须提高阈值或补充 hard negative。\n")
        f.write("3. 如果 FN 很多，说明敌人被漏掉，正样本不足或样本分布不一致。\n")
        f.write("4. 4ch RGB+mask 只能提高信息上限，不能替代真实 hard negative 闭环。\n")


def parse_thresholds(text: str, cfg: Dict) -> List[float]:
    base = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    base.append(float(cfg_value(cfg, "model_filter_threshold", 0.5)))
    base.append(float(cfg_value(cfg, "fire_threshold", 0.7)))
    if text.strip():
        for part in text.split(","):
            part = part.strip()
            if part:
                base.append(float(part))
    vals = sorted({round(min(max(v, 0.0), 1.0), 4) for v in base})
    return vals


def print_table(rows: List[Dict[str, float]]):
    print("\n[阈值扫描]")
    print("thr    TP   FP   TN   FN   acc    prec   recall fpr    f1")
    print("-" * 72)
    for r in rows:
        print(f"{r['threshold']:<5.2f} {r['tp']:>4} {r['fp']:>4} {r['tn']:>4} {r['fn']:>4} "
              f"{r['accuracy']:<6.3f} {r['precision']:<6.3f} {r['recall']:<6.3f} {r['fpr']:<6.3f} {r['f1']:<6.3f}")


def main():
    parser = argparse.ArgumentParser(description="评估 fire_model.onnx 是否能区分敌人与非敌人。")
    parser.add_argument("--model", default="fire_model.onnx", help="ONNX模型路径")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--dataset", default="dataset", help="数据集根目录，内部应有 fire / no_fire")
    parser.add_argument("--out", default="eval_report", help="评估报告输出目录")
    parser.add_argument("--batch-size", type=int, default=256, help="ONNX批量推理batch大小")
    parser.add_argument("--max-files-per-class", type=int, default=0, help="每类最多读取多少张图片；0表示不限制")
    parser.add_argument("--thresholds", default="", help="额外阈值，用逗号分隔，如 0.45,0.55,0.65")
    parser.add_argument("--save-misclassified", type=int, default=50, help="每类最多保存多少个误判ROI")
    parser.add_argument("--use-runtime-filters", action="store_true", help="启用和主循环接近的几何过滤；默认关闭")
    parser.add_argument("--clean", action="store_true", help="运行前清空输出目录")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    img_size = max(8, int(cfg_value(cfg, "img_size", 48)))
    thresholds = parse_thresholds(args.thresholds, cfg)
    out_dir = Path(args.out)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ONNX 模型离线评估：敌人 vs 非敌人")
    print("=" * 70)
    print(f"[配置] config={args.config} img_size={img_size}")
    print(f"[路径] model={args.model}")
    print(f"[路径] dataset={args.dataset}")
    print(f"[路径] out={out_dir}")
    print(f"[模式] use_runtime_filters={args.use_runtime_filters}")

    sess = make_session(args.model)
    input_meta = sess.get_inputs()[0]
    input_channels, channels_first = infer_input_layout(input_meta.shape)
    mode = {1: "mask-only", 3: "RGB", 4: "RGB+mask"}.get(input_channels, str(input_channels))
    print(f"[模型] input_shape={input_meta.shape} input_channels={input_channels} mode={mode} layout={'NCHW' if channels_first else 'NHWC'}")

    samples = load_eval_samples(args.dataset, cfg, img_size, input_channels, args.max_files_per_class, args.use_runtime_filters)
    n_pos = sum(1 for s in samples if s.label == 1)
    n_neg = sum(1 for s in samples if s.label == 0)
    print(f"[ROI] 总数={len(samples)} 敌人={n_pos} 非敌人={n_neg}")

    if len(samples) == 0:
        raise RuntimeError("没有提取到任何ROI。请检查 dataset/fire 与 dataset/no_fire 是否有图片，以及颜色阈值是否正确。")
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("正负样本ROI至少各需要1个，否则无法判断区分能力。")

    records = batch_predict(sess, samples, max(1, int(args.batch_size)), channels_first)
    rows = [metrics_at(records, t) for t in thresholds]

    write_predictions_csv(out_dir / "predictions.csv", records)
    write_metrics_csv(out_dir / "metrics.csv", rows)
    save_misclassified(out_dir, samples, records, float(cfg_value(cfg, "model_filter_threshold", 0.5)), args.save_misclassified)
    write_summary_txt(out_dir / "summary.txt", rows, cfg, records, input_channels)

    print_table(rows)
    print("\n[输出]")
    print(f"  {out_dir / 'summary.txt'}")
    print(f"  {out_dir / 'metrics.csv'}")
    print(f"  {out_dir / 'predictions.csv'}")
    print(f"  {out_dir / 'misclassified'}")

    filter_thr = float(cfg_value(cfg, "model_filter_threshold", 0.5))
    fire_thr = float(cfg_value(cfg, "fire_threshold", 0.7))
    filter_row = min(rows, key=lambda r: abs(r["threshold"] - filter_thr))
    fire_row = min(rows, key=lambda r: abs(r["threshold"] - fire_thr))

    print("\n[关键结论]")
    print(f"  锁定/移动阈值 {filter_thr:.2f}: FPR={filter_row['fpr']:.3f}, Recall={filter_row['recall']:.3f}, Precision={filter_row['precision']:.3f}")
    print(f"  开火阈值     {fire_thr:.2f}: FPR={fire_row['fpr']:.3f}, Recall={fire_row['recall']:.3f}, Precision={fire_row['precision']:.3f}")
    if filter_row["fp"] > 0:
        print("  结论: 仍有非敌人会被放行到锁定/移动级。优先查看 false_positive_non_enemy_as_enemy 文件夹，并把这些样本继续作为 no_fire 训练。")
    else:
        print("  结论: 当前数据集上，非敌人未被放行到锁定/移动级。仍需真实场景继续验证。")
    if fire_row["fp"] > 0:
        print("  警告: 仍有非敌人达到开火级阈值。此时不建议直接信任该模型。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[错误] {e}")
        sys.exit(1)
