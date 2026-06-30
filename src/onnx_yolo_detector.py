from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Critical: configure Windows DLL search path before importing onnxruntime.
# Otherwise packaged EXE can see CUDAExecutionProvider but fail LoadLibrary(error 126)
# and silently fall back to CPU or freeze during session creation.
try:
    from .app_paths import configure_dll_search_path
    configure_dll_search_path()
except Exception:
    pass

import cv2
import numpy as np
import onnxruntime as ort

# ORT 1.21+ 支持 preload_dlls；旧版本没有这个函数。这里做兼容尝试。
try:
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except TypeError:
            ort.preload_dlls()
except Exception:
    pass

from .log_utils import log
from .types import DetectionBox


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    input_w: int
    input_h: int
    orig_w: int
    orig_h: int


class OnnxYoloDetector:
    """Ultralytics YOLO detect ONNX 推理器。

    支持两类输出：
    1. 原始 YOLO 输出: [cx, cy, w, h, cls0, cls1] 或 [cx,cy,w,h,obj,cls0,cls1]
    2. 带 NMS 输出: [x1, y1, x2, y2, conf, cls]

    默认类别：0=body，1=head。
    """

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 320,
        conf: float = 0.25,
        iou: float = 0.70,
        class_names: Dict[int, str] | None = None,
        providers: Sequence[str] | None = None,
        max_candidates: int = 300,
        require_gpu: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {self.model_path}")
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_candidates = int(max_candidates or 0)
        self.class_names = class_names or {0: "body", 1: "head"}
        available = ort.get_available_providers()
        wanted = list(providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        actual = [p for p in wanted if p in available]
        if not actual:
            actual = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        # Suppress verbose ORT warnings that can print ANSI escape sequences in Windows terminals.
        # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal.
        so.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=actual)
        except Exception:
            # TensorRT is an optional acceleration layer. If TensorRT provider initialization fails
            # but CUDA is available, retry CUDA directly instead of killing the whole application.
            if "TensorrtExecutionProvider" in actual and "CUDAExecutionProvider" in available:
                actual = [p for p in actual if p != "TensorrtExecutionProvider"]
                if "CUDAExecutionProvider" not in actual:
                    actual.insert(0, "CUDAExecutionProvider")
                self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=actual)
            else:
                raise
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        active = self.session.get_providers()
        log(f"ONNX 模型已加载: {self.model_path}", "SUCCESS")
        log(f"ONNX available providers: {available}", "INFO")
        log(f"ONNX requested providers: {actual}", "INFO")
        log(f"ONNX active providers: {active}", "INFO")
        gpu_active = ("TensorrtExecutionProvider" in active) or ("CUDAExecutionProvider" in active)
        if require_gpu and not gpu_active:
            msg = (
                "GPU 推理未启用：ONNX Runtime 当前回退到 CPU。"
                "请打开 VisionForge 的环境助手修复 CUDA/cuDNN/ONNX Runtime GPU Provider，"
                "或由作者重新执行保护型构建脚本收集运行库。"
            )
            log(msg, "ERROR")
            raise RuntimeError(msg)
        if ("CUDAExecutionProvider" in wanted or "TensorrtExecutionProvider" in wanted) and not gpu_active:
            log("当前没有启用 GPU Provider，程序会退回 CPU。", "WARN")
        log(f"input={self.input_name}, outputs={self.output_names}, imgsz={self.imgsz}", "INFO")

    def predict(self, image_bgr: np.ndarray) -> List[DetectionBox]:
        boxes, _ = self.predict_with_profile(image_bgr)
        return boxes

    def predict_with_profile(self, image_bgr: np.ndarray) -> tuple[List[DetectionBox], Dict[str, float]]:
        t0 = time.perf_counter()
        inp, info = self._preprocess(image_bgr)
        t1 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        t2 = time.perf_counter()
        boxes = self._postprocess(outputs, info)
        t3 = time.perf_counter()
        return boxes, {
            "pre_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
        }

    def _preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空")
        orig_h, orig_w = image_bgr.shape[:2]
        new_h = new_w = self.imgsz
        scale = min(new_w / orig_w, new_h / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))
        pad_x = (new_w - resized_w) / 2.0
        pad_y = (new_h - resized_h) / 2.0

        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        x0 = int(round(pad_x - 0.1))
        y0 = int(round(pad_y - 0.1))
        canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x), LetterboxInfo(scale, pad_x, pad_y, new_w, new_h, orig_w, orig_h)

    def _postprocess(self, outputs: Sequence[np.ndarray], info: LetterboxInfo) -> List[DetectionBox]:
        if not outputs:
            return []
        arr = outputs[0]
        arr = np.asarray(arr)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1])

        # Ultralytics raw export often returns [channels, anchors], e.g. [6, 2100].
        if arr.shape[0] <= 32 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        arr = arr.astype(np.float32, copy=False)
        if arr.shape[1] < 6:
            return []

        boxes_xyxy, scores, cls_ids = self._decode_rows(arr)
        if boxes_xyxy.size == 0:
            return []

        keep = scores >= self.conf
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        if boxes_xyxy.size == 0:
            return []

        boxes_xyxy = self._unletterbox_xyxy(boxes_xyxy, info)
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, info.orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, info.orig_h - 1)

        # Low confidence thresholds are useful for far targets, but they can leave
        # many weak candidates for Python NMS. Keep the strongest candidates first.
        if self.max_candidates > 0 and scores.shape[0] > self.max_candidates:
            top = np.argpartition(scores, -self.max_candidates)[-self.max_candidates:]
            boxes_xyxy = boxes_xyxy[top]
            scores = scores[top]
            cls_ids = cls_ids[top]

        keep_idx = self._nms(boxes_xyxy, scores, cls_ids, self.iou)
        out: List[DetectionBox] = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            cls = int(cls_ids[i])
            out.append(DetectionBox(cls, self.class_names.get(cls, str(cls)), float(scores[i]), x1, y1, x2, y2))
        return out

    def _decode_rows(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_cols = arr.shape[1]
        nc = len(self.class_names)

        # 带 NMS 的 ONNX: [x1, y1, x2, y2, conf, cls]
        if n_cols == 6:
            xyxy_like = np.mean((arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1]))
            cls_like = np.mean(np.isclose(arr[:, 5], np.round(arr[:, 5])))
            if xyxy_like > 0.70 and cls_like > 0.80:
                boxes = arr[:, :4]
                scores = arr[:, 4]
                cls_ids = np.round(arr[:, 5]).astype(np.int32)
                return boxes, scores, cls_ids
            # raw two-class output: [cx, cy, w, h, cls0, cls1]
            boxes = self._xywh_to_xyxy(arr[:, :4])
            cls_scores = arr[:, 4:4 + nc]
            cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            return boxes, scores, cls_ids

        # [cx, cy, w, h, obj, cls0, cls1, ...]
        if n_cols == 5 + nc:
            boxes = self._xywh_to_xyxy(arr[:, :4])
            obj = arr[:, 4]
            cls_scores = arr[:, 5:5 + nc]
            cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = obj * cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            return boxes, scores, cls_ids

        # [cx, cy, w, h, cls_scores...]
        if n_cols >= 4 + nc:
            boxes = self._xywh_to_xyxy(arr[:, :4])
            cls_scores = arr[:, 4:]
            cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            return boxes, scores, cls_ids

        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        out = np.empty_like(xywh[:, :4])
        out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
        out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
        out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
        out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
        return out

    @staticmethod
    def _unletterbox_xyxy(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
        out = boxes.copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_y) / info.scale
        return out

    @staticmethod
    def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area1 = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
        area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        return inter / np.maximum(area1 + area2 - inter, 1e-6)

    @classmethod
    def _nms(cls, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_thr: float) -> List[int]:
        keep: List[int] = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            order = idx[np.argsort(scores[idx])[::-1]]
            while order.size > 0:
                i = int(order[0])
                keep.append(i)
                if order.size == 1:
                    break
                ious = cls._iou_one_to_many(boxes[i], boxes[order[1:]])
                order = order[1:][ious <= iou_thr]
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        return keep
