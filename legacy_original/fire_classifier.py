import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import onnxruntime as ort

from model_input import adapt_batch_layout, infer_input_layout, make_model_input


class FireClassifier:
    def __init__(self, model_path, img_size, fire_threshold,
                 model_filter_threshold, model_filter_consecutive,
                 model_filter_cache_ttl, color_lower_np=None, color_upper_np=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        sess_opt = ort.SessionOptions()
        sess_opt.enable_mem_pattern = True
        sess_opt.enable_cpu_mem_arena = True
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        self.input_channels, self.channels_first = infer_input_layout(self.input_meta.shape)

        self.img_size = int(img_size)
        self.fire_threshold = float(fire_threshold)
        self.model_filter_threshold = float(model_filter_threshold)
        self.model_filter_consecutive = int(model_filter_consecutive)
        self.cache_ttl = float(model_filter_cache_ttl)
        self._cache = OrderedDict()
        self._cache_max_size = 512
        self._last_cache_cleanup = 0.0

        # 保留参数以兼容旧初始化；主流程已直接复用全局 mask，不再重复 HSV 转换。
        self.color_lower_np = color_lower_np
        self.color_upper_np = color_upper_np

        if self.channels_first:
            self._input_tensor = np.zeros((1, self.input_channels, self.img_size, self.img_size), dtype=np.float32)
            layout = "NCHW"
        else:
            self._input_tensor = np.zeros((1, self.img_size, self.img_size, self.input_channels), dtype=np.float32)
            layout = "NHWC"

        mode = {1: "mask-only 旧模型兼容", 3: "RGB", 4: "RGB+mask 新模型"}.get(self.input_channels, "未知")
        print(f"[FireClassifier] 模型已加载 providers={providers} img_size={self.img_size} input={self.input_channels}ch {layout} mode={mode}")

    def _cache_key(self, target):
        bbox_x = getattr(target, "bbox_x", None)
        bbox_y = getattr(target, "bbox_y", None)
        base_x = int(bbox_x) if bbox_x is not None else int(target.x)
        base_y = int(bbox_y) if bbox_y is not None else int(target.y)
        return (
            self.input_channels,
            base_x,
            base_y,
            int(target.w),
            int(target.h),
            int(round(float(getattr(target, "area", 0.0)))),
            int(round(float(getattr(target, "x", base_x)))),
            int(round(float(getattr(target, "y", base_y)))),
        )

    def _cleanup_cache(self, now):
        if now - self._last_cache_cleanup < max(self.cache_ttl, 0.05):
            return
        self._last_cache_cleanup = now
        if not self._cache:
            return
        expired = [k for k, v in self._cache.items() if now - v[1] >= self.cache_ttl]
        for k in expired:
            self._cache.pop(k, None)
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _target_bbox(target):
        bbox_x = getattr(target, "bbox_x", None)
        bbox_y = getattr(target, "bbox_y", None)
        if bbox_x is not None and bbox_y is not None:
            x1, y1 = int(bbox_x), int(bbox_y)
            x2, y2 = x1 + int(target.w), y1 + int(target.h)
        else:
            x1 = int(target.x - target.w / 2)
            y1 = int(target.y - target.h * 0.1)
            x2 = int(target.x + target.w / 2)
            y2 = int(target.y + target.h * 0.9)
        return x1, y1, x2, y2

    def predict_proba(self, frame_or_mask, mask_or_target=None, target=None, use_cache=True):
        """返回 P(目标) ∈ [0, 1]。

        新推荐调用：predict_proba(frame_rgb, mask_frame, target)
        旧兼容调用：predict_proba(mask_frame, target)

        新版 4 通道模型会同时读取 RGB ROI 与 mask ROI；旧 1 通道模型会自动只读 mask。
        """
        if target is None:
            frame_rgb = None
            mask_frame = frame_or_mask
            target = mask_or_target
        else:
            frame_rgb = frame_or_mask
            mask_frame = mask_or_target

        if target is None or mask_frame is None:
            return 0.0

        now = time.perf_counter()
        key = self._cache_key(target)

        if use_cache and self.cache_ttl > 0:
            hit = self._cache.get(key)
            if hit is not None:
                prob, ts = hit
                if now - ts < self.cache_ttl:
                    self._cache.move_to_end(key)
                    return prob
                self._cache.pop(key, None)

        x1, y1, x2, y2 = self._target_bbox(target)
        h, w = mask_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi_mask = mask_frame[y1:y2, x1:x2]
        total_pixels = roi_mask.size
        if total_pixels <= 0:
            return 0.0

        white_pixels = int(np.count_nonzero(roi_mask))
        if white_pixels / total_pixels < 0.01:
            return 0.0

        roi_rgb = None
        if frame_rgb is not None and self.input_channels in (3, 4):
            roi_rgb = frame_rgb[y1:y2, x1:x2]

        try:
            sample = make_model_input(
                roi_rgb, roi_mask, self.img_size,
                channels=self.input_channels,
                source_color="RGB",
            )
            batch = adapt_batch_layout(sample.reshape(1, *sample.shape), self.channels_first)
            np.copyto(self._input_tensor, batch, casting="same_kind")
            outputs = self.session.run([self.output_name], {self.input_name: self._input_tensor})
            prob = float(np.asarray(outputs[0]).reshape(-1)[0])
            prob = min(max(prob, 0.0), 1.0)
        except Exception:
            # 模型异常时拒绝，避免故障时无条件放行。
            prob = 0.0

        if use_cache and self.cache_ttl > 0:
            self._cache[key] = (prob, now)
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_max_size:
                self._cleanup_cache(now)
        return prob
