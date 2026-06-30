"""统一模型输入预处理：支持旧 1 通道 mask 模型，以及新版 4 通道 RGB+mask 模型。

核心约定：
- 1 通道：只输入二值 mask，兼容旧模型。
- 3 通道：输入 RGB 原图 ROI。
- 4 通道：输入 RGB 原图 ROI + mask 辅助通道，默认新版训练/推理格式。
"""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

SUPPORTED_CHANNELS = (1, 3, 4)
DEFAULT_CHANNELS = 4


def _dim_to_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def infer_input_layout(input_shape: Iterable) -> Tuple[int, bool]:
    """从 ONNX/TensorFlow 输入 shape 判断通道数与布局。

    返回：
        channels: 1 / 3 / 4，无法判断时默认 4。
        channels_first: True 表示 NCHW；False 表示 NHWC。
    """
    shape = list(input_shape or [])
    if len(shape) != 4:
        return DEFAULT_CHANNELS, False

    dims = [_dim_to_int(x) for x in shape]
    c_first = dims[1]
    c_last = dims[3]

    if c_last in SUPPORTED_CHANNELS:
        return int(c_last), False
    if c_first in SUPPORTED_CHANNELS:
        return int(c_first), True

    # tf2onnx 通常导出 NHWC；无法判断时按新版 4 通道 NHWC 处理。
    return DEFAULT_CHANNELS, False


def ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask 不能为空")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask


def make_model_input(
    roi_img: np.ndarray | None,
    roi_mask: np.ndarray,
    img_size: int,
    channels: int = DEFAULT_CHANNELS,
    source_color: str = "BGR",
) -> np.ndarray:
    """把目标 ROI 转成模型输入张量，返回 NHWC 单样本：H×W×C，float32 [0,1]。

    roi_img:
        彩色原图 ROI。source_color="BGR" 表示 OpenCV 读入图；"RGB" 表示截图主循环图。
    roi_mask:
        与 ROI 对应的单通道 mask。
    channels:
        1：mask；3：RGB；4：RGB+mask。
    """
    img_size = max(8, int(img_size))
    channels = int(channels)
    if channels not in SUPPORTED_CHANNELS:
        channels = DEFAULT_CHANNELS

    mask = ensure_uint8_mask(roi_mask)
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_f = (mask_resized.astype(np.float32) / 255.0).reshape(img_size, img_size, 1)

    if channels == 1:
        return mask_f

    if roi_img is None or roi_img.size <= 0:
        rgb = np.zeros((img_size, img_size, 3), dtype=np.float32)
    else:
        img = roi_img
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # 先丢 alpha，再按颜色顺序处理。
            img = img[:, :, :3]

        order = (source_color or "BGR").upper()
        if order == "BGR":
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif order == "RGB":
            img_rgb = img
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0

    if channels == 3:
        return rgb.astype(np.float32)

    return np.concatenate([rgb, mask_f], axis=-1).astype(np.float32)


def adapt_batch_layout(batch_nhwc: np.ndarray, channels_first: bool) -> np.ndarray:
    """把 NHWC batch 转成模型需要的布局。"""
    if channels_first:
        return np.transpose(batch_nhwc, (0, 3, 1, 2)).astype(np.float32, copy=False)
    return batch_nhwc.astype(np.float32, copy=False)
