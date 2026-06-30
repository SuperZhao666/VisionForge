from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from model_input import adapt_batch_layout, infer_input_layout


def validate_onnx_model(model_path: str, img_size: int) -> dict:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    channels, channels_first = infer_input_layout(inp.shape)
    x = np.zeros((1, int(img_size), int(img_size), channels), dtype=np.float32)
    x = adapt_batch_layout(x, channels_first)
    y = sess.run([out.name], {inp.name: x})[0]
    prob = float(np.asarray(y).reshape(-1)[0])
    if not np.isfinite(prob):
        raise ValueError("ONNX validation produced non-finite output")
    return {
        "input_shape": [str(v) for v in inp.shape],
        "channels": int(channels),
        "layout": "NCHW" if channels_first else "NHWC",
        "zero_prob": prob,
    }


def atomic_publish_onnx(temp_model_path: str, final_model_path: str, img_size: int, ready_path: str | None = None) -> dict:
    temp_path = Path(temp_model_path)
    final_path = Path(final_model_path)
    if not temp_path.exists():
        raise FileNotFoundError(f"temporary ONNX model not found: {temp_path}")

    meta = validate_onnx_model(str(temp_path), img_size)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(temp_path), str(final_path))

    ready = Path(ready_path) if ready_path else final_path.with_suffix(final_path.suffix + ".ready.json")
    payload = {
        "model_path": str(final_path),
        "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mtime": os.path.getmtime(final_path),
        **meta,
    }
    tmp_ready = ready.with_suffix(ready.suffix + ".tmp")
    with open(tmp_ready, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp_ready), str(ready))
    return payload
