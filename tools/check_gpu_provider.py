from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.app_paths import configure_dll_search_path, model_path
    configure_dll_search_path()
except Exception:
    pass

import numpy as np
import onnxruntime as ort

try:
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except TypeError:
            ort.preload_dlls()
except Exception as e:
    print("preload_dlls warning:", repr(e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--strict", action="store_true", help="return non-zero unless CUDAExecutionProvider is active")
    args = ap.parse_args()
    mp = Path(args.model) if args.model else model_path()
    print("onnxruntime:", ort.__version__)
    available = list(ort.get_available_providers())
    print("available providers:", available)
    wanted = [p for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
    if not wanted:
        wanted = ["CPUExecutionProvider"]
    try:
        try:
            sess = ort.InferenceSession(str(mp), providers=wanted)
        except Exception as e:
            if "TensorrtExecutionProvider" in wanted and "CUDAExecutionProvider" in available:
                print("TensorRT provider init warning, retry CUDA:", repr(e))
                wanted = [p for p in wanted if p != "TensorrtExecutionProvider"]
                sess = ort.InferenceSession(str(mp), providers=wanted)
            else:
                raise
        active = list(sess.get_providers())
        print("active providers:", active)
        inp = sess.get_inputs()[0]
        shape = [1 if not isinstance(x, int) else int(x) for x in inp.shape]
        x = np.zeros(shape, dtype=np.float32)
        _ = sess.run([sess.get_outputs()[0].name], {inp.name: x})
        print("dummy inference: OK")
        if "CUDAExecutionProvider" in active or "TensorrtExecutionProvider" in active:
            print("GPU inference: OK")
            return 0
        print("GPU inference: NOT ACTIVE; current fallback is CPU")
        return 2 if args.strict else 0
    except Exception as e:
        print("GPU/session check failed:", repr(e))
        print("请修复 CUDA 12.x、cuDNN 9.x、ONNX Runtime GPU Provider、Microsoft VC++ x64 运行库或 DLL 搜索路径。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
