from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onnxruntime as ort

try:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception as e:
    print("preload_dlls warning:", repr(e))


def parse_providers(s: str):
    if s.lower() in ("auto", "default"):
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif s.lower() == "cpu":
        wanted = ["CPUExecutionProvider"]
    elif s.lower() == "cuda":
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        wanted = [x.strip() for x in s.split(",") if x.strip()]
    available = ort.get_available_providers()
    return [p for p in wanted if p in available] or ["CPUExecutionProvider"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--providers", default="auto", help="auto | cuda | cpu | CUDAExecutionProvider,CPUExecutionProvider")
    args = ap.parse_args()
    print("onnxruntime:", ort.__version__)
    print("available providers:", ort.get_available_providers())
    providers = parse_providers(args.providers)
    print("requested providers:", providers)
    sess = ort.InferenceSession(args.model, providers=providers)
    print("active providers:", sess.get_providers())
    print("inputs:")
    for i in sess.get_inputs():
        print(" ", i.name, i.shape, i.type)
    print("outputs:")
    for o in sess.get_outputs():
        print(" ", o.name, o.shape, o.type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
