from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import site
import sys
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime_dlls"

PATTERNS = [
    # CUDA Runtime
    "cudart64*.dll", "nvrtc64*.dll", "nvrtc-builtins64*.dll",
    # CUDA Math / DNN libs
    "cublas64*.dll", "cublasLt64*.dll", "cufft64*.dll", "curand64*.dll",
    "cusolver64*.dll", "cusparse64*.dll", "cudnn*.dll", "cudnn64*.dll",
    # ONNX Runtime provider binaries
    "onnxruntime*.dll", "onnxruntime_providers_*.dll",
    # TensorRT optional acceleration
    "nvinfer*.dll", "nvonnxparser*.dll", "nvinfer_plugin*.dll", "myelin*.dll",
    # Common transitive runtime
    "zlibwapi.dll", "vcruntime140*.dll", "msvcp140*.dll", "concrt140*.dll",
]

REQUIRED_FOR_GPU = [
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
]


def _site_roots() -> List[Path]:
    roots: List[Path] = []
    for fn in [site.getusersitepackages]:
        try:
            roots.append(Path(fn()))
        except Exception:
            pass
    try:
        roots.extend(Path(x) for x in site.getsitepackages())
    except Exception:
        pass
    roots.extend(Path(x) for x in sys.path if x)
    return [p for p in roots if p.exists()]


def candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    for key in ["CUDA_PATH", "CUDA_HOME"]:
        base = os.environ.get(key)
        if base:
            for sub in ["bin", "lib", "lib/x64", ""]:
                p = Path(base) / sub
                if p.exists():
                    dirs.append(p)
    if os.name == "nt":
        for pf in [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]:
            if not pf:
                continue
            base = Path(pf)
            for pattern in [
                "NVIDIA GPU Computing Toolkit/CUDA/v*/bin",
                "NVIDIA GPU Computing Toolkit/CUDA/*/bin",
                "NVIDIA GPU Computing Toolkit/CUDA/v*/lib/x64",
                "NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64",
                "NVIDIA/CUDNN*/bin",
                "NVIDIA/CUDNN*/lib",
                "NVIDIA/CUDNN/v*/bin",
                "NVIDIA/CUDNN/v*/lib",
                "NVIDIA/TensorRT*/lib",
                "NVIDIA/TensorRT*/bin",
                "NVIDIA/TensorRT/lib",
                "NVIDIA/TensorRT/bin",
            ]:
                try:
                    dirs.extend([p for p in base.glob(pattern) if p.exists()])
                except Exception:
                    pass

            # Discover parent directories for cuDNN/TensorRT DLLs installed under CUDA\12.x\bin or custom NVIDIA folders.
            for root_name in ["NVIDIA", "NVIDIA GPU Computing Toolkit"]:
                root = base / root_name
                if not root.exists():
                    continue
                for dll_pat in ["**/cudnn64_*.dll", "**/cudnn*.dll", "**/cublas64*.dll", "**/cudart64*.dll", "**/nvinfer*.dll", "**/nvonnxparser*.dll"]:
                    try:
                        for fp in list(root.glob(dll_pat))[:120]:
                            if fp.is_file():
                                dirs.append(fp.parent)
                    except Exception:
                        pass
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            p = Path(part)
            if p.exists():
                dirs.append(p)
    for sp in _site_roots():
        for rel in [
            "onnxruntime/capi", "onnxruntime", "cv2", "numpy.libs", "torch/lib",
            "nvidia", "nvidia/cudnn/bin", "nvidia/cublas/bin", "nvidia/cuda_runtime/bin",
            "nvidia/cuda_nvrtc/bin", "nvidia/cufft/bin", "nvidia/curand/bin",
            "nvidia/cusolver/bin", "nvidia/cusparse/bin",
        ]:
            q = sp / rel
            if q.exists():
                dirs.append(q)
                try:
                    dirs.extend([x for x in q.rglob("bin") if x.exists()])
                    dirs.extend([x for x in q.rglob("lib") if x.exists()])
                except Exception:
                    pass
    uniq: List[Path] = []
    seen: Set[str] = set()
    for d in dirs:
        try:
            key = str(d.resolve()).lower()
        except Exception:
            key = str(d).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def find_files(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[str] = set()
    for d in candidate_dirs():
        for pat in patterns:
            try:
                for fp in d.glob(pat):
                    if not fp.is_file():
                        continue
                    key = fp.name.lower()
                    if key not in seen:
                        seen.add(key)
                        files.append(fp)
            except Exception:
                pass
    return files


def ensure_gpu_package() -> int:
    try:
        import onnxruntime as ort  # type: ignore
        providers = list(ort.get_available_providers())
        print(f"[INFO] onnxruntime={getattr(ort, '__version__', '?')} providers={providers}")
        if "CUDAExecutionProvider" not in providers:
            print("[ERROR] 当前环境不是 onnxruntime-gpu，或 CUDA Provider 未被安装。禁止构建保护型 GPU 发行版。")
            print("[HINT] 运行：python -m pip uninstall -y onnxruntime && python -m pip install onnxruntime-gpu==1.20.1")
            return 2
        return 0
    except Exception as e:
        print(f"[ERROR] onnxruntime 检查失败: {e!r}")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-gpu", action="store_true", help="require ONNX Runtime CUDA provider DLLs")
    args = parser.parse_args()
    if OUT.exists():
        for fp in OUT.glob("*"):
            if fp.is_file():
                try:
                    fp.unlink()
                except Exception:
                    pass
    OUT.mkdir(exist_ok=True)
    rc = ensure_gpu_package()
    if args.strict_gpu and rc != 0:
        return rc
    found = find_files(PATTERNS)
    if not found:
        print("[WARN] No runtime DLLs found. EXE can build, but GPU runtime may fail on other machines.")
        return 3 if args.strict_gpu else 0
    print(f"[INFO] Copying {len(found)} runtime DLL(s) into {OUT}")
    copied: Set[str] = set()
    for fp in found:
        dst = OUT / fp.name
        try:
            shutil.copy2(fp, dst)
            copied.add(fp.name.lower())
            print(f"[OK] {fp} -> {dst}")
        except Exception as e:
            print(f"[WARN] copy failed: {fp}: {e}")
    missing = [name for name in REQUIRED_FOR_GPU if name.lower() not in copied]
    if missing:
        print(f"[ERROR] Missing required ONNX Runtime GPU provider DLL(s): {missing}")
        print("[HINT] 请确认 requirements-exe.txt 安装的是 onnxruntime-gpu，并重新运行本脚本。")
        return 4 if args.strict_gpu else 0
    if not any(name.startswith("cudnn") for name in copied):
        print("[WARN] cuDNN DLL not collected. 如果目标机器没有系统级 cuDNN，GPU 会话可能回退 CPU。")
    if not any(name.startswith("nvinfer") for name in copied):
        print("[INFO] TensorRT DLL not collected. TensorRT 为可选加速；CUDA GPU 推理仍可运行。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
