from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import site
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from src.app_paths import configure_dll_search_path, ensure_runtime_layout, model_path, resource_root, runtime_dll_dir, user_data_dir, _candidate_runtime_dirs
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    def configure_dll_search_path() -> None: pass
    def ensure_runtime_layout(): return {"model": ROOT / "vendor_models" / "valorant_320_v11n.onnx"}
    def model_path() -> Path: return ROOT / "vendor_models" / "valorant_320_v11n.onnx"
    def resource_root() -> Path: return ROOT
    def runtime_dll_dir() -> Path: return ROOT / "runtime_dlls"
    def user_data_dir() -> Path: return Path.home() / "VisionForge"
    def _candidate_runtime_dirs() -> List[Path]: return [ROOT, ROOT / "runtime_dlls"]

OFFICIAL_LINKS: Dict[str, str] = {
    "NVIDIA 驱动下载": "https://www.nvidia.com/en-us/drivers/",
    "GeForce 驱动下载": "https://www.nvidia.com/en-us/geforce/drivers/",
    "CUDA Toolkit 下载": "https://developer.nvidia.com/cuda-12-8-0-download-archive",
    "CUDA Toolkit 历史版本": "https://developer.nvidia.com/cuda-toolkit-archive",
    "cuDNN 下载": "https://developer.nvidia.com/cudnn-downloads",
    "ONNX Runtime 安装说明": "https://onnxruntime.ai/docs/install/",
    "ONNX Runtime CUDA EP 要求": "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
    "TensorRT 下载": "https://developer.nvidia.com/tensorrt/download",
    "Microsoft VC++ 运行库": "https://aka.ms/vs/17/release/vc_redist.x64.exe",
    "Python Windows 下载": "https://www.python.org/downloads/windows/",
}

RECOMMENDED_RUNTIME = {
    "onnxruntime_gpu": "1.20.1（CUDA 12.x / cuDNN 9.x）",
    "cuda": "CUDA 12.x（建议 12.8；已有 12.1/12.6/12.8 也可优先沿用）",
    "cudnn": "cuDNN 9.x for CUDA 12.x",
    "vcredist": "Microsoft Visual C++ 2015-2022 x64",
}

INSTALL_STEPS = {
    "NVIDIA": [
        "打开 NVIDIA 驱动下载页面。",
        "选择当前显卡型号与 Windows 11 64-bit。",
        "安装 Game Ready 或 Studio Driver 后重启。",
        "重新运行本程序的环境诊断。",
    ],
    "ONNX_GPU": [
        "确认显卡驱动正常，nvidia-smi 能显示 GPU。",
        "安装与 onnxruntime-gpu 匹配的 CUDA/cuDNN。",
        "安装 Microsoft Visual C++ 2015-2022 x64 运行库。",
        "重新打开程序，确认 CUDAExecutionProvider 出现在 active providers。",
    ],
    "VCREDIST": [
        "打开 Microsoft 最新 Visual C++ Redistributable 页面。",
        "下载并安装 X64 版本 vc_redist.x64.exe。",
        "重启 Windows 或至少重启本程序。",
    ],
}

@dataclass
class DiagnosticItem:
    name: str
    status: str  # OK / WARN / ERROR / INFO
    detail: str
    hint: str = ""
    link_label: str = ""
    link_url: str = ""
    install_steps: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _run(cmd: List[str], timeout: float = 6.0) -> Tuple[int, str]:
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return int(cp.returncode), cp.stdout.strip()
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or "timeout")
    except Exception as e:
        return 1, repr(e)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _import_version(module: str, attr: str = "__version__") -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        return True, str(getattr(mod, attr, "installed"))
    except Exception as e:
        return False, repr(e)


def _path_dirs() -> List[Path]:
    try:
        return list(_candidate_runtime_dirs())
    except Exception:
        dirs: List[Path] = []
        rd = runtime_dll_dir()
        if rd.exists():
            dirs.append(rd)
        root = resource_root()
        if root.exists():
            dirs.append(root)
        for part in os.environ.get("PATH", "").split(os.pathsep):
            if part:
                try:
                    p = Path(part)
                    if p.exists():
                        dirs.append(p)
                except Exception:
                    pass
        return dirs

def _find_dll_patterns(patterns: Iterable[str], limit: int = 20) -> List[str]:
    matches: List[str] = []
    seen = set()
    for d in _path_dirs():
        try:
            for pat in patterns:
                candidates = list(d.glob(pat))
                # Some installations put cuDNN/TensorRT one level deeper than the known bin/lib dirs.
                if not candidates and d.name.lower() in {"nvidia", "cuda", "tensorrt"}:
                    candidates = list(d.rglob(pat))[:limit]
                for fp in candidates:
                    if not fp.is_file():
                        continue
                    s = str(fp)
                    key = s.lower()
                    if key not in seen:
                        seen.add(key)
                        matches.append(s)
                        if len(matches) >= limit:
                            return matches
        except Exception:
            continue
    return matches


def detect_nvidia_gpu() -> Tuple[Optional[Dict[str, str]], str]:
    if _which("nvidia-smi"):
        rc, out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=6)
        cuda_reported = "unknown"
        try:
            rc2, out2 = _run(["nvidia-smi"], timeout=6)
            m = re.search(r"CUDA Version:\s*([0-9.]+)", out2 or "")
            if m:
                cuda_reported = m.group(1)
        except Exception:
            pass
        if rc == 0 and out:
            line = out.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                return {"name": parts[0], "driver_version": parts[1], "cuda_version": cuda_reported}, out
        return None, out
    if platform.system().lower() == "windows":
        rc, out = _run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"], timeout=6)
        if rc == 0 and "NVIDIA" in out.upper():
            names = [x.strip() for x in out.splitlines() if x.strip()]
            return {"name": "; ".join(names), "driver_version": "unknown", "cuda_version": "unknown"}, out
        return None, out
    return None, "nvidia-smi not found"


def _version_tuple(v: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", v or "")
    return tuple(int(x) for x in nums[:4])


def _check_cuda_provider(mp: Path) -> DiagnosticItem:
    configure_dll_search_path()
    try:
        import numpy as np  # type: ignore
        import onnxruntime as ort  # type: ignore
        preload_notes: List[str] = []
        if hasattr(ort, "preload_dlls"):
            for kwargs in [
                {"cuda": True, "cudnn": True, "msvc": True},
                {"cuda": True, "cudnn": True, "msvc": True, "directory": ""},
            ]:
                try:
                    ort.preload_dlls(**kwargs)
                    preload_notes.append("运行库预加载成功")
                    break
                except TypeError:
                    continue
                except Exception as e:
                    preload_notes.append(f"运行库预加载未完全成功: {e!r}")
        providers = list(ort.get_available_providers())
        if "CUDAExecutionProvider" not in providers:
            # This usually means CPU-only onnxruntime was bundled or installed. Do not claim
            # the user's CUDA is missing; point to the exact package/provider mismatch.
            return DiagnosticItem(
                "GPU 推理",
                "ERROR",
                f"onnxruntime={getattr(ort, '__version__', '?')}; providers={providers}",
                "当前程序内置/当前环境的 ONNX Runtime 没有 GPU Provider。请使用本项目构建脚本安装 onnxruntime-gpu 后重新打包，或切换 CPU 模式。",
                "ONNX Runtime CUDA EP 要求",
                OFFICIAL_LINKS["ONNX Runtime CUDA EP 要求"],
                "重新执行构建脚本；确认 requirements-exe.txt 中是 onnxruntime-gpu；不要混入 CPU-only onnxruntime。",
            )
        if not mp.exists():
            return DiagnosticItem("GPU 推理", "WARN", f"providers={providers}; 模型未找到: {mp}", "模型资源未就绪，无法做真实 Session 测试。")
        sess = ort.InferenceSession(str(mp), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        active = list(sess.get_providers())
        inp = sess.get_inputs()[0]
        shape = [1 if not isinstance(x, int) else int(x) for x in inp.shape]
        if len(shape) == 4:
            x = np.zeros(shape, dtype=np.float32)
            sess.run([sess.get_outputs()[0].name], {inp.name: x})
        if "CUDAExecutionProvider" in active:
            return DiagnosticItem("GPU 推理", "OK", f"active={active}; dummy inference=OK", "GPU 推理链路可用。")
        return DiagnosticItem(
            "GPU 推理",
            "WARN",
            f"providers={providers}; active={active}; {'; '.join(preload_notes)}",
            "GPU Provider 已存在，但当前测试会话没有真正进入 GPU。通常是 EXE 打包未收集 Provider/CUDA DLL，或 DLL 搜索路径没有进入运行时；不直接判定你的电脑缺依赖。",
            "ONNX Runtime CUDA EP 要求",
            OFFICIAL_LINKS["ONNX Runtime CUDA EP 要求"],
            "运行 tools\\collect_cuda_dlls.py 后重新构建；确认 cuDNN 9.x、CUDA 12.x 和 VC++ x64 运行库齐全。",
        )
    except Exception as e:
        return DiagnosticItem(
            "GPU 推理",
            "WARN",
            f"GPU Session 测试未完成: {e!r}",
            "这不一定代表用户电脑缺依赖；也可能是打包资源路径、模型文件或 DLL 搜索路径未正确进入运行时。请先用构建脚本重新收集 DLL 并查看日志目录。",
            "ONNX Runtime CUDA EP 要求",
            OFFICIAL_LINKS["ONNX Runtime CUDA EP 要求"],
            "运行环境检查；执行 tools\\collect_cuda_dlls.py；重新构建 onefile EXE。",
        )

def _check_tensorrt_provider(mp: Path) -> DiagnosticItem:
    configure_dll_search_path()
    try:
        import onnxruntime as ort  # type: ignore
        providers = list(ort.get_available_providers())
        if "TensorrtExecutionProvider" not in providers:
            dlls = _find_dll_patterns(["nvinfer*.dll", "nvonnxparser*.dll", "onnxruntime_providers_tensorrt.dll"], limit=8)
            if dlls:
                return DiagnosticItem("TensorRT 加速", "WARN", "; ".join(dlls[:4]), "检测到部分 TensorRT 文件，但 ONNX Runtime 尚未启用 TensorRT Provider。GPU 仍可使用 CUDA 推理。", "TensorRT 下载", OFFICIAL_LINKS.get("TensorRT 下载", ""))
            return DiagnosticItem("TensorRT 加速", "INFO", "未启用 TensorRT Provider", "这是可选加速项。未安装 TensorRT 时，VisionForge 会使用 CUDA GPU 推理。", "TensorRT 下载", OFFICIAL_LINKS.get("TensorRT 下载", ""))
        return DiagnosticItem("TensorRT 加速", "OK", f"providers={providers}", "TensorRT Provider 可被 ONNX Runtime 识别。")
    except Exception as e:
        return DiagnosticItem("TensorRT 加速", "INFO", repr(e), "TensorRT 是可选加速项；不影响 CUDA GPU 推理。", "TensorRT 下载", OFFICIAL_LINKS.get("TensorRT 下载", ""))

def _check_vcredist() -> DiagnosticItem:
    if platform.system().lower() != "windows":
        return DiagnosticItem("Microsoft VC++ 运行库", "INFO", "非 Windows 环境，跳过注册表检测。")
    ps = r"""
$paths = @(
 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
)
foreach ($p in $paths) { if (Test-Path $p) { $v=Get-ItemProperty $p; Write-Output ($v.Version + ' Installed=' + $v.Installed); exit 0 } }
exit 1
"""
    rc, out = _run(["powershell", "-NoProfile", "-Command", ps], timeout=5)
    if rc == 0 and out.strip():
        return DiagnosticItem("Microsoft VC++ 运行库", "OK", out.strip())
    return DiagnosticItem("Microsoft VC++ 运行库", "WARN", out.strip() or "未检测到 x64 VC++ Runtime 注册表项", "缺失时会导致 onnxruntime/cv2 等 DLL 加载失败。", "Microsoft VC++ 运行库", OFFICIAL_LINKS["Microsoft VC++ 运行库"], "\n".join(INSTALL_STEPS["VCREDIST"]))


def collect_diagnostics(project_root: Optional[Path] = None) -> List[DiagnosticItem]:
    paths = ensure_runtime_layout()
    mp = model_path()
    items: List[DiagnosticItem] = []
    items.append(DiagnosticItem("程序路径", "INFO", f"config={paths.get('config')} | model={mp} | data={paths.get('user_data_dir')}"))
    items.append(DiagnosticItem("操作系统", "INFO", f"{platform.platform()} | Python {sys.version.split()[0]}", "系统信息读取正常。"))
    gpu, raw_gpu = detect_nvidia_gpu()
    if gpu:
        items.append(DiagnosticItem("显卡", "OK", f"{gpu.get('name')} | 驱动 {gpu.get('driver_version')} | 最高 CUDA {gpu.get('cuda_version')}", "显卡和驱动可被系统识别。", "NVIDIA 驱动下载", OFFICIAL_LINKS["NVIDIA 驱动下载"]))
    else:
        items.append(DiagnosticItem("显卡", "WARN", raw_gpu or "未检测到 NVIDIA GPU", "未能确认 NVIDIA 显卡或驱动状态。若你的电脑可以正常 GPU 推理，可忽略该项；否则请安装/更新驱动。", "NVIDIA 驱动下载", OFFICIAL_LINKS["NVIDIA 驱动下载"], "安装 NVIDIA Windows 11 x64 驱动后重启。"))
    for mod, label in [("onnxruntime", "ONNX Runtime"), ("cv2", "图像处理组件"), ("numpy", "数值计算组件"), ("yaml", "配置组件"), ("serial", "串口组件"), ("keyboard", "按键组件"), ("dxcam", "屏幕采集组件"), ("mss", "兼容采集组件"), ("psutil", "系统信息组件")]:
        ok, detail = _import_version(mod)
        items.append(DiagnosticItem(label, "OK" if ok else "ERROR", detail, "运行组件未就绪。请重新安装 VisionForge；如果你是作者本机构建，请重新执行保护型构建脚本。" if not ok else "组件已就绪。"))
    items.append(_check_vcredist())
    cudart = _find_dll_patterns(["cudart64*.dll"])
    cudnn = _find_dll_patterns(["cudnn*.dll", "cudnn64*.dll"])
    cublas = _find_dll_patterns(["cublas64*.dll", "cublasLt64*.dll"])
    gpu_provider_item = _check_cuda_provider(mp)
    gpu_ok = gpu_provider_item.status == "OK"
    # If the actual GPU inference session succeeds, dependency cards must be shown as OK even when
    # a standalone DLL filename scan cannot find every transitive DLL. The session result is the source of truth.
    items.append(DiagnosticItem("CUDA 运行库", "OK" if (cudart or gpu_ok) else "WARN", "; ".join(cudart[:4]) if cudart else ("GPU 推理已验证，可认为运行库可用" if gpu_ok else "未找到 cudart64*.dll"), "用于 GPU 推理的基础运行库。", "CUDA Toolkit 下载", OFFICIAL_LINKS["CUDA Toolkit 下载"]))
    items.append(DiagnosticItem("cuDNN", "OK" if (cudnn or gpu_ok) else "WARN", "; ".join(cudnn[:4]) if cudnn else ("GPU 推理已验证，可认为 cuDNN 链路可用" if gpu_ok else "未找到 cudnn*.dll"), "ONNX Runtime GPU 需要匹配的 cuDNN。", "cuDNN 下载", OFFICIAL_LINKS["cuDNN 下载"]))
    items.append(DiagnosticItem("cuBLAS", "OK" if (cublas or gpu_ok) else "WARN", "; ".join(cublas[:4]) if cublas else ("GPU 推理已验证，可认为 cuBLAS 链路可用" if gpu_ok else "未找到 cublas64*.dll"), "CUDA 线性代数运行库。", "CUDA Toolkit 下载", OFFICIAL_LINKS["CUDA Toolkit 下载"]))
    items.append(gpu_provider_item)
    items.append(_check_tensorrt_provider(mp))
    if mp.exists():
        items.append(DiagnosticItem("模型文件", "OK", f"{mp.name} | {mp.stat().st_size / 1024 / 1024:.2f} MB", "模型已就绪。"))
    else:
        items.append(DiagnosticItem("模型文件", "ERROR", str(mp), "模型文件缺失，无法启动运行。"))
    return items

def _simple_status_text(status: str) -> str:
    return {"OK": "正常", "WARN": "需要注意", "ERROR": "需要修复", "INFO": "信息"}.get(status, status)


def summarize(items: List[DiagnosticItem]) -> str:
    counts: Dict[str, int] = {"OK": 0, "WARN": 0, "ERROR": 0, "INFO": 0}
    by_name = {item.name: item for item in items}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1

    lines: List[str] = []
    if counts.get("ERROR", 0):
        lines.append("环境检查结果：需要修复")
    elif counts.get("WARN", 0):
        lines.append("环境检查结果：基本可用，但有项目需要注意")
    else:
        lines.append("环境检查结果：正常")
    lines.append("")

    gpu = by_name.get("显卡")
    if gpu:
        lines.append(f"显卡：{_simple_status_text(gpu.status)}")
        if gpu.status == "OK":
            lines.append(f"  {gpu.detail}")
        else:
            lines.append("  未能确认 NVIDIA GPU 或驱动状态。请先安装/更新 NVIDIA Windows 11 x64 驱动。")

    ort = by_name.get("ONNX Runtime")
    if ort:
        lines.append(f"ONNX Runtime：{_simple_status_text(ort.status)}" + (f"（{ort.detail}）" if ort.status == "OK" else ""))

    gpu_ep = by_name.get("GPU 推理")
    if gpu_ep:
        lines.append(f"GPU 推理：{_simple_status_text(gpu_ep.status)}")
        if gpu_ep.status != "OK":
            lines.append("  当前 GPU 推理链路没有完全打通。请按下面建议处理。")

    cudart = by_name.get("CUDA 运行库")
    cudnn = by_name.get("cuDNN")
    cublas = by_name.get("cuBLAS")
    if cudart:
        lines.append(f"CUDA：{_simple_status_text(cudart.status)}")
    if cudnn:
        lines.append(f"cuDNN：{_simple_status_text(cudnn.status)}")
    if cublas:
        lines.append(f"cuBLAS：{_simple_status_text(cublas.status)}")
    vc = by_name.get("Microsoft VC++ 运行库")
    if vc:
        lines.append(f"VC++ 运行库：{_simple_status_text(vc.status)}")

    lines.append("")
    lines.append("本项目推荐环境：")
    lines.append(f"  ONNX Runtime GPU：{RECOMMENDED_RUNTIME['onnxruntime_gpu']}")
    lines.append(f"  CUDA：{RECOMMENDED_RUNTIME['cuda']}")
    lines.append(f"  cuDNN：{RECOMMENDED_RUNTIME['cudnn']}")
    lines.append(f"  VC++：{RECOMMENDED_RUNTIME['vcredist']}")

    problem_items = [i for i in items if i.status in {"ERROR", "WARN"} and i.name not in {"程序路径", "操作系统"}]
    if problem_items:
        lines.append("")
        lines.append("需要处理的项目：")
        for item in problem_items:
            lines.append(f"- {item.name}：{_simple_status_text(item.status)}")
            if item.hint:
                lines.append(f"  建议：{item.hint}")
            if item.link_url:
                lines.append(f"  官方入口：{item.link_label} -> {item.link_url}")
            if item.install_steps:
                for step in item.install_steps.splitlines():
                    lines.append(f"  · {step}")
    else:
        lines.append("")
        lines.append("未发现会阻止运行的依赖问题。")

    lines.append("")
    lines.append("用户数据目录：" + str(user_data_dir() if 'user_data_dir' in globals() else resource_root()))
    return "\n".join(lines)
def main() -> int:
    items = collect_diagnostics()
    print(summarize(items))
    return 1 if any(i.status == "ERROR" for i in items) else 0

if __name__ == "__main__":
    raise SystemExit(main())
