from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

APP_NAME = "VisionForge"
LEGACY_APP_NAMES = ["V17_8_Runtime_GUI"]
VERSION = "v17.8.32_gpu_runtime_scroll_log_fix"
DEFAULT_CONFIG_NAME = "config.default_v17_8_32.yaml"
MODEL_RELATIVE = Path("vendor_models") / "valorant_320_v11n.onnx"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def exe_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    """Root that contains bundled read-only resources.

    Source mode: project root.
    PyInstaller onefile: temporary extraction directory (sys._MEIPASS).
    PyInstaller onedir/Nuitka: executable directory, with fallback to _MEIPASS.
    """
    if hasattr(sys, "_MEIPASS"):
        p = Path(getattr(sys, "_MEIPASS")).resolve()
        if p.exists():
            return p
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _base_user_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base)
    return Path.home()


def user_data_dir() -> Path:
    base = _base_user_dir()
    if os.name == "nt":
        return base / APP_NAME
    return base / f".{APP_NAME.lower()}"


def legacy_user_data_dirs() -> list[Path]:
    base = _base_user_dir()
    out = []
    for name in LEGACY_APP_NAMES:
        out.append(base / name if os.name == "nt" else base / f".{name.lower()}")
    return out


def config_path() -> Path:
    if is_frozen():
        return user_data_dir() / "config.yaml"
    return exe_dir() / "config.yaml"


def default_config_path() -> Path:
    candidates = [
        resource_root() / DEFAULT_CONFIG_NAME,
        resource_root() / "config.yaml",
        exe_dir() / DEFAULT_CONFIG_NAME,
        exe_dir() / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def logs_dir() -> Path:
    return user_data_dir() / "logs" if is_frozen() else exe_dir() / "logs"


def backups_dir() -> Path:
    return user_data_dir() / "config_backups" if is_frozen() else exe_dir() / "config_backups"


def runtime_dll_dir() -> Path:
    return resource_root() / "runtime_dlls"


def model_path() -> Path:
    candidates = [
        resource_root() / MODEL_RELATIVE,
        exe_dir() / MODEL_RELATIVE,
        user_data_dir() / MODEL_RELATIVE,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def icon_path() -> Path:
    candidates = [resource_root() / "assets" / "app_icon.ico", exe_dir() / "assets" / "app_icon.ico"]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def ensure_runtime_layout() -> Dict[str, Path]:
    """Create writable runtime directories and seed config.yaml when needed.

    A one-file EXE cannot rely on `dist/.../config.yaml`, because bundled files are
    extracted to a temporary read-only folder. Runtime config/logs therefore live
    in `%LOCALAPPDATA%\\VisionForge` while the ONNX model remains bundled.
    """
    user_data_dir().mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    backups_dir().mkdir(parents=True, exist_ok=True)
    (user_data_dir() / "exports").mkdir(parents=True, exist_ok=True)
    cfg = config_path()
    if not cfg.exists():
        migrated = False
        for legacy in legacy_user_data_dirs():
            old_cfg = legacy / "config.yaml"
            if old_cfg.exists():
                try:
                    shutil.copy2(old_cfg, cfg)
                    migrated = True
                    break
                except Exception:
                    pass
        if not migrated:
            src = default_config_path()
            if not src.exists():
                raise FileNotFoundError(f"内置默认配置不存在: {src}")
            shutil.copy2(src, cfg)
    # Migrate an existing offline license from the pre-branding AppData folder.
    lic = user_data_dir() / "license.key"
    if not lic.exists():
        for legacy in legacy_user_data_dirs():
            old_lic = legacy / "license.key"
            if old_lic.exists():
                try:
                    shutil.copy2(old_lic, lic)
                    break
                except Exception:
                    pass
    return {
        "resource_root": resource_root(),
        "exe_dir": exe_dir(),
        "user_data_dir": user_data_dir(),
        "config": cfg,
        "default_config": default_config_path(),
        "logs": logs_dir(),
        "backups": backups_dir(),
        "model": model_path(),
        "runtime_dlls": runtime_dll_dir(),
    }


def apply_runtime_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Patch runtime paths after reading config, without overwriting user config."""
    cfg.setdefault("model", {})["path"] = str(model_path())
    cfg.setdefault("logging", {})["log_dir"] = str(logs_dir())
    return cfg


def _dedupe_dirs(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        try:
            if not d.exists() or not d.is_dir():
                continue
            key = str(d.resolve()).lower()
        except Exception:
            key = str(d).lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _candidate_runtime_dirs() -> list[Path]:
    """Return directories that may contain CUDA/cuDNN/TensorRT/ORT DLLs.

    This function is deliberately more complete than the old v31 path injector.
    In protected one-file builds ONNX Runtime is imported after this function runs;
    if CUDA/cuDNN directories are not already in Windows DLL search path, ORT can
    see CUDAExecutionProvider but still fail LoadLibrary(error 126) and fall back
    to CPU. The source of truth is not a simple PATH string: we also scan common
    CUDA, cuDNN, TensorRT, Python-package and bundled runtime folders.
    """
    dirs: list[Path] = []
    root = resource_root()
    exe = exe_dir()
    # Bundled folders first.
    for base in [runtime_dll_dir(), root / "runtime_dlls", exe / "runtime_dlls", root, exe]:
        if base.exists():
            dirs.append(base)
            for rel in ["onnxruntime", "onnxruntime/capi", "cv2", "numpy.libs", "nvidia"]:
                q = base / rel
                if q.exists():
                    dirs.append(q)
                    try:
                        dirs.extend([x for x in q.rglob("bin") if x.is_dir()])
                        dirs.extend([x for x in q.rglob("lib") if x.is_dir()])
                    except Exception:
                        pass
    # CUDA env vars.
    for key in ["CUDA_PATH", "CUDA_HOME"]:
        value = os.environ.get(key)
        if value:
            base = Path(value)
            for sub in ["bin", "lib/x64", "lib", ""]:
                dirs.append(base / sub)
    # Program Files common layouts. Support both CUDA\\v12.8\\bin and CUDA\\12.8\\bin.
    if os.name == "nt":
        for pf in [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]:
            if not pf:
                continue
            base = Path(pf)
            patterns = [
                "NVIDIA GPU Computing Toolkit/CUDA/v*/bin",
                "NVIDIA GPU Computing Toolkit/CUDA/*/bin",
                "NVIDIA GPU Computing Toolkit/CUDA/v*/lib/x64",
                "NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64",
                "NVIDIA/CUDNN*/bin",
                "NVIDIA/CUDNN*/lib",
                "NVIDIA/CUDNN/v*/bin",
                "NVIDIA/CUDNN/v*/lib",
                "NVIDIA/TensorRT*/bin",
                "NVIDIA/TensorRT*/lib",
                "NVIDIA/TensorRT/bin",
                "NVIDIA/TensorRT/lib",
            ]
            for pat in patterns:
                try:
                    dirs.extend([p for p in base.glob(pat) if p.is_dir()])
                except Exception:
                    pass
            # If user installed cuDNN/TensorRT into a custom NVIDIA subfolder, discover
            # the exact DLL parent directories without recursively scanning the whole disk.
            nvidia_root = base / "NVIDIA"
            if nvidia_root.exists():
                for pat in ["**/cudnn64_*.dll", "**/cudnn*.dll", "**/nvinfer*.dll", "**/nvonnxparser*.dll"]:
                    try:
                        for fp in list(nvidia_root.glob(pat))[:80]:
                            if fp.is_file():
                                dirs.append(fp.parent)
                    except Exception:
                        pass
            cuda_root = base / "NVIDIA GPU Computing Toolkit" / "CUDA"
            if cuda_root.exists():
                for pat in ["**/cudnn64_*.dll", "**/cudnn*.dll", "**/cublas64*.dll", "**/cudart64*.dll"]:
                    try:
                        for fp in list(cuda_root.glob(pat))[:80]:
                            if fp.is_file():
                                dirs.append(fp.parent)
                    except Exception:
                        pass
    # Python package vendor DLLs, e.g. nvidia-cudnn-cu12 installed by pip.
    try:
        import site
        site_roots = []
        try:
            site_roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            site_roots.append(site.getusersitepackages())
        except Exception:
            pass
        site_roots.extend(sys.path)
        for sp in site_roots:
            try:
                base = Path(sp)
            except Exception:
                continue
            if not base.exists():
                continue
            for rel in ["nvidia", "onnxruntime", "onnxruntime/capi", "torch/lib", "cv2", "numpy.libs"]:
                q = base / rel
                if q.exists():
                    dirs.append(q)
                    try:
                        dirs.extend([x for x in q.rglob("bin") if x.is_dir()])
                        dirs.extend([x for x in q.rglob("lib") if x.is_dir()])
                    except Exception:
                        pass
    except Exception:
        pass
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            try:
                dirs.append(Path(part))
            except Exception:
                pass
    return _dedupe_dirs(dirs)


def configure_dll_search_path() -> None:
    """Add bundled/local/system CUDA runtime DLL folders to Windows DLL search path.

    Must be called before importing onnxruntime. This is what prevents a packaged
    EXE from seeing CUDAExecutionProvider but failing LoadLibrary(error 126).
    """
    if os.name != "nt":
        return
    prefix: list[str] = []
    for d in _candidate_runtime_dirs():
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass
        prefix.append(str(d))
    current = os.environ.get("PATH", "")
    existing = {x.lower() for x in current.split(os.pathsep) if x}
    add = [x for x in prefix if x.lower() not in existing]
    if add:
        os.environ["PATH"] = os.pathsep.join(add + [current])
