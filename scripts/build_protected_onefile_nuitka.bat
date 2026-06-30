@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."

echo [INFO] Build VisionForge protected one-file EXE with GPU runtime hardening.
where python >nul 2>nul || (echo [ERROR] Python not found.& pause & exit /b 1)
if not exist ".venv" python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y onnxruntime >nul 2>nul
python -m pip install -r requirements-exe.txt
python -m pip install -r requirements-protection.txt
python tools\collect_cuda_dlls.py --strict-gpu || goto :err
python tools\audit_build_readiness.py || goto :err
python app_gui.py --self-test || goto :err
python tools\check_gpu_provider.py --strict || goto :err
python -m nuitka ^
  --mode=onefile ^
  --assume-yes-for-downloads ^
  --enable-plugin=tk-inter ^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=assets\app_icon.ico ^
  --include-package=src ^
  --include-package=onnxruntime ^
  --include-package=cv2 ^
  --include-package=numpy ^
  --include-package=customtkinter ^
  --include-module=tools.env_diagnostics ^
  --include-module=tools.config_tuner_gui ^
  --include-module=psutil ^
  --include-module=requests ^
  --include-data-files=config.default_v17_8_32.yaml=config.default_v17_8_32.yaml ^
  --include-data-files=config.yaml=config.yaml ^
  --include-data-files=vendor_models\valorant_320_v11n.onnx=vendor_models\valorant_320_v11n.onnx ^
  --include-data-dir=assets=assets ^
  --include-data-dir=docs=docs ^
  --include-data-dir=runtime_dlls=runtime_dlls ^
  --output-dir=dist_nuitka ^
  --output-filename=VisionForge_Protected.exe ^
  app_gui.py
if errorlevel 1 goto :err
echo [OK] Protected EXE: dist_nuitka\VisionForge_Protected.exe
pause
exit /b 0
:err
echo [ERROR] Protected build failed. Read docs\AI_AGENT_VISIONFORGE_V17_8_32_PROTECTED_BUILD_PROMPT.md
pause
exit /b 1
