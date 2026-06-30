@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."
echo [INFO] Project root: %CD%
echo.
echo [1/5] Python
python --version
python -c "import sys; print(sys.executable)"
if errorlevel 1 goto :fail

echo.
echo [2/5] ONNX Runtime providers
python -c "import onnxruntime as ort; print('onnxruntime:', ort.__version__); print('available providers:', ort.get_available_providers())"
if errorlevel 1 goto :fail

echo.
echo [3/5] NVIDIA SMI
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [WARN] nvidia-smi not found.
) else (
  nvidia-smi
)

echo.
echo [4/5] Model inspect
if exist "vendor_models\valorant_320_v11n.onnx" (
  python tools\inspect_onnx.py --model vendor_models\valorant_320_v11n.onnx --providers auto
) else (
  echo [WARN] Missing vendor_models\valorant_320_v11n.onnx
)

echo.
echo [5/5] Actual GPU session test
if exist "vendor_models\valorant_320_v11n.onnx" (
  python tools\check_gpu_provider.py --model vendor_models\valorant_320_v11n.onnx
) else (
  echo [WARN] Skip GPU session test because model is missing.
)

echo.
echo [OK] Diagnose script finished.
pause
exit /b 0

:fail
echo.
echo [ERROR] Diagnose failed. Copy the output and send it for diagnosis.
pause
exit /b 1
