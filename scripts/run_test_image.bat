@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."

echo [INFO] Project root: %CD%
if not exist "vendor_models\valorant_320_v11n.onnx" (
  echo [ERROR] Missing vendor_models\valorant_320_v11n.onnx
  echo Put the vendor ONNX model into vendor_models first.
  pause
  exit /b 1
)
if not exist "samples\test.jpg" (
  echo [ERROR] Missing samples\test.jpg
  echo Put one test image at samples\test.jpg first.
  pause
  exit /b 1
)
if not exist "outputs" mkdir outputs
python tools\test_image.py --model vendor_models\valorant_320_v11n.onnx --imgsz 320 --image samples\test.jpg --out outputs\test.jpg
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo [ERROR] Test image failed. Exit code: %ERR%
) else (
  echo [OK] Test image finished: outputs\test.jpg
)
pause
exit /b %ERR%
