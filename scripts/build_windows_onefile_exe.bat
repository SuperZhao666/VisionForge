@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."

echo [INFO] Build VisionForge one-file EXE with PyInstaller.
where python >nul 2>nul || (echo [ERROR] Python not found.& pause & exit /b 1)
if not exist ".venv" python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-exe.txt
python tools\collect_cuda_dlls.py
python tools\audit_build_readiness.py || goto :err
python app_gui.py --self-test || goto :err
python -m PyInstaller --clean --noconfirm packaging\VisionForge_ONEFILE.spec || goto :err
echo [OK] EXE: dist\VisionForge.exe
pause
exit /b 0
:err
echo [ERROR] Build failed.
pause
exit /b 1
