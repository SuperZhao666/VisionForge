@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."
echo [INFO] Opening official driver, CUDA, cuDNN, and VC++ links...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_driver_cuda_links.ps1"
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" echo [ERROR] Link opener failed. Exit code: %ERR%.
pause
exit /b %ERR%
