@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."
echo [INFO] Project root: %CD%
echo [INFO] Running Windows CN setup...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows_cn.ps1"
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo [ERROR] Setup failed. Exit code: %ERR%
  echo [HINT] Copy the error text and send it for diagnosis.
) else (
  echo [OK] Setup finished.
)
echo.
pause
exit /b %ERR%
