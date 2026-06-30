@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."
echo [INFO] Installing requirements with Tsinghua mirror...
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" echo [ERROR] Install failed. Exit code: %ERR%.
if "%ERR%"=="0" echo [OK] Install finished.
pause
exit /b %ERR%
