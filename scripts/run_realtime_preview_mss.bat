@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo [INFO] Starting realtime preview with MSS. Console logs are disabled; detailed logs go to logs\run_*.txt.
echo.
python main.py --config config.yaml --source screen --control off --visual on --profile on --threaded-capture on --capture-backend mss --console-log off
set "ERR=%ERRORLEVEL%"
echo.
echo [INFO] Latest log files:
dir /b /o-d logs\run_*.txt 2>nul | more +0
if not "%ERR%"=="0" echo [ERROR] Realtime preview MSS exited with code %ERR%.
pause
exit /b %ERR%
