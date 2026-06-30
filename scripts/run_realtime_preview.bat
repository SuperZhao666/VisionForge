@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo [INFO] Starting realtime preview. Console logs are disabled; detailed logs go to logs\run_*.txt.
echo [INFO] Control OFF, visual ON. Press F10 to quit.
echo.
python main.py --config config.yaml --source screen --control off --visual on --profile on --threaded-capture on
set "ERR=%ERRORLEVEL%"
echo.
echo [INFO] Latest log files:
dir /b /o-d logs\run_*.txt 2>nul | more +0
if not "%ERR%"=="0" echo [ERROR] Realtime preview exited with code %ERR%.
pause
exit /b %ERR%
