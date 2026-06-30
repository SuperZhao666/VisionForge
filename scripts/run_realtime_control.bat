@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo [INFO] Realtime control mode. Console logs are disabled; detailed logs go to logs\run_*.txt.
echo [INFO] Control ON, visual OFF. Hold LSHIFT to move. F8 toggles. F10 quits.
echo.
python main.py --config config.yaml --source screen --control on --visual off --profile on --threaded-capture on
set "ERR=%ERRORLEVEL%"
echo.
echo [INFO] Latest log files:
dir /b /o-d logs\run_*.txt 2>nul | more +0
if not "%ERR%"=="0" echo [ERROR] Realtime control exited with code %ERR%.
pause
exit /b %ERR%
