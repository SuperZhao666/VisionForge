@echo off
chcp 65001 >nul
cd /d "%~dp0.."
python tools\config_tuner_gui.py --config config.yaml --host 127.0.0.1 --port 8765 --open-browser
pause
