@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."
if not exist ".venv" python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements-owner.txt
python tools\license_keygen.py %*
pause
