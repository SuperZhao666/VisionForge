@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0.."
python -m tools.env_diagnostics
echo.
pause
