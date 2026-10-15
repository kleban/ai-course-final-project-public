@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The local environment is missing. Run Setup.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B tools\start.py --stop
if errorlevel 1 pause
