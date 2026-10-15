@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if not exist ".venv\Scripts\python.exe" (
    echo Run Setup.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B tools\check.py
set "DATA_STORY_EXIT=%ERRORLEVEL%"
pause
exit /b %DATA_STORY_EXIT%
