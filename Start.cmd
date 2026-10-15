@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if not exist ".venv\Scripts\python.exe" (
    echo Please run Setup.cmd once, then open Start.cmd again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B tools\start.py
set "DATA_STORY_EXIT=%ERRORLEVEL%"
if not "%DATA_STORY_EXIT%"=="0" pause
exit /b %DATA_STORY_EXIT%
