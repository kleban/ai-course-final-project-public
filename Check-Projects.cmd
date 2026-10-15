@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tools\check_projects.py
) else (
    py -3.13 tools\check_projects.py
)
set "DATA_STORY_EXIT=%ERRORLEVEL%"
if not "%DATA_STORY_EXIT%"=="0" echo Project checks failed. Python 3.13 or a local .venv is required.
pause
exit /b %DATA_STORY_EXIT%
