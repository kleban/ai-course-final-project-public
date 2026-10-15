@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist "tmp\install" mkdir "tmp\install"
set "TEMP=%CD%\tmp\install"
set "TMP=%TEMP%"
set "PIP_CACHE_DIR=%CD%\tmp\pip-cache"
set "PIP_CONFIG_FILE=NUL"
set "PIP_EXTRA_INDEX_URL=https://pypi.org/simple"
set PYTHONDONTWRITEBYTECODE=1
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.13 with the Windows Python Launcher is required.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple -r requirements.lock
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B tools/setup_quarto.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B tools/setup_embeddings.py
if errorlevel 1 goto failed
echo Setup complete. Open Start.cmd to launch Data Story.
pause
exit /b 0
:failed
echo Setup failed. Review the message above. Your project data has not been removed.
pause
exit /b 1
