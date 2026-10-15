@echo off
chcp 65001 >nul
title Data Story - full check
cd /d "%~dp0\..\.."
".venv\Scripts\python.exe" -B -u tools\run_steps.py check-only "Повна перевірка (pip check, ruff, mypy, тести)::.venv\Scripts\python.exe -B tools\check.py"
pause
