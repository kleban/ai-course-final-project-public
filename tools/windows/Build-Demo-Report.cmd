@echo off
chcp 65001 >nul
title Data Story - demo: report
cd /d "%~dp0\..\.."
echo Stage 2: report after the conclusions are approved. Stop the service first (Stop.cmd).
".venv\Scripts\python.exe" -B -u tools\run_steps.py build-demo-report "Demo stage 2::.venv\Scripts\python.exe -B tools\build_demo.py --report"
pause
