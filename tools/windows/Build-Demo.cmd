@echo off
chcp 65001 >nul
title Data Story - demo: data, model, conclusions
cd /d "%~dp0\..\.."
echo Seoul demo into runtime. Stop the service first (Stop.cmd).
if exist tmp\demo-137-test rd /s /q tmp\demo-137-test
".venv\Scripts\python.exe" -B -u tools\run_steps.py build-demo-test "Test run on a copy, no OpenAI::.venv\Scripts\python.exe -B tools\build_demo.py --database tmp\demo-137-test\data.sqlite3 --no-ai"
if errorlevel 1 (
  echo Test run failed - runtime NOT changed. Log: tmp\build-demo-test.log
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B -u tools\run_steps.py build-demo "Demo into runtime (OpenAI: conclusions draft)::.venv\Scripts\python.exe -B tools\build_demo.py"
pause
