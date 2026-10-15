@echo off
chcp 65001 >nul
title Data Story - index books
cd /d "%~dp0\..\.."
".venv\Scripts\python.exe" -B -u tools\run_steps.py index-books "Індексація 620 фрагментів книжок (bge-m3, ~5-10 хв)::.venv\Scripts\python.exe -B -u tools\index_sources.py" "Оцінка пошуку (правила + книжки)::.venv\Scripts\python.exe -B -u tools\knowledge_eval.py" "Повна перевірка (pip check, ruff, mypy, тести)::.venv\Scripts\python.exe -B tools\check.py"
pause
