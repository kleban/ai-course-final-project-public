@echo off
chcp 65001 >nul
title Data Story - embeddings model check
cd /d "%~dp0\..\.."
".venv\Scripts\python.exe" -B -u tools\run_steps.py check-embeddings-load "Завантаження bge-m3 як у сервісі (повний текст помилки)::.venv\Scripts\python.exe -B tools\embeddings_diagnose.py"
pause
