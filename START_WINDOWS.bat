@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL%==0 (set "PYTHON=py -3") else (set "PYTHON=python")
if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv .venv
    if errorlevel 1 (echo Could not create Python virtual environment. & pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Failed to install Python packages. & pause & exit /b 1)
start "" "http://127.0.0.1:7865"
".venv\Scripts\python.exe" app.py
pause
