@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" app.py
if errorlevel 1 (
    echo.
    echo Startup failed. Run: python -m pip install -r requirements.txt
    pause
)
endlocal
