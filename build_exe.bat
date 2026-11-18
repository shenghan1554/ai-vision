@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Build AI Vision Portable

echo [1/4] Checking 64-bit Python...
python -c "import struct,sys; print(sys.version); assert struct.calcsize('P') == 8, '64-bit Python is required'"
if errorlevel 1 goto :error

if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating local build environment...
    python -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [2/4] Using existing local build environment...
)
set "PYTHON=.venv\Scripts\python.exe"

echo [3/4] Installing/verifying dependencies...
"%PYTHON%" -m pip install -r requirements-build.txt
if errorlevel 1 goto :error

echo [4/4] Building portable EXE folder...
"%PYTHON%" -m PyInstaller --noconfirm --clean AIVision.spec
if errorlevel 1 goto :error

copy /y README.md "dist\AIVision\使用说明.md" >nul
copy /y 快速使用.txt "dist\AIVision\快速使用.txt" >nul
copy /y THIRD_PARTY_NOTICES.md "dist\AIVision\第三方许可说明.md" >nul
copy /y settings.example.json "dist\AIVision\settings.example.json" >nul
copy /y "docs\CAMERA_SETUP.md" "dist\AIVision\相机接入指南.md" >nul

echo.
echo Build completed:
echo %~dp0dist\AIVision\AIVision.exe
echo Keep the EXE together with the _internal folder.
pause
exit /b 0

:error
echo.
echo Build failed. Review the error messages above.
pause
exit /b 1
