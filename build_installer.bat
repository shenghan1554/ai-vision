@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Build AI Vision Installer

set "ISCC="
for %%P in (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
) do (
    if exist "%%~P" set "ISCC=%%~P"
)

if not defined ISCC (
    echo Inno Setup 6 not found.
    echo Install it with:  winget install --id JRSoftware.InnoSetup -e
    pause
    exit /b 1
)

if not exist "dist\AIVision\AIVision.exe" (
    echo Portable build missing. Run build_exe.bat first.
    pause
    exit /b 1
)

echo Building installer...
"%ISCC%" "installer\AIVision.iss"
if errorlevel 1 (
    echo Installer build failed.
    pause
    exit /b 1
)

echo.
echo Done. See the release folder for the setup EXE.
pause
endlocal