@echo off
setlocal
chcp 65001 >nul
rem ============================================================
rem 安装包静默安装/卸载冒烟测试（可选）
rem 需要已用 build_installer.bat 生成 Setup.exe
rem 用法：
rem   tests\install_test.bat "release\AI Vision-v2.6-Setup.exe"
rem 不传参数时默认找 ..\release\AI Vision-v2.6-Setup.exe
rem ============================================================

set "SETUP=%~1"
if "%SETUP%"=="" set "SETUP=%~dp0..\release\AI Vision-v2.6-Setup.exe"
if not exist "%SETUP%" (
    echo 找不到安装包：%SETUP%
    echo.
    echo 用法：tests\install_test.bat "路径\AI Vision-v2.6-Setup.exe"
    pause
    exit /b 1
)

set "TARGET=%TEMP%\AIVisionInstallTest"
set "LOG=%TEMP%\aivision_setup.log"

if exist "%TARGET%" rmdir /s /q "%TARGET%"

echo [1/3] 静默安装到 %TARGET% ...
start /wait "" "%SETUP%" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG="%LOG%" /DIR="%TARGET%"
echo       ExitCode=%errorlevel%

if not exist "%TARGET%\AIVision.exe" (
    echo [FAIL] 安装后未找到 AIVision.exe，日志：%LOG%
    pause
    exit /b 1
)
echo [2/3] INSTALL-OK

if exist "%TARGET%\unins000.exe" (
    echo [3/3] 静默卸载 ...
    start /wait "" "%TARGET%\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
    if exist "%TARGET%\AIVision.exe" (
        echo [FAIL] 卸载后 AIVision.exe 仍然存在
        pause
        exit /b 1
    )
    echo UNINSTALL-OK
) else (
    echo [3/3] 未找到 unins000.exe，跳过卸载测试
)

echo.
echo 安装/卸载冒烟测试通过。日志：%LOG%
pause
endlocal
