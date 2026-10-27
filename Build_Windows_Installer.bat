@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "ROOT=%~dp0"
set "ISCC="
for /f "delims=" %%P in ('where ISCC.exe 2^>nul') do if not defined ISCC set "ISCC=%%P"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
  echo Inno Setup 6 compiler was not found.
  echo Install it from https://jrsoftware.org/isdl.php, then run this file again.
  pause
  exit /b 2
)
echo Building P.AI AI IMAGE STUDIO Windows installer...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%build_release.ps1"
if errorlevel 1 (
  echo Installer build failed. See errors above.
  pause
  exit /b 3
)
if not exist "%ROOT%dist-installer\PAI-AI-Image-Studio-Setup-1.0.0-beta.1.exe" (
  echo The build finished, but the expected Setup executable is missing.
  echo Check build_release.ps1 and the Inno Setup compiler output.
  pause
  exit /b 4
)
echo.
echo Setup created successfully:
echo %ROOT%dist-installer\PAI-AI-Image-Studio-Setup-1.0.0-beta.1.exe
pause
endlocal
