@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1"
if errorlevel 1 (
  echo.
  echo Build failed. Review the PowerShell error above.
  pause
  exit /b 1
)
echo.
echo Build script completed. Review README_FA.md and run clean-Windows tests before publishing.
pause
endlocal
