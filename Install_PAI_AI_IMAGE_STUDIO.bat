@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "SOURCE=%~dp0"
set "INSTALL=%LOCALAPPDATA%\Programs\P.AI\AI Image Studio"
set "PY_EXE="
echo ============================================================
echo P.AI AI IMAGE STUDIO - Source Mode Installer
echo Installs app files locally and uses your existing system Python.
echo Model weights are not included in this source installer.
echo ============================================================
echo.
if /I "%SOURCE%"=="%INSTALL%\" (
  echo This installer is already running from the target folder.
  echo Extract the source archive to a temporary folder and run it there.
  pause
  exit /b 2
)
for %%V in (3.13 3.12 3.11) do if not defined PY_EXE (
  py -%%V -c "import sys,struct; assert sys.version_info >= (3,11) and sys.version_info < (3,14) and struct.calcsize('P')*8==64" >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do set "PY_EXE=%%P"
)
if not defined PY_EXE (
  echo Compatible system Python 3.11, 3.12 or 3.13 x64 was not found.
  echo Install Python from https://www.python.org/downloads/windows/ and enable the Python Launcher.
  pause
  exit /b 3
)
echo Using: %PY_EXE%
echo Installing application files to:
echo %INSTALL%
echo.
if not exist "%INSTALL%" mkdir "%INSTALL%"
if errorlevel 1 exit /b 4
robocopy "%SOURCE%" "%INSTALL%" /E /R:2 /W:1 /XD .git .github tests docs .build-cache build dist dist-installer .pytest_cache __pycache__ data cache models loras runtime images /XF *.pyc *.log settings.json .system-python-path.txt
if errorlevel 8 (
  echo Failed to copy application files.
  pause
  exit /b 5
)
echo %PY_EXE%>"%INSTALL%\.system-python-path.txt"
call "%INSTALL%Install_Update_Requirements.bat"
if errorlevel 1 (
  echo Dependency installation failed. App files remain installed so setup can be retried.
  pause
  exit /b 6
)
set "PAI_INSTALL=%INSTALL%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$install=$env:PAI_INSTALL; $w=New-Object -ComObject WScript.Shell; $dir=[Environment]::GetFolderPath('Programs'); $p=Join-Path $dir 'P.AI AI IMAGE STUDIO.lnk'; $s=$w.CreateShortcut($p); $s.TargetPath=Join-Path $install 'Run_PAI_AI_IMAGE_STUDIO.bat'; $s.WorkingDirectory=$install; $s.IconLocation=Join-Path $install 'assets\pai_icon.ico'; $s.Save(); $d=[Environment]::GetFolderPath('Desktop'); $q=Join-Path $d 'P.AI AI IMAGE STUDIO.lnk'; $t=$w.CreateShortcut($q); $t.TargetPath=Join-Path $install 'Run_PAI_AI_IMAGE_STUDIO.bat'; $t.WorkingDirectory=$install; $t.IconLocation=Join-Path $install 'assets\pai_icon.ico'; $t.Save()"
if errorlevel 1 echo WARNING: Shortcuts could not be created automatically; launch from Run_PAI_AI_IMAGE_STUDIO.bat.
echo.
echo Source-mode install completed.
echo App folder: %INSTALL%
echo Data/models/cache/outputs will live in the app folder's data directory.
echo Launch from Start Menu or Run_PAI_AI_IMAGE_STUDIO.bat.
echo.
pause
endlocal
