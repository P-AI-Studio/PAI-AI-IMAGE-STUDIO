@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "ROOT=%~dp0"
set "APP=%ROOT%pai_ai_image_studio.py"
set "PY_EXE="
if not exist "%APP%" (
  echo ERROR: pai_ai_image_studio.py was not found beside this launcher.
  pause
  exit /b 2
)
rem Prefer an installed system CPython; do not create or use a project-local Python/venv.
for %%V in (3.13 3.12 3.11) do if not defined PY_EXE (
  py -%%V -c "import sys,struct; assert sys.version_info >= (3,11) and struct.calcsize('P')*8==64" >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do set "PY_EXE=%%P"
)
if not defined PY_EXE (
  for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY_EXE set "PY_EXE=%%P"
)
if not defined PY_EXE (
  echo No compatible system Python 3.11, 3.12 or 3.13 x64 was found.
  echo Install Python from python.org and enable the Python Launcher, then run Install_Update_Requirements.bat.
  pause
  exit /b 3
)
"%PY_EXE%" -c "import sys,struct; print('Using system Python:',sys.executable); print('Version:',sys.version); assert sys.version_info >= (3,11) and sys.version_info < (3,14) and struct.calcsize('P')*8==64, 'Need Python 3.11-3.13 x64'"
if errorlevel 1 (
  echo Unsupported system Python. Install Python 3.11, 3.12 or 3.13 x64.
  pause
  exit /b 4
)
"%PY_EXE%" -c "import PySide6, PIL" >nul 2>nul
if errorlevel 1 (
  echo Required Python packages are missing. Running setup with the selected system Python...
  call "%ROOT%Install_Update_Requirements.bat"
  if errorlevel 1 exit /b 5
)
pushd "%ROOT%"
"%PY_EXE%" "%APP%" %*
set "APP_EXIT=%ERRORLEVEL%"
popd
if not "%APP_EXIT%"=="0" (
  echo.
  echo P.AI exited with code %APP_EXIT%. Review the Activity Log and setup/runtime notes.
  pause
)
exit /b %APP_EXIT%
