@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "ROOT=%~dp0"
set "APP=%ROOT%pai_ai_image_studio.py"
set "PY_EXE="

echo ============================================================
echo P.AI AI IMAGE STUDIO - Version 1 Beta - System Python setup
echo Uses Python installed on Windows and the installed NVIDIA driver/GPU.
echo App settings, models, runtime and outputs remain under this project folder.
echo ============================================================
echo.

if not exist "%APP%" (
  echo ERROR: pai_ai_image_studio.py was not found beside this installer.
  pause
  exit /b 2
)

rem Prefer supported versions discovered through the official Python Launcher.
for %%V in (3.13 3.12 3.11) do if not defined PY_EXE (
  py -%%V -c "import sys,struct; assert sys.version_info >= (3,11) and struct.calcsize('P')*8==64" >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do set "PY_EXE=%%P"
)
if not defined PY_EXE (
  for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY_EXE set "PY_EXE=%%P"
)
if not defined PY_EXE (
  echo ERROR: No system Python 3.11, 3.12 or 3.13 x64 was found.
  echo Install CPython from https://www.python.org/downloads/windows/ and enable the Python Launcher.
  echo This setup intentionally does not install a second Python copy into the project.
  pause
  exit /b 3
)

"%PY_EXE%" -c "import sys,struct; print('System Python:',sys.executable); print('Version:',sys.version); assert sys.version_info >= (3,11) and sys.version_info < (3,14) and struct.calcsize('P')*8==64, 'Need Python 3.11-3.13 x64'"
if errorlevel 1 (
  echo ERROR: Unsupported system Python. Install Python 3.11, 3.12 or 3.13 x64.
  pause
  exit /b 4
)

rem Store the selected interpreter path for diagnostics only; it is not a private runtime.
echo %PY_EXE%>"%ROOT%.system-python-path.txt"
"%PY_EXE%" -m pip --version >nul 2>nul
if errorlevel 1 (
  echo pip is unavailable for the selected system Python. Trying ensurepip...
  "%PY_EXE%" -m ensurepip --upgrade
  if errorlevel 1 (
    echo ERROR: Could not enable pip for the system Python.
    pause
    exit /b 5
  )
)
"%PY_EXE%" -m pip install --user --upgrade pip setuptools wheel
if errorlevel 1 (
  echo ERROR: Could not update pip for the selected system Python.
  pause
  exit /b 6
)
"%PY_EXE%" -m pip install --user --upgrade -r "%ROOT%requirements.txt"
if errorlevel 1 (
  echo ERROR: Requirements installation failed. Review the pip output above.
  pause
  exit /b 7
)
"%PY_EXE%" -m pip check
if errorlevel 1 (
  echo WARNING: pip reports a conflict in the shared system Python environment.
  echo This may come from an unrelated package. Required P.AI imports are checked separately below.
)
"%PY_EXE%" -c "import PySide6, PIL, certifi, numpy"
if errorlevel 1 (
  echo ERROR: One or more required P.AI packages cannot be imported after installation.
  pause
  exit /b 8
)

echo.
echo NVIDIA driver / GPU check:
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
) else if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
  "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" --query-gpu=name,memory.total,driver_version --format=csv
) else (
  echo nvidia-smi was not found. Install/update the NVIDIA display driver if you use an NVIDIA GPU.
)
echo.
echo Setup complete.
echo System Python: %PY_EXE%
echo Python packages: installed into the selected system Python user site.
echo CUDA: the app uses the installed NVIDIA driver/GPU; its native engine retains required CUDA runtime DLLs in data\runtime.
echo A separately installed CUDA Toolkit is not substituted blindly because DLL/API versions must match the native engine.
echo To run the app: Run_PAI_AI_IMAGE_STUDIO.bat
echo.
pause
endlocal
