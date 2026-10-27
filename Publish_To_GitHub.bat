@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================================
echo   P.AI AI IMAGE STUDIO — Safe Release Tag Publisher
echo   Repository: https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO
echo ============================================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

if not exist ".git" (
    echo [ERROR] This folder is not a Git clone.
    echo Clone the repository first; do not publish from a downloaded ZIP.
    pause
    exit /b 2
)

for /f "delims=" %%B in ('git branch --show-current') do set "BRANCH=%%B"
if /I not "%BRANCH%"=="main" (
    echo [ERROR] Current branch is "%BRANCH%", not main.
    echo Switch to main after the release fix has been reviewed and merged.
    pause
    exit /b 3
)

echo [1/4] Fetching current main branch from GitHub...
git fetch origin main
if errorlevel 1 (
    echo [ERROR] Could not fetch origin/main.
    pause
    exit /b 4
)

for /f "delims=" %%V in (VERSION) do set "VERSION=%%V"
for /f "delims=" %%V in ('git show origin/main:VERSION') do set "REMOTE_VERSION=%%V"
if not defined VERSION (
    echo [ERROR] VERSION file is empty or missing.
    pause
    exit /b 5
)
if not "%VERSION%"=="%REMOTE_VERSION%" (
    echo [ERROR] Local VERSION does not match origin/main.
    echo Only tag a version that is already committed on the remote main branch.
    pause
    exit /b 6
)

for /f "delims=" %%H in ('git rev-parse HEAD') do set "LOCAL_HEAD=%%H"
for /f "delims=" %%H in ('git rev-parse origin/main') do set "REMOTE_HEAD=%%H"
if not "%LOCAL_HEAD%"=="%REMOTE_HEAD%" (
    echo [ERROR] Local main is not exactly equal to origin/main.
    echo Pull the reviewed, merged main branch and run again.
    pause
    exit /b 7
)

for /f "delims=" %%S in ('git status --porcelain') do set "DIRTY=1"
if defined DIRTY (
    echo [ERROR] Working tree has uncommitted or untracked files.
    echo Commit changes through a reviewed pull request before publishing a tag.
    pause
    exit /b 8
)

set "TAG=v%VERSION%"
echo [2/4] Checking whether tag %TAG% already exists remotely...
for /f "delims=" %%T in ('git ls-remote --tags origin "refs/tags/%TAG%"') do set "REMOTE_TAG=%%T"
if defined REMOTE_TAG (
    echo [ERROR] Remote tag %TAG% already exists. Published tags are never overwritten.
    pause
    exit /b 9
)

echo [3/4] Creating or verifying a local annotated tag at reviewed main...
git rev-parse --verify "refs/tags/%TAG%" >nul 2>nul
if errorlevel 1 (
    git tag -a "%TAG%" -m "P.AI AI IMAGE STUDIO %VERSION%"
    if errorlevel 1 (
        echo [ERROR] Could not create local tag %TAG%.
        pause
        exit /b 10
    )
) else (
    for /f "delims=" %%H in ('git rev-parse "%TAG%^{commit}"') do set "TAG_TARGET=%%H"
    if not "%TAG_TARGET%"=="%REMOTE_HEAD%" (
        echo [ERROR] Local tag %TAG% points to a different commit. It was not changed.
        pause
        exit /b 11
    )
)

echo [4/4] Pushing new tag %TAG% without force...
git push origin "%TAG%"
if errorlevel 1 (
    echo [ERROR] Tag push failed. No force push or tag rewrite was attempted.
    pause
    exit /b 12
)

echo.
echo Tag pushed. GitHub Actions will build and attach the Windows installer if the build succeeds:
echo https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO/actions
start "" "https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO/actions"
pause
endlocal
