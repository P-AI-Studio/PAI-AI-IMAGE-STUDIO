@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================================
echo   P.AI AI IMAGE STUDIO — Safe GitHub Sync and Release Trigger
echo   Repository: https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO
echo ============================================================================

echo.
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Install Git from https://git-scm.com/ and try again.
    pause
    exit /b 1
)

if not exist ".git" (
    echo [ERROR] This directory is not a Git clone.
    echo Do not initialize an empty repository from the release ZIP.
    echo First clone https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO.git,
    echo copy your intended changes into that clone, then run this script there.
    pause
    exit /b 2
)

for /f "delims=" %%B in ('git branch --show-current') do set "BRANCH=%%B"
if /I not "%BRANCH%"=="main" (
    echo [ERROR] Current branch is "%BRANCH%", not main.
    echo Switch to main and review or commit your changes before syncing.
    pause
    exit /b 3
)

git remote get-url origin >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Remote origin is missing. Add the expected GitHub repository first.
    pause
    exit /b 4
)
for /f "delims=" %%V in (VERSION) do set "VERSION=%%V"
if not defined VERSION (
    echo [ERROR] VERSION file is empty or missing.
    pause
    exit /b 5
)
set "TAG=v%VERSION%"

echo [1/5] Staging current changes...
git add -A
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Release P.AI AI IMAGE STUDIO %VERSION%"
    if errorlevel 1 (
        echo [ERROR] Commit failed. Review Git output and try again.
        pause
        exit /b 6
    )
) else (
    echo No new tracked changes to commit.
)

echo [2/5] Fetching origin and replaying local commits safely...
git fetch origin
if errorlevel 1 (
    echo [ERROR] Could not fetch origin. No force push was attempted.
    pause
    exit /b 7
)
git pull --rebase origin main
if errorlevel 1 (
    echo [ERROR] Rebase could not finish. Resolve conflicts manually, then rerun this script.
    pause
    exit /b 8
)

echo [3/5] Pushing main without force...
git push -u origin main
if errorlevel 1 (
    echo [ERROR] Push was rejected. Review branch divergence and permissions; history was not overwritten.
    pause
    exit /b 9
)

echo [4/5] Checking release tag %TAG%...
git ls-remote --exit-code --tags origin "refs/tags/%TAG%" >nul 2>nul
if not errorlevel 1 (
    echo [ERROR] Tag %TAG% already exists on GitHub and will not be overwritten.
    echo To publish a new build, update VERSION to a new beta version and commit that change.
    echo The Windows release workflow publishes the installer when a new version tag is pushed.
    pause
    exit /b 10
)
git tag -a "%TAG%" -m "P.AI AI IMAGE STUDIO %VERSION%"
if errorlevel 1 (
    echo [ERROR] Could not create tag %TAG%. The existing tags were not changed.
    pause
    exit /b 11
)

echo [5/5] Pushing new tag %TAG% to start the Windows installer workflow...
git push origin "%TAG%"
if errorlevel 1 (
    echo [ERROR] Tag push failed. The tag was not force-pushed.
    pause
    exit /b 12
)

echo.
echo Sync completed. GitHub Actions will build and attach the Windows installer if the workflow succeeds:
echo https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO/actions
start "" "https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO/actions"
pause
endlocal
