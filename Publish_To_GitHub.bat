@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================================
echo   P.AI AI IMAGE STUDIO — Version 1 Beta (v1.0.0-beta.1) GitHub Sync ^& Upload
echo   Repository: https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO
echo ============================================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/ and run this script again.
    pause
    exit /b 1
)

if not exist ".git" (
    echo [1/4] Initializing local Git repository on branch main...
    git init -b main
) else (
    git branch -M main
)

git config user.name >nul 2>nul
if errorlevel 1 (
    git config user.name "P-AI-Studio"
)
git config user.email >nul 2>nul
if errorlevel 1 (
    git config user.email "P-AI-Studio@users.noreply.github.com"
)

echo [2/4] Staging and committing all project files...
git add -A
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Release P.AI AI IMAGE STUDIO Version 1 Beta (v1.0.0-beta.1)"
)

git remote get-url origin >nul 2>nul
if errorlevel 1 (
    git remote add origin https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO.git
) else (
    git remote set-url origin https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO.git
)

echo [3/4] Pushing full source code and assets to GitHub (main branch)...
git push -u origin main --force
if errorlevel 1 (
    echo.
    echo [WARNING] Git push required authentication or encountered an error.
    echo Please complete the GitHub sign-in window if prompted and run again.
    pause
    exit /b 1
)

echo [4/4] Updating tag v1.0.0-beta.1 on GitHub...
git tag -f v1.0.0-beta.1
git push origin v1.0.0-beta.1 --force

where gh >nul 2>nul
if not errorlevel 1 (
    echo [INFO] GitHub CLI detected — updating Pre-release v1.0.0-beta.1...
    gh release view v1.0.0-beta.1 --repo P-AI-Studio/PAI-AI-IMAGE-STUDIO >nul 2>nul
    if errorlevel 1 (
        gh release create v1.0.0-beta.1 --repo P-AI-Studio/PAI-AI-IMAGE-STUDIO --title "P.AI AI IMAGE STUDIO — Version 1 Beta (v1.0.0-beta.1)" --notes-file "RELEASE_NOTES.md" --prerelease --target main
    ) else (
        gh release edit v1.0.0-beta.1 --repo P-AI-Studio/PAI-AI-IMAGE-STUDIO --title "P.AI AI IMAGE STUDIO — Version 1 Beta (v1.0.0-beta.1)" --notes-file "RELEASE_NOTES.md" --prerelease
    )
)

echo.
echo ============================================================================
echo   SUCCESS! All code, tests, scripts, and assets are pushed to:
echo   https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO
echo   https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO/releases/tag/v1.0.0-beta.1
echo ============================================================================
start "" "https://github.com/P-AI-Studio/PAI-AI-IMAGE-STUDIO"
pause
