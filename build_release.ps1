# Build the compact Windows x64 app. Large native runtime/model assets are downloaded after installation.
# No CUDA Toolkit, CMake, Visual Studio compiler or PyTorch is needed for end users.
# Python and PyInstaller are only needed on the developer/build machine.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppVersion = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'VERSION') -Raw).Trim()
if ($AppVersion -notmatch '^\d+\.\d+\.\d+-(alpha|beta|rc)\.\d+$') { throw "Invalid VERSION file value: $AppVersion" }
$DistName = 'PAI-AI-Image-Studio'
$DistDir = Join-Path $ProjectRoot "dist\$DistName"
$LogoIcon = Join-Path $ProjectRoot 'assets\pai_icon.ico'
$LogoPng = Join-Path $ProjectRoot 'assets\pai_logo.png'
$LogoData = "$LogoPng;assets"
$RuntimeDir = Join-Path $DistDir 'runtime'


Write-Host '== P.AI AI IMAGE STUDIO · Version 1 Beta release build =='
# Use the installed system CPython selected by the setup helper; no project-local Python/venv.
$Python = $null
foreach ($Version in @('3.13','3.12','3.11')) {
    $Candidate = (& py ("-" + $Version) -c "import sys,struct; print(sys.executable)" 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $Candidate -and (Test-Path $Candidate)) { $Python = $Candidate.Trim(); break }
}
if (-not $Python) {
    $Candidate = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($Candidate) { $Python = $Candidate }
}
if (-not $Python) { throw 'System Python 3.11, 3.12 or 3.13 x64 not found. Run Install_Update_Requirements.bat first.' }
$PythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { throw 'The selected system Python is not runnable.' }
if ($PythonVersion -notmatch '^3\.(11|12|13)$') { throw "Unsupported system Python $PythonVersion. Install Python 3.11, 3.12 or 3.13 x64." }
$PyArch = & $Python -c "import struct; print(struct.calcsize('P')*8)"
if ($PyArch -ne '64') { throw 'A 64-bit Python installation is required.' }

Push-Location $ProjectRoot
try {
    & $Python -m pip install --user --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'pip/setuptools/wheel upgrade failed.' }
    & $Python -m pip install --user --upgrade -r (Join-Path $ProjectRoot 'requirements-build.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Build dependency installation failed.' }
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { Write-Warning 'pip check found a conflict in the shared system Python environment; checking required P.AI imports separately.' }
    & $Python -c "import PySide6, PIL, certifi, numpy"
    if ($LASTEXITCODE -ne 0) { throw 'One or more required P.AI packages cannot be imported after dependency installation.' }
    & $Python -m compileall -q (Join-Path $ProjectRoot 'pai_ai_image_studio.py') (Join-Path $ProjectRoot 'pai_studio')
    if ($LASTEXITCODE -ne 0) { throw 'Python source syntax check failed.' }
    & $Python -m pytest -q (Join-Path $ProjectRoot 'tests')
    if ($LASTEXITCODE -ne 0) { throw 'Release smoke tests failed.' }

    Remove-Item -Recurse -Force (Join-Path $ProjectRoot 'build'), (Join-Path $ProjectRoot 'dist'), (Join-Path $ProjectRoot 'build-installer'), (Join-Path $ProjectRoot 'dist-installer') -ErrorAction SilentlyContinue
    & $Python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name $DistName `
        --icon $LogoIcon `
        --add-data $LogoData `
        --hidden-import certifi `
        --collect-submodules pai_studio `
        (Join-Path $ProjectRoot 'pai_ai_image_studio.py')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

    if (Test-Path -LiteralPath $RuntimeDir) {
        throw 'The native engine runtime must not be staged inside dist. The installed app downloads and validates it on first model load.'
    }
    # Keep the installed directory lean: include only app/runtime dependencies, branding,
    # and required license notices. README/changelog/version metadata stay in the repository.
    Copy-Item (Join-Path $ProjectRoot 'licenses\THIRD_PARTY_NOTICES.txt') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'LICENSE') -Destination $DistDir -Force

    Write-Host "Compact application staged: $DistDir"
    Write-Host 'The native engine, CUDA companion runtime, and model weights are intentionally not bundled.'
    Write-Host 'Only application files, branding assets, and license notices are copied to the install package.'
    Write-Host 'The installed app downloads and validates the native runtime on first model load; model weights stay managed by Model Catalog.'

    $ISCC = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
    if (-not $ISCC) {
        $PossibleISCC = @(
            (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
            (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
        ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
        if ($PossibleISCC) { $ISCC = @{ Source = $PossibleISCC } }
    }
    if ($ISCC) {
        & $ISCC.Source "/DAppVersion=$AppVersion" (Join-Path $ProjectRoot 'installer.iss')
        if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compilation failed.' }
        Write-Host "Installer output: $(Join-Path $ProjectRoot 'dist-installer')"
    } else {
        Write-Warning 'Inno Setup was not found. The runnable onedir build is ready; install Inno Setup 6 and rerun this script to create P.AI-Setup.exe.'
    }
} finally {
    Pop-Location
}
