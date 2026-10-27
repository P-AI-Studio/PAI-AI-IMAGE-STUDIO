# Build a Windows x64 release with the stable-diffusion.cpp runtime and P.AI brand assets.
# No CUDA Toolkit, CMake, Visual Studio compiler or PyTorch is needed for end users.
# Python and PyInstaller are only needed on the developer/build machine.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Get-Sha256Hex([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot calculate SHA-256: file does not exist: $Path"
    }
    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $stream = [System.IO.File]::OpenRead($resolvedPath)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppVersion = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'VERSION') -Raw).Trim()
if ($AppVersion -notmatch '^\d+\.\d+\.\d+-(alpha|beta|rc)\.\d+$DistName = 'PAI-AI-Image-Studio'
$DistDir = Join-Path $ProjectRoot "dist\$DistName"
$LogoIcon = Join-Path $ProjectRoot 'assets\pai_icon.ico'
$LogoPng = Join-Path $ProjectRoot 'assets\pai_logo.png'
$LogoData = "$LogoPng;assets"
$RuntimeDir = Join-Path $DistDir 'runtime'
$CacheRoot = Join-Path $ProjectRoot '.build-cache\stable-diffusion-cpp-master-964-1d0a58a'
$BinaryZip = Join-Path $CacheRoot 'sd-master-1d0a58a-bin-win-cuda12-x64.zip'
$BinaryUrl = 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip'
$BinaryMirrorUrl = 'https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip/download'
$BinarySha = '2f08572e143f0b15a764197ea937d219dd70f0cc7eefb30eb84c9dae6a7cba6d'
$CudaRuntimeZip = Join-Path $CacheRoot 'cudart-sd-bin-win-cu12-x64.zip'
$CudaRuntimeUrl = 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-964-1d0a58a/cudart-sd-bin-win-cu12-x64.zip'
$CudaRuntimeMirrorUrl = 'https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/cudart-sd-bin-win-cu12-x64.zip/download'

function Download-Verified([string[]]$Urls, [string]$Path, [string]$ExpectedSha) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (Test-Path $Path) {
        $existing = (Get-Sha256Hex $Path)
        if ($existing -eq $ExpectedSha) {
            if (-not $script:VerifiedBinarySourceUrl) { $script:VerifiedBinarySourceUrl = 'verified project-local cache (original URL not recorded)' }
            Write-Host "Verified cached asset: $(Split-Path -Leaf $Path)"
            return
        }
        Remove-Item -Force $Path
    }
    $errors = @()
    foreach ($SourceUrl in $Urls) {
        try {
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Host "Downloading official stable-diffusion.cpp runtime asset: $(Split-Path -Leaf $Path)"
            Write-Host "Source: $SourceUrl"
            Invoke-WebRequest -Uri $SourceUrl -OutFile $Path -MaximumRedirection 10 -TimeoutSec 1800
            $actual = (Get-Sha256Hex $Path)
            if ($actual -ne $ExpectedSha) {
                Remove-Item -Force $Path -ErrorAction SilentlyContinue
                throw "SHA-256 mismatch. Expected $ExpectedSha, got $actual."
            }
            $script:VerifiedBinarySourceUrl = $SourceUrl
            return
        } catch {
            $errors += "$SourceUrl -> $($_.Exception.Message)"
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Warning "Download/verification failed; trying next source. $($_.Exception.Message)"
        }
    }
    throw "All native-engine download sources failed: $($errors -join ' | ')"
}

function Test-CudaRuntimeZip([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    $TestRoot = Join-Path $CacheRoot 'cudart-validate'
    Remove-Item -Recurse -Force $TestRoot -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $TestRoot | Out-Null
    try {
        Expand-Archive -LiteralPath $Path -DestinationPath $TestRoot -Force
        $Required = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
        foreach ($Name in $Required) {
            if (-not (Get-ChildItem -Path $TestRoot -Filter $Name -Recurse -File | Select-Object -First 1)) { return $false }
        }
        return $true
    } catch {
        return $false
    } finally {
        Remove-Item -Recurse -Force $TestRoot -ErrorAction SilentlyContinue
    }
}

function Download-CudaRuntime([string[]]$Urls, [string]$Path) {
    if (Test-CudaRuntimeZip $Path) {
        $script:CudaRuntimeSha = (Get-Sha256Hex $Path)
        if (-not $script:CudaRuntimeSourceUrl) { $script:CudaRuntimeSourceUrl = 'cached asset (original URL not recorded)' }
        Write-Host "Verified cached CUDA 12 companion archive (required DLLs present); SHA-256: $script:CudaRuntimeSha"
        return
    }
    $errors = @()
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    foreach ($SourceUrl in $Urls) {
        try {
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Host "Downloading CUDA 12 companion DLL archive: $SourceUrl"
            Invoke-WebRequest -Uri $SourceUrl -OutFile $Path -MaximumRedirection 10 -TimeoutSec 1800
            if (-not (Test-CudaRuntimeZip $Path)) {
                throw 'CUDA runtime companion ZIP is invalid or lacks cudart64_12.dll/cublas64_12.dll/cublasLt64_12.dll.'
            }
            $script:CudaRuntimeSha = (Get-Sha256Hex $Path)
            $script:CudaRuntimeSourceUrl = $SourceUrl
            Write-Host "CUDA companion SHA-256 for build record: $script:CudaRuntimeSha"
            Write-Warning 'The upstream engine.json pins the main CUDA binary, but does not publish a separate cudart ZIP SHA-256; the companion is pinned by the same fixed release tag and validated by ZIP CRC + required DLL names.'
            return
        } catch {
            $errors += "$SourceUrl -> $($_.Exception.Message)"
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Warning "CUDA runtime download/verification failed; trying next source. $($_.Exception.Message)"
        }
    }
    throw "All CUDA runtime download sources failed: $($errors -join ' | ')"
}

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

    Download-Verified @($BinaryUrl, $BinaryMirrorUrl) $BinaryZip $BinarySha

    $Scratch = Join-Path $CacheRoot 'extract'
    $BinaryExtract = Join-Path $Scratch 'binary'
    Remove-Item -Recurse -Force $Scratch -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $BinaryExtract | Out-Null
    Expand-Archive -LiteralPath $BinaryZip -DestinationPath $BinaryExtract -Force

    $Server = Get-ChildItem -Path $BinaryExtract -Filter 'sd-server.exe' -Recurse -File | Select-Object -First 1
    if (-not $Server) { throw 'sd-server.exe was not found in the pinned upstream archive.' }
    $CudaBackend = Get-ChildItem -Path $BinaryExtract -Filter 'ggml-cuda.dll' -Recurse -File | Select-Object -First 1
    if (-not $CudaBackend) { throw 'Pinned binary does not contain ggml-cuda.dll; refusing to package a CPU-only runtime.' }

    # The upstream Windows CUDA binary archive intentionally ships without cudart/cublas DLLs.
    # Those are distributed as a separate companion ZIP at the same pinned release tag.
    Download-CudaRuntime @($CudaRuntimeUrl, $CudaRuntimeMirrorUrl) $CudaRuntimeZip
    $CudaStage = Join-Path $CacheRoot 'cudart-stage'
    Remove-Item -Recurse -Force $CudaStage -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $CudaStage | Out-Null
    Expand-Archive -LiteralPath $CudaRuntimeZip -DestinationPath $CudaStage -Force
    $CudaDllDir = Join-Path $BinaryExtract 'cudart'
    New-Item -ItemType Directory -Force -Path $CudaDllDir | Out-Null
    foreach ($DllName in @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')) {
        $Dll = Get-ChildItem -Path $CudaStage -Filter $DllName -Recurse -File | Select-Object -First 1
        if (-not $Dll) { throw "Required CUDA runtime DLL missing: $DllName" }
        Copy-Item -LiteralPath $Dll.FullName -Destination (Join-Path $CudaDllDir $DllName) -Force
    }
    $RuntimeManifest = @{
        source_url = $script:CudaRuntimeSourceUrl
        release_tag = 'master-964-1d0a58a'
        main_binary_source_url = $script:VerifiedBinarySourceUrl
        archive_sha256_computed = $script:CudaRuntimeSha
        upstream_sha256_pinned = $false
        required_dlls = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
    } | ConvertTo-Json -Depth 4
    Set-Content -Path (Join-Path $CudaDllDir 'runtime-manifest.json') -Value $RuntimeManifest -Encoding UTF8
    Remove-Item -Recurse -Force $CudaStage -ErrorAction SilentlyContinue

    Remove-Item -Recurse -Force (Join-Path $ProjectRoot 'build'), (Join-Path $ProjectRoot 'dist'), (Join-Path $ProjectRoot 'build-installer'), (Join-Path $ProjectRoot 'dist-installer') -ErrorAction SilentlyContinue
    & $Python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name $DistName `
        --icon $LogoIcon `
        --add-data $LogoData `
        --hidden-import certifi `
        --collect-submodules pai_studio `
        (Join-Path $ProjectRoot 'pai_ai_image_studio.py')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    # Copy the whole official runtime tree, not only the folder containing sd-server.exe:
    # some release assets place native DLLs in sibling subfolders.
    Copy-Item -Path (Join-Path $BinaryExtract '*') -Destination $RuntimeDir -Recurse -Force
    # Bundle the main CUDA engine plus its separate CUDA 12 runtime dependency DLLs.
    $UpstreamNotices = Get-ChildItem -Path $BinaryExtract -Recurse -File | Where-Object { $_.Name -match '^(LICENSE|NOTICE|COPYING|README)(\.|$)' }
    $NoticeDir = Join-Path $RuntimeDir 'upstream-notices'
    $NativeNoticeDir = Join-Path $NoticeDir 'stable-diffusion.cpp'
    New-Item -ItemType Directory -Force -Path $NativeNoticeDir | Out-Null
    foreach ($Notice in $UpstreamNotices) { Copy-Item -LiteralPath $Notice.FullName -Destination $NativeNoticeDir -Force }

    Copy-Item (Join-Path $ProjectRoot 'licenses\THIRD_PARTY_NOTICES.txt') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'LICENSE') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'VERSION') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'CHANGELOG.md') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'README.md') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'README_FA.md') -Destination $DistDir -Force

    $ExpectedServer = Get-ChildItem -Path $RuntimeDir -Filter 'sd-server.exe' -Recurse -File | Select-Object -First 1
    if (-not $ExpectedServer) { throw 'Bundled runtime staging failed: runtime\sd-server.exe is missing.' }
    $RuntimeRoot = Join-Path $DistDir 'runtime'
    foreach ($RequiredName in @('ggml-cuda.dll', 'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')) {
        if (-not (Get-ChildItem -Path $RuntimeRoot -Filter $RequiredName -Recurse -File | Select-Object -First 1)) {
            throw "CUDA runtime validation failed: missing $RequiredName in staged runtime."
        }
    }
    $RuntimeDllDirs = (Get-ChildItem -Path $RuntimeRoot -Filter '*.dll' -Recurse -File | ForEach-Object { Split-Path -Parent $_.FullName } | Sort-Object -Unique) -join ';'
    $OldPath = $env:PATH
    try {
        $env:PATH = $RuntimeDllDirs + ';' + $env:PATH
        $RuntimeHelp = (& $ExpectedServer.FullName --help 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0 -or $RuntimeHelp -notmatch '--diffusion-model' -or $RuntimeHelp -notmatch '--backend') {
            throw "The staged native engine failed its --help/backend smoke test. Output:`n$RuntimeHelp"
        }
    } finally { $env:PATH = $OldPath }
    Write-Host "Native engine startup and CUDA dependency file smoke tests passed."
    Write-Host "Staged application: $DistDir"
    Write-Host 'Run the app on a clean Windows machine before public distribution; verify the runtime loads and complete a real generation/edit test.'

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
) { throw "Invalid VERSION file value: $AppVersion" }
$DistName = 'PAI-AI-Image-Studio'
$DistDir = Join-Path $ProjectRoot "dist\$DistName"
$LogoIcon = Join-Path $ProjectRoot 'assets\pai_icon.ico'
$LogoPng = Join-Path $ProjectRoot 'assets\pai_logo.png'
$LogoData = "$LogoPng;assets"
$RuntimeDir = Join-Path $DistDir 'runtime'
$CacheRoot = Join-Path $ProjectRoot '.build-cache\stable-diffusion-cpp-master-964-1d0a58a'
$BinaryZip = Join-Path $CacheRoot 'sd-master-1d0a58a-bin-win-cuda12-x64.zip'
$BinaryUrl = 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip'
$BinaryMirrorUrl = 'https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip/download'
$BinarySha = '2f08572e143f0b15a764197ea937d219dd70f0cc7eefb30eb84c9dae6a7cba6d'
$CudaRuntimeZip = Join-Path $CacheRoot 'cudart-sd-bin-win-cu12-x64.zip'
$CudaRuntimeUrl = 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-964-1d0a58a/cudart-sd-bin-win-cu12-x64.zip'
$CudaRuntimeMirrorUrl = 'https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/cudart-sd-bin-win-cu12-x64.zip/download'

function Download-Verified([string[]]$Urls, [string]$Path, [string]$ExpectedSha) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (Test-Path $Path) {
        $existing = (Get-Sha256Hex $Path)
        if ($existing -eq $ExpectedSha) {
            if (-not $script:VerifiedBinarySourceUrl) { $script:VerifiedBinarySourceUrl = 'verified project-local cache (original URL not recorded)' }
            Write-Host "Verified cached asset: $(Split-Path -Leaf $Path)"
            return
        }
        Remove-Item -Force $Path
    }
    $errors = @()
    foreach ($SourceUrl in $Urls) {
        try {
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Host "Downloading official stable-diffusion.cpp runtime asset: $(Split-Path -Leaf $Path)"
            Write-Host "Source: $SourceUrl"
            Invoke-WebRequest -Uri $SourceUrl -OutFile $Path -MaximumRedirection 10 -TimeoutSec 1800
            $actual = (Get-Sha256Hex $Path)
            if ($actual -ne $ExpectedSha) {
                Remove-Item -Force $Path -ErrorAction SilentlyContinue
                throw "SHA-256 mismatch. Expected $ExpectedSha, got $actual."
            }
            $script:VerifiedBinarySourceUrl = $SourceUrl
            return
        } catch {
            $errors += "$SourceUrl -> $($_.Exception.Message)"
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Warning "Download/verification failed; trying next source. $($_.Exception.Message)"
        }
    }
    throw "All native-engine download sources failed: $($errors -join ' | ')"
}

function Test-CudaRuntimeZip([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    $TestRoot = Join-Path $CacheRoot 'cudart-validate'
    Remove-Item -Recurse -Force $TestRoot -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $TestRoot | Out-Null
    try {
        Expand-Archive -LiteralPath $Path -DestinationPath $TestRoot -Force
        $Required = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
        foreach ($Name in $Required) {
            if (-not (Get-ChildItem -Path $TestRoot -Filter $Name -Recurse -File | Select-Object -First 1)) { return $false }
        }
        return $true
    } catch {
        return $false
    } finally {
        Remove-Item -Recurse -Force $TestRoot -ErrorAction SilentlyContinue
    }
}

function Download-CudaRuntime([string[]]$Urls, [string]$Path) {
    if (Test-CudaRuntimeZip $Path) {
        $script:CudaRuntimeSha = (Get-Sha256Hex $Path)
        if (-not $script:CudaRuntimeSourceUrl) { $script:CudaRuntimeSourceUrl = 'cached asset (original URL not recorded)' }
        Write-Host "Verified cached CUDA 12 companion archive (required DLLs present); SHA-256: $script:CudaRuntimeSha"
        return
    }
    $errors = @()
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    foreach ($SourceUrl in $Urls) {
        try {
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Host "Downloading CUDA 12 companion DLL archive: $SourceUrl"
            Invoke-WebRequest -Uri $SourceUrl -OutFile $Path -MaximumRedirection 10 -TimeoutSec 1800
            if (-not (Test-CudaRuntimeZip $Path)) {
                throw 'CUDA runtime companion ZIP is invalid or lacks cudart64_12.dll/cublas64_12.dll/cublasLt64_12.dll.'
            }
            $script:CudaRuntimeSha = (Get-Sha256Hex $Path)
            $script:CudaRuntimeSourceUrl = $SourceUrl
            Write-Host "CUDA companion SHA-256 for build record: $script:CudaRuntimeSha"
            Write-Warning 'The upstream engine.json pins the main CUDA binary, but does not publish a separate cudart ZIP SHA-256; the companion is pinned by the same fixed release tag and validated by ZIP CRC + required DLL names.'
            return
        } catch {
            $errors += "$SourceUrl -> $($_.Exception.Message)"
            Remove-Item -Force $Path -ErrorAction SilentlyContinue
            Write-Warning "CUDA runtime download/verification failed; trying next source. $($_.Exception.Message)"
        }
    }
    throw "All CUDA runtime download sources failed: $($errors -join ' | ')"
}

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

    Download-Verified @($BinaryUrl, $BinaryMirrorUrl) $BinaryZip $BinarySha

    $Scratch = Join-Path $CacheRoot 'extract'
    $BinaryExtract = Join-Path $Scratch 'binary'
    Remove-Item -Recurse -Force $Scratch -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $BinaryExtract | Out-Null
    Expand-Archive -LiteralPath $BinaryZip -DestinationPath $BinaryExtract -Force

    $Server = Get-ChildItem -Path $BinaryExtract -Filter 'sd-server.exe' -Recurse -File | Select-Object -First 1
    if (-not $Server) { throw 'sd-server.exe was not found in the pinned upstream archive.' }
    $CudaBackend = Get-ChildItem -Path $BinaryExtract -Filter 'ggml-cuda.dll' -Recurse -File | Select-Object -First 1
    if (-not $CudaBackend) { throw 'Pinned binary does not contain ggml-cuda.dll; refusing to package a CPU-only runtime.' }

    # The upstream Windows CUDA binary archive intentionally ships without cudart/cublas DLLs.
    # Those are distributed as a separate companion ZIP at the same pinned release tag.
    Download-CudaRuntime @($CudaRuntimeUrl, $CudaRuntimeMirrorUrl) $CudaRuntimeZip
    $CudaStage = Join-Path $CacheRoot 'cudart-stage'
    Remove-Item -Recurse -Force $CudaStage -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $CudaStage | Out-Null
    Expand-Archive -LiteralPath $CudaRuntimeZip -DestinationPath $CudaStage -Force
    $CudaDllDir = Join-Path $BinaryExtract 'cudart'
    New-Item -ItemType Directory -Force -Path $CudaDllDir | Out-Null
    foreach ($DllName in @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')) {
        $Dll = Get-ChildItem -Path $CudaStage -Filter $DllName -Recurse -File | Select-Object -First 1
        if (-not $Dll) { throw "Required CUDA runtime DLL missing: $DllName" }
        Copy-Item -LiteralPath $Dll.FullName -Destination (Join-Path $CudaDllDir $DllName) -Force
    }
    $RuntimeManifest = @{
        source_url = $script:CudaRuntimeSourceUrl
        release_tag = 'master-964-1d0a58a'
        main_binary_source_url = $script:VerifiedBinarySourceUrl
        archive_sha256_computed = $script:CudaRuntimeSha
        upstream_sha256_pinned = $false
        required_dlls = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
    } | ConvertTo-Json -Depth 4
    Set-Content -Path (Join-Path $CudaDllDir 'runtime-manifest.json') -Value $RuntimeManifest -Encoding UTF8
    Remove-Item -Recurse -Force $CudaStage -ErrorAction SilentlyContinue

    Remove-Item -Recurse -Force (Join-Path $ProjectRoot 'build'), (Join-Path $ProjectRoot 'dist'), (Join-Path $ProjectRoot 'build-installer'), (Join-Path $ProjectRoot 'dist-installer') -ErrorAction SilentlyContinue
    & $Python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name $DistName `
        --icon $LogoIcon `
        --add-data $LogoData `
        --hidden-import certifi `
        --collect-submodules pai_studio `
        (Join-Path $ProjectRoot 'pai_ai_image_studio.py')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    # Copy the whole official runtime tree, not only the folder containing sd-server.exe:
    # some release assets place native DLLs in sibling subfolders.
    Copy-Item -Path (Join-Path $BinaryExtract '*') -Destination $RuntimeDir -Recurse -Force
    # Bundle the main CUDA engine plus its separate CUDA 12 runtime dependency DLLs.
    $UpstreamNotices = Get-ChildItem -Path $BinaryExtract -Recurse -File | Where-Object { $_.Name -match '^(LICENSE|NOTICE|COPYING|README)(\.|$)' }
    $NoticeDir = Join-Path $RuntimeDir 'upstream-notices'
    $NativeNoticeDir = Join-Path $NoticeDir 'stable-diffusion.cpp'
    New-Item -ItemType Directory -Force -Path $NativeNoticeDir | Out-Null
    foreach ($Notice in $UpstreamNotices) { Copy-Item -LiteralPath $Notice.FullName -Destination $NativeNoticeDir -Force }

    Copy-Item (Join-Path $ProjectRoot 'licenses\THIRD_PARTY_NOTICES.txt') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'LICENSE') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'VERSION') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'CHANGELOG.md') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'README.md') -Destination $DistDir -Force
    Copy-Item (Join-Path $ProjectRoot 'README_FA.md') -Destination $DistDir -Force

    $ExpectedServer = Get-ChildItem -Path $RuntimeDir -Filter 'sd-server.exe' -Recurse -File | Select-Object -First 1
    if (-not $ExpectedServer) { throw 'Bundled runtime staging failed: runtime\sd-server.exe is missing.' }
    $RuntimeRoot = Join-Path $DistDir 'runtime'
    foreach ($RequiredName in @('ggml-cuda.dll', 'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')) {
        if (-not (Get-ChildItem -Path $RuntimeRoot -Filter $RequiredName -Recurse -File | Select-Object -First 1)) {
            throw "CUDA runtime validation failed: missing $RequiredName in staged runtime."
        }
    }
    $RuntimeDllDirs = (Get-ChildItem -Path $RuntimeRoot -Filter '*.dll' -Recurse -File | ForEach-Object { Split-Path -Parent $_.FullName } | Sort-Object -Unique) -join ';'
    $OldPath = $env:PATH
    try {
        $env:PATH = $RuntimeDllDirs + ';' + $env:PATH
        $RuntimeHelp = (& $ExpectedServer.FullName --help 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0 -or $RuntimeHelp -notmatch '--diffusion-model' -or $RuntimeHelp -notmatch '--backend') {
            throw "The staged native engine failed its --help/backend smoke test. Output:`n$RuntimeHelp"
        }
    } finally { $env:PATH = $OldPath }
    Write-Host "Native engine startup and CUDA dependency file smoke tests passed."
    Write-Host "Staged application: $DistDir"
    Write-Host 'Run the app on a clean Windows machine before public distribution; verify the runtime loads and complete a real generation/edit test.'

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
