from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_builder_hashes_with_dotnet_and_not_get_file_hash():
    script = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert "function Get-Sha256Hex([string]$Path)" in script
    assert "[System.Security.Cryptography.SHA256]::Create()" in script
    assert "Get-Sha256Hex $Path" in script
    assert "Get-FileHash" not in script


def test_windows_release_workflow_builds_directly_and_runs_on_pull_requests():
    workflow = (ROOT / ".github/workflows/windows-release.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "run: .\\build_release.ps1" in workflow
    assert "run: .\\Build_Windows_Installer.bat" not in workflow
    assert "SHA-256: $digest" in workflow


def test_publish_script_never_force_pushes_or_rewrites_existing_release_tags():
    script = (ROOT / "Publish_To_GitHub.bat").read_text(encoding="utf-8")
    assert "--force" not in script.lower()
    assert "git tag -f" not in script.lower()
    assert "RELEASE_NOTES.md" not in script
    assert 'if not exist ".git"' in script
    assert "git push -u origin main" not in script
    assert "Remote tag %TAG% already exists" in script
    assert "git status --porcelain" in script
    assert "git rev-parse origin/main" in script


def test_installer_filename_uses_the_version_file():
    batch = (ROOT / "Build_Windows_Installer.bat").read_text(encoding="utf-8")
    iss = (ROOT / "installer.iss").read_text(encoding="utf-8")
    builder = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert 'set /p VERSION=<"%ROOT%VERSION"' in batch
    assert "Setup-%VERSION%.exe" in batch
    assert "OutputBaseFilename=PAI-AI-Image-Studio-Setup-{#AppVersion}" in iss
    assert "Get-Content -LiteralPath (Join-Path $ProjectRoot 'VERSION')" in builder
