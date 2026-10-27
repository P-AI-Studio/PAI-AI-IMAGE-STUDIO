from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_builder_keeps_large_native_runtime_out_of_installer():
    script = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    engine = (ROOT / "pai_studio" / "engine.py").read_text(encoding="utf-8")

    # The build machine must not download or stage the hundreds-of-MB native
    # runtime. A clean dist directory is the only input to Inno Setup.
    assert "Download-Verified" not in script
    assert "Download-CudaRuntime" not in script
    assert "$BinaryExtract" not in script
    assert "sd-server.exe" not in script
    assert "Copy-Item -Path (Join-Path $BinaryExtract '*')" not in script
    assert "if (Test-Path -LiteralPath $RuntimeDir)" in script

    # The installed app still performs the pinned download, size/hash checks,
    # safe extraction and install-time progress reporting.
    assert 'runtime_sha256 = "2f08572e143f0b15a764197ea937d219dd70f0cc7eefb30eb84c9dae6a7cba6d"' in engine
    assert "runtime_size = 338_016_915" in engine
    assert "actual != runtime_sha256" in engine
    assert 'DATA_ROOT / "runtime" / runtime_tag' in engine
    assert "Downloading sd-master-1d0a58a-bin-win-cuda12-x64.zip" in engine



def test_missing_models_open_model_catalog_before_native_runtime_download():
    window = (ROOT / "pai_studio" / "ui" / "main_window.py").read_text(encoding="utf-8")
    engine = (ROOT / "pai_studio" / "engine.py").read_text(encoding="utf-8")
    method_start = window.index("    def load_model_async(self):")
    method_end = window.index("    def _model_loaded(self", method_start)
    load_method = window[method_start:method_end]
    ensure_start = engine.index("    def ensure_loaded(self):")
    ensure_end = engine.index("    @staticmethod", ensure_start)
    ensure_method = engine[ensure_start:ensure_end]

    assert load_method.index("catalog._profile_has_verified_catalog_assets()") < load_method.index("self._configure_engine_from_ui()")
    assert "self.download_model_bundle()" in load_method
    assert load_method.index("if self.download_model_bundle():") < load_method.index("self.load_model_async()")
    assert "The native engine download will start only after the required model files are ready." in load_method
    assert ensure_method.index("parts = self._find_model_parts(model_dir)") < ensure_method.index("executable = self._runtime_executable()")


def test_windows_release_workflow_builds_directly_and_runs_on_pull_requests():
    workflow = (ROOT / ".github/workflows/windows-release.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "run: .\\build_release.ps1" in workflow
    assert "run: .\\Build_Windows_Installer.bat" not in workflow
    assert "SHA-256: $digest" in workflow


def test_release_publishing_uses_version_tags_and_keeps_build_output_out_of_git():
    workflow = (ROOT / ".github/workflows/windows-release.yml").read_text(encoding="utf-8")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert 'push:' in workflow and 'tags:' in workflow and '"v*"' in workflow
    assert "softprops/action-gh-release@v2" in workflow
    assert "files: dist-installer/PAI-AI-Image-Studio-Setup-*.exe" in workflow
    for pattern in ("build/", "dist/", "dist-installer/", ".build-cache/"):
        assert pattern in ignore
    assert not (ROOT / "Publish_To_GitHub.bat").exists()


def test_installer_filename_uses_the_version_file():
    batch = (ROOT / "Build_Windows_Installer.bat").read_text(encoding="utf-8")
    iss = (ROOT / "installer.iss").read_text(encoding="utf-8")
    builder = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert 'set /p VERSION=<"%ROOT%VERSION"' in batch
    assert "Setup-%VERSION%.exe" in batch
    assert "OutputBaseFilename=PAI-AI-Image-Studio-Setup-{#AppVersion}" in iss
    assert "Get-Content -LiteralPath (Join-Path $ProjectRoot 'VERSION')" in builder
