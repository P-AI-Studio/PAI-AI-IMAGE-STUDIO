# Contributing

Thank you for helping improve P.AI AI Image Studio. This is a beta application with local image inference and a modular UI. Distinguish automated source tests from actual GPU/image-quality validation.

## Before a pull request

1. For significant behavior changes, open an issue first and describe the user-facing problem.
2. Keep application-owned UI and workflow logic inside `pai_studio/`; document the runtime/model interfaces separately from UI behavior.
3. Validate external files, paths, URLs, prompts, model roles, and numeric parameters through the existing validation layer.
4. Do not add cloud image-generation calls or transmit user images/prompts off-device as a default behavior. Any opt-in network feature must be disclosed and reviewed before merging.
5. Do not commit personal images, generated images, model weights, runtime binaries, logs, settings, tokens, credentials, `.build-cache/`, `data/`, or build outputs.
6. Update English and Persian documentation when setup, model catalog, workflows, privacy behavior, or release requirements change.
7. Add regression tests for bugs and externally visible behavior.

## Local checks

```powershell
python -m pip install -r requirements-test.txt
python -m compileall -q pai_ai_image_studio.py pai_studio tests
python -m pytest -q
```

## Hardware checks

When a change affects CUDA, VRAM placement, the native runtime, model formats, or image composition, record the GPU, driver, runtime revision, model IDs, image dimensions, and whether the test ran on real Windows hardware. Write `Not tested` rather than inferring success from unit tests.

## Version and releases

`VERSION`, `pai_studio/constants.py`, and `installer.iss` must agree. A Windows installer is produced on Windows x64 using `Build_Windows_Installer.bat` and Inno Setup. Tag-triggered GitHub Actions builds attach the setup executable to the release. Do not publish a beta as production-ready without a clean-machine installation and real inference acceptance test.

## Licensing

The current root license is restrictive and does not grant a general open-source permission. By contributing, you confirm you have the rights to submit the contribution and that you understand it will be governed by the repository's terms unless the project owner changes the license.
