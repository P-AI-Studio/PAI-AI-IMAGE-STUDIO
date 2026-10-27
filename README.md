# P.AI AI Image Studio

**Version:** `1.0.0-beta.2` (Version 1 Beta) · **Platform:** Windows 10/11 x64 · **Inference Backend:** Native `stable-diffusion.cpp` (CUDA 12) + `realesrgan-ncnn-vulkan`

**P.AI AI Image Studio** is a standalone, local-first Windows desktop studio for generative image creation, instruction-guided editing, generative restoration & upscaling, semantic target-guided close-up zoom, 1:1 outpainting, and neural super-resolution. The application owns its entire desktop UI, workflow engine, Model Catalog, diagnostic tools, interactive comparison canvas, and local `data/` workspace—with **zero dependency on external web UIs or ComfyUI servers**.

> **License Notice:** This repository carries a restrictive evaluation/review-only license (`LICENSE`). Public visibility on GitHub does not grant open-source redistribution or modification rights. Third-party native inference engines and model weights are governed by their respective licenses in `licenses/THIRD_PARTY_NOTICES.txt`.

## Key Highlights in Version 1 Beta (`v1.0.0-beta.2`)

- **100% Local Native CUDA Inference:** Powered directly by a loopback-isolated `stable-diffusion.cpp` runtime (`sd-server.exe`) with strict GPU-only placement policies, automatic runtime discovery across existing P.AI installations, and zero cloud API calls during generation.
- **Six Purpose-Built AI Workflows:**
  1. **Image Generate:** Text-to-image synthesis with automatic local translation for non-English prompts (including Persian).
  2. **Image Edit:** Instruction-guided image editing with multi-reference conditioning and identity/color preservation controls.
  3. **Enhance / Upscale:** Generative texture restoration with **Balanced**, **Strong**, and **Maximum** detail profiles.
  4. **Close-up / Smart Zoom:** Open-vocabulary semantic target localization powered by **Qwen3-VL 8B** (supporting both GGUF + `mmproj` and embedded-vision Safetensors such as `qwen3vl_8b_int8_convrot.safetensors`) alongside interactive manual focus selection.
  5. **Square 1:1 Outpaint:** Context-aware canvas extension with feathered seam blending.
  6. **Real-ESRGAN / Real-ESRNet:** Standalone Vulkan neural super-resolution with strict output dimension verification.
- **Two-Stage Lanczos + Ultimate Tile Refinement Pipeline:**
  - **When Ultimate Tile Refinement is ON:** Pass 1 runs full-frame Qwen reconstruction at `requested_resolution ÷ ultimate_scale` (e.g., `512×512` for `1024×1024`, `1024×1024` for `2048×2048`, or `2048×2048` for `4096×4096`), scales the result by `ultimate_scale` (`2×`, `3×`, or `4×` from Model Catalog) using a high-precision **Lanczos** filter to reach the target resolution, and finishes with an overlapping $2\times2$ **Ultimate Tile Refinement** pass featuring 2D cosine feather blending and per-tile mean/std color matching.
  - **When Ultimate Tile Refinement is OFF:** Runs directly at the requested resolution without Lanczos scaling or tile refinement.
- **Fully Connected Model Catalog Tile Controls & Step Compensation:**
  - Every Ultimate tile refinement parameter in **Model Catalog** (`Steps per tile`, `Denoise strength`, `Tile coverage + overlap`, `Refinement passes`, and `Ultimate output scale`) is wired directly into the live execution engine across all workflows.
  - Automatic **Img2Img step schedule compensation** (`compensate_img2img_steps=True`) calculates $\lceil \text{desired\_steps} / \text{strength} \rceil$ so `stable-diffusion.cpp` executes the exact number of requested denoising steps per tile instead of truncating schedules at low denoise strengths.
  - **High-Speed Tile Refinement:** Tile refinement passes automatically set `skip_ref_images=True` (using each cropped tile strictly as `init_image`) and reuse the 8-entry LRU text-conditioning cache, cutting per-tile latency by more than 50%.
- **Interactive 7-Mode Preview Canvas:** Switch seamlessly between **Original**, **Enhanced**, **Slider**, **Side by Side**, **Overlay**, **Difference**, and **Flicker** comparison modes.

## Supported Workflows Overview

| Workflow | Primary Function | Ultimate Tile Refinement Support |
|---|---|---|
| **Image Generate** | Text-to-image synthesis from English or non-English prompts | Optional two-stage half-res Pass 1 $\rightarrow$ Lanczos $\rightarrow$ $2\times2$ tile refinement |
| **Image Edit** | Instruction-guided scene, outfit, lighting, or object editing | Optional two-stage half-res Pass 1 $\rightarrow$ Lanczos $\rightarrow$ $2\times2$ tile refinement |
| **Enhance / Upscale** | Generative restoration and high-frequency detail recovery | Full two-stage pipeline with **Balanced**, **Strong**, and **Maximum** profiles |
| **Close-up / Smart Zoom** | Semantic target crop (Qwen3-VL) or manual click focus + reconstruction | Full two-stage pipeline (`Pass 1` $\rightarrow$ `Lanczos` $\rightarrow$ `4-tile Ultimate stitch`) |
| **Square 1:1** | Non-square aspect ratio extension to a seamless 1:1 square | Full two-stage pipeline with seam-aware compositing |
| **Real-ESRGAN / ESRNet** | Fast standalone neural super-resolution (`2×`, `3×`, `4×`) | Uses dedicated NCNN Vulkan tile engine |

## System Requirements

### Running from Source
- **OS:** Windows 10 or Windows 11 (64-bit).
- **Python:** CPython 3.11, 3.12, or 3.13 x64 with the Windows Python Launcher (`py`).
- **GPU:** NVIDIA CUDA-capable GPU recommended (e.g., RTX 30/40/50 series with 12 GB+ VRAM for full GPU residency, or Auto/Low-VRAM placement modes).
- **Storage:** Sufficient free disk space under `data/` for native runtimes, model weights (GGUF / Safetensors), and generated outputs.

### Installing a Packaged Release
A built Windows installer contains the frozen PAI application, branding assets, required Python libraries and license notices. To keep the installer small, it does **not** include the native `stable-diffusion.cpp` engine, its separate CUDA 12 companion runtime, or model weights. After installation, use **Model Catalog** to choose and download the required model components; the download UI shows progress and validates files. Then click **Load Selected Models**: when needed, the app downloads the pinned native engine (338 MB archive) and the separate CUDA runtime, verifies the engine's SHA-256 and archive contents, and extracts it under `data/`. A network connection is needed for setup; once all selected components are installed, inference runs locally and can work offline.

## Quick Start (Source Mode)

1. Extract the release archive to a writable directory under your Windows user profile.
2. Install 64-bit Python 3.11–3.13 and ensure the `py` launcher is enabled.
3. Run `Install_Update_Requirements.bat` to install Python dependencies (`PySide6`, `Pillow`, `certifi`, `numpy`, and `psutil`).
4. Launch the studio via `Run_PAI_AI_IMAGE_STUDIO.bat`.
5. Open **Model Catalog** in the sidebar to download or select your local Qwen-Image 2.1 diffusion weights (e.g., `rayQwen21_turboV13Int8convrot.safetensors`), text encoder / vision model (`qwen3vl_8b_int8_convrot.safetensors` or GGUF + `mmproj`), and VAE (`texture_fix_vae_for_qwen_image_2.1_bf16.safetensors`).
6. Configure your **Ultimate Tile Refinement** preferences (`Ultimate output scale`, `Steps per tile`, `Denoise strength`, `Tile coverage + overlap`, and `Refinement passes`) directly inside **Model Catalog**.

## Building a Windows Installer

To compile a standalone Windows installer (`PAI-AI-Image-Studio-Setup-1.0.0-beta.2.exe`) on a Windows x64 machine:

1. Install Python 3.11–3.13 x64 and [Inno Setup 6](https://jrsoftware.org/isdl.php).
2. Run `Build_Windows_Installer.bat` from the repository root.
3. The build pipeline runs bytecode compilation and the full automated test suite, freezes the application via PyInstaller into a clean `dist/` folder, and compiles the installer.
4. The installer is saved in the repository's `dist-installer/` folder as `PAI-AI-Image-Studio-Setup-<VERSION>.exe` (for the current `VERSION`, `PAI-AI-Image-Studio-Setup-1.0.0-beta.2.exe`). Copy that EXE to your distribution folder, cloud storage, or a GitHub Release. The output folders are intentionally excluded from Git; **do not commit the compiled installer, models, or runtime files**.

The native engine, CUDA runtime and model weights are deliberately excluded and are downloaded after installation. For GitHub Actions release publishing, merge the intended source changes first, ensure `VERSION` has the new release version, and create a matching new tag (for example, `v1.0.0-beta.3`). Do not overwrite or force-update an existing tag.

## Running Automated Tests

```powershell
python -m pip install -r requirements-test.txt
python -m compileall -q pai_ai_image_studio.py pai_studio tests
python -m pytest -q
```

The full unit, validation, regression, and release-smoke test suite must pass prior to packaging. The CI job reports the current count; documentation intentionally avoids a stale hard-coded number.

## Repository Structure

Only source, required branding assets, licensing/documentation, build configuration, and automated tests are tracked. Compiled installers, model weights, native runtimes, generated images, and local caches are excluded.

```text
README.md / README_FA.md          Project and build instructions
LICENSE / SECURITY.md             Repository terms and security reporting
CHANGELOG.md / VERSION            Release notes and version source of truth
licenses/                         Third-party notices
pai_ai_image_studio.py            Application entry point
pai_studio/                       Application engine, catalog, validation and UI
assets/                           Icons and branding used by the app/installer
requirements*.txt                 Runtime, build and test dependencies
Build_Windows_Installer.bat       One-click local installer build
build_release.ps1 / installer.iss PyInstaller build and Inno Setup configuration
tests/                            Regression and release validation suite
.github/workflows/                Source tests and Windows installer CI/release
```

## Privacy & Security Architecture

- **100% Offline Generation:** Prompts, source photos, and outputs stay strictly on your local machine; the engine binds exclusively to loopback (`127.0.0.1`).
- **Strict File & Archive Validation:** Validates image magic headers, pixel dimensions, GGUF/Safetensors headers, path traversal boundaries, and published SHA-256 digests where available. For assets without a publisher-provided hash, a locally computed SHA-256 can detect later corruption but does not prove publisher authenticity.
- **Explicit GPU Residency Controls:** In strict `full_gpu` mode, all diffusion, text-encoder, vision, and VAE weights are assigned to CUDA without silent CPU fallback.

## License & Third-Party Notices

Refer to `LICENSE` for repository terms and `licenses/THIRD_PARTY_NOTICES.txt` for third-party runtimes (`stable-diffusion.cpp`, `realesrgan-ncnn-vulkan`), Python libraries, and model weight licenses.
