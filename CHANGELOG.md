# Changelog

All notable changes to **P.AI AI Image Studio** are documented in this file.

## [1.0.0-beta.2] — Version 1 Beta hotfix (2026-10-27)

- **Compact installer:** Exclude the large native engine and CUDA companion runtime from the Windows installer; download and verify them on first model load instead. Model weights remain a separate, user-selected Model Catalog download.
- **Installer build:** Replace the fragile `Get-FileHash` command dependency with direct .NET SHA-256 hashing for native runtime integrity checks.
- **Release CI:** Run the Windows installer build directly under PowerShell and on pull requests, so packaging failures are caught before tagging a release.
- **Version consistency:** Read the application version from `VERSION` and use it for the Inno Setup output filename.
- **Safe publishing:** Remove force-push and force-tag behavior; refuse to overwrite an existing release tag and let GitHub Actions publish the installer.
- **Release hygiene:** Remove a cached diagnostic log from the source ZIP and remove hard-coded automated-test totals from the documentation.

## [1.0.0-beta.1] — Version 1 Beta (2026-10-25)

### Highlights & Core Engine Architecture
- **Version 1 Beta Release (`1.0.0-beta.1`):** Packaged clean, GitHub-ready source tree (`pai_studio/` modular architecture + `pai_ai_image_studio.py` entry point) with non-release clutter removed.
- **Native CUDA Inference (`stable-diffusion.cpp`):** Loopback-only (`127.0.0.1`) server orchestration with automatic cross-installation runtime discovery (`sd-server.exe`) and strict GPU-only (`full_gpu`) residency support.
- **Embedded-Vision Safetensors Support:** Full compatibility with single-fileINT8-ConvRot Qwen3-VL Safetensors (`qwen3vl_8b_int8_convrot.safetensors`) without requiring an external `mmproj` file, alongside standard GGUF + `mmproj` bundles.

### Two-Stage Lanczos + Ultimate Tile Refinement Pipeline
- **Resolution-Halved Pass 1 When Ultimate is Active:** When **Ultimate Tile Refinement** is enabled, Pass 1 Qwen runs at `requested_resolution ÷ ultimate_scale` (e.g., `512×512` when `1024×1024` is requested with `2×` Ultimate scale, `1024×1024` when `2048×2048` is requested, or `2048×2048` when `4096×4096` is requested).
- **Model-Catalog-Driven Lanczos Upscaling:** After Pass 1 completes, the intermediate canvas is upscaled using a high-quality **Lanczos** filter by the exact `Ultimate output scale` configured in **Model Catalog** (`2×`, `3×`, or `4×`) to reach the target output resolution before running the $2\times2$ overlapping Ultimate tile refinement and seamless stitch.
- **Direct Single-Pass Execution When Ultimate is Off:** When **Ultimate Tile Refinement** is disabled, workflows render directly at the requested resolution without Lanczos scaling or tile refinement.

### Model Catalog Tile Parameter Connectivity & Speed Optimizations
- **Full Model Catalog Parameter Wiring:** Connected `Steps per tile` (`ultimate_tile_steps`), `Denoise strength` (`ultimate_tile_denoise`), `Tile coverage + overlap` (`ultimate_tile_coverage`), `Refinement passes` (`ultimate_tile_passes`), and `Ultimate output scale` (`ultimate_scale_factor`) directly to tile refinement across all five generative workflows (`Close-up`, `Enhance`, `Square 1:1`, `Image Generate`, and `Image Edit`).
- **Img2Img Step Schedule Compensation (`compensate_img2img_steps=True`):** Fixed `stable-diffusion.cpp` truncating tile steps to `ceil(steps * strength)` (e.g., 3 steps at 8 steps & 0.25 denoise) by automatically sending `ceil(desired_steps / strength)` to the backend so every tile runs the exact number of steps configured in Model Catalog.
- **2× Faster Tile Refinement (`skip_ref_images=True`):** Disabled redundant reference-image vision encoding during tile refinement passes (passing each cropped tile solely as `init_image` for Img2Img refinement) and enabled full reuse of the 8-prompt LRU text-conditioning cache.

### Workflows, Preview & Validation
- **Six Integrated Workflows:** Image Generate, Image Edit, Enhance / Upscale (`Balanced`, `Strong`, `Maximum` profiles), Close-up / Smart Zoom (Qwen3-VL semantic grounding + interactive manual click focus), Square 1:1 Outpaint, and standalone Real-ESRGAN / Real-ESRNet super-resolution.
- **Interactive 7-Mode Comparison Canvas:** Original, Enhanced, Slider, Side by Side, Overlay, Difference, and Flicker modes.
- **Comprehensive Test Suite:** Includes automated unit, validation, regression, and release-smoke tests in `tests/`; the exact count is reported by CI and may change over time.
