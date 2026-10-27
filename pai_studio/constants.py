# -*- coding: utf-8 -*-
"""
Core constants, presets, detail profiles, and model catalogs for P.AI AI IMAGE STUDIO.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import urllib.parse
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

try:
    import certifi
except ImportError:
    certifi = None

from .image_ops import Aborted

VALID_WORKFLOWS: tuple[str, ...] = (
    "image_generate",
    "image_edit",
    "enhance",
    "lora_closeup_1024",
    "square_1_1",
    "upscale_only",
)
VALID_SEED_MODES: tuple[str, ...] = ("Random", "Fixed", "Increment")
VALID_OUTPUT_FORMATS: tuple[str, ...] = ("png", "jpg", "jpeg")
VALID_MEMORY_MODES: tuple[str, ...] = ("auto", "full_gpu", "low_vram")
VALID_PREVIEW_METHODS: tuple[str, ...] = ("proj", "tae", "vae", "none")

APP_NAME = "P.AI AI IMAGE STUDIO"
APP_VERSION = "1.0.0-beta.2"  # Semantic version; beta.2 fixes release packaging and publishing safety
DISPLAY_VERSION = "Version 1 Beta"
SETTINGS_VERSION = 44
MAX_INFERENCE_SIDE = 2048
# These values are passed to stable-diffusion.cpp's native img_gen `strength` field.
# The previous 0.28–0.44 range was too conservative to yield reliably visible differences.
# Higher values intentionally trade source fidelity for stronger texture reconstruction.
ENHANCE_IMG2IMG_STRENGTH = 0.46
ENHANCE_TILE_IMG2IMG_STRENGTH = 0.38
QWEN_DETAIL_PROFILES = {
    "balanced": {"label": "Balanced · preserve identity", "short_label": "Balanced", "strength": 0.46, "tile_strength": 0.38, "description": "Two-stage Qwen cleanup with restrained texture recovery and source-identity priority."},
    "strong": {"label": "Strong · more detail", "short_label": "Strong", "strength": 0.64, "tile_strength": 0.54, "description": "Clearly stronger Qwen denoising and texture reconstruction; moderate identity/texture drift risk."},
    "maximum": {"label": "Maximum · experimental", "short_label": "Maximum", "strength": 0.82, "tile_strength": 0.72, "description": "Aggressive Qwen denoising, reconstruction and dense detail passes; highest identity/content drift risk."},
}
QWEN_DETAIL_PROFILE_TILE_LIMITS = {"balanced": 2048, "strong": 1536, "maximum": 1280}
QWEN_DETAIL_PROFILE_PROMPTS = {
    "balanced": (
        "DETAIL PROFILE: BALANCED. First honor the user's actual edit/target request. Use Qwen img2img for a visible, source-faithful restoration pass. Actively suppress fine random luminance/chroma noise, colored speckles, JPEG blocking/ringing, resampling grit and tiny stair-step/pixelated edges before reconstructing texture; do not simply blur or sharpen noise. Recover only genuine source-supported hair, eyelash/eyebrow strands, fabric weave and surface clarity while keeping smooth skin gradients naturally smooth, soft, and free of artificial pores, stippled grain, or plastic texture. Preserve geometry, identity, markings, colors, light and framing. Keep edges clean, continuous and natural rather than crunchy."
    ),
    "strong": (
        "DETAIL PROFILE: STRONG. First honor the user's actual edit/target request. Perform clearly stronger Qwen source-conditioned denoising and reconstruction than Balanced: remove noticeable luminance/chroma noise, colored speckles, compression grain, ringing, stair-step pixels and jagged edges, then rebuild coherent fine hair strands, eyelashes, cloth weave and clean surface transitions instead of amplifying existing noise. Keep natural skin tones and smooth body/facial skin gradients clean and supple—never add synthetic macro pores, gritty stippling, or pixelated skin texture. Do not use aggressive sharpening as a substitute for reconstruction. Preserve subject identity, geometry, markings, colors, lighting and framing; never add unsupported objects or surface patterns."
    ),
    "maximum": (
        "DETAIL PROFILE: MAXIMUM. First honor the user's actual edit/target request. Use the strongest Qwen reconstruction profile: aggressively suppress visible random luminance/chroma noise, colored speckles, JPEG blocks/ringing, resampling grit, pixelated stair-step edges and halos across the crop, then regenerate crisp, coherent fine material detail—individual hair strands, sharp eyes, clean textile weave, and smooth, natural, unpixelated skin transitions—instead of sharpening pixels or inventing coarse pores. Use repeated source-conditioned reconstruction passes for residual noise and broken edge texture. Preserve the same person/object, geometry, markings, colors, light and composition. Do not invent facial features, objects, lettering or unrelated surface patterns. This is an experimental high-change profile."
    ),
}

def qwen_detail_profile_values(profile: str = "balanced") -> tuple[float, float]:
    """Return selected native Qwen img2img main/tile strengths."""
    item = QWEN_DETAIL_PROFILES.get(str(profile or "balanced").strip().lower(), QWEN_DETAIL_PROFILES["balanced"])
    return float(item["strength"]), float(item["tile_strength"])


def qwen_enhance_reconstruction_strengths(profile: str = "balanced") -> tuple[float, float]:
    """Return Enhance source/detail strengths: Balanced retains the prior Maximum feel;
    Strong adds reconstruction freedom, while Maximum deliberately allows much larger changes.
    """
    values = {"balanced": (0.84, 0.74), "strong": (0.91, 0.82), "maximum": (0.97, 0.92)}
    return values.get(str(profile or "balanced").strip().lower(), values["balanced"])

def qwen_detail_refinement_metadata(profile: str = "balanced", detail_mode: str = "balanced", enabled: bool = True) -> dict[str, Any]:
    """Return the exact selected Qwen detail-pass settings for output metadata.

    Metadata records the configured per-pass tile limit, selected tile strength and number
    of serial tile passes, so results from different UI buttons can be compared later.
    """
    key = str(profile or "balanced").strip().lower()
    if key not in QWEN_DETAIL_PROFILES:
        key = "balanced"
    mode = str(detail_mode or "balanced").strip().lower()
    if mode not in {"fast", "balanced", "max_detail", "ultimate"}:
        mode = "ultimate"
    _main_strength, tile_strength = qwen_detail_profile_values(key)
    if not enabled or mode == "fast":
        tile_limit = None
        passes = 0
    elif mode == "ultimate":
        tile_limit = "overlapping-tile-grid"
        passes = 1
    else:
        coverage_limit = 2048 if mode == "balanced" else 1536
        tile_limit = min(coverage_limit, QWEN_DETAIL_PROFILE_TILE_LIMITS[key])
        passes = 2 if key == "maximum" else 1
    return {
        "qwen_detail_profile": key if enabled else None,
        "qwen_tile_strength": tile_strength if enabled else None,
        "qwen_detail_mode": mode if enabled else None,
        "qwen_tile_limit": tile_limit,
        "qwen_detail_refinement_passes": passes,
    }


def apply_qwen_detail_profile_prompt(prompt: str, profile: str = "balanced") -> str:
    """Inject exactly one selected profile early in the real Qwen prompt."""
    key = str(profile or "balanced").strip().lower()
    if key not in QWEN_DETAIL_PROFILE_PROMPTS:
        key = "balanced"
    text = str(prompt or "").strip()
    # If a caller reuses an already-profiled prompt, remove every former profile instruction
    # before applying the new selection; otherwise prompt text and strength can disagree.
    for previous_guidance in QWEN_DETAIL_PROFILE_PROMPTS.values():
        text = text.replace(previous_guidance, "").strip()
    guidance = QWEN_DETAIL_PROFILE_PROMPTS[key]
    if not text:
        return guidance
    # Put the chosen profile after the first instruction sentence, not at the very tail where
    # a model's text-encoder truncation could make Balanced/Strong/Maximum indistinguishable.
    boundary = text.find(". ")
    if 35 <= boundary <= 280:
        return f"{text[:boundary + 1]}\n\n{guidance}\n\n{text[boundary + 2:]}"
    return f"{guidance}\n\nUSER TASK (honor this instruction): {text}"

# Ultimate 4× Enhance can legitimately exceed 4096 px on one side. Qwen inference
# is still capped at MAX_INFERENCE_SIDE and large outputs are refined in overlapping
# tiles, so this is an output-canvas cap, not a promise of one 8K diffusion pass.
MAX_OUTPUT_SIDE = 8192
FIXED_SQUARE_OUTPUT_SIZES = (1024, 2048, 4096)
REALESRGAN_NCNN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)
# Official release first; SourceForge mirror is a fallback for networks that block GitHub release assets.
REALESRGAN_NCNN_URLS = (
    REALESRGAN_NCNN_URL,
    "https://sourceforge.net/projects/real-esrgan.mirror/files/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip/download",
)
# The upstream Real-ESRGAN NCNN runner is retained for its original models.
# Upscayl-NCNN supports custom NCNN .param/.bin models for the extended photo catalogue.
UPSCAYL_NCNN_URL = (
    "https://github.com/upscayl/upscayl-ncnn/releases/download/20251207-174704/"
    "upscayl-bin-20251207-174704-windows.zip"
)
UPSCAYL_NCNN_SHA256 = "1f0f65c5d2ade866555e2ac467d35952c35a47080a4060fe56a1ab028f67258a"
UPSCAYL_NCNN_ARCHIVE_SIZE = 2421760
UPSCAYL_NCNN_MODEL_COMMIT = "4b6d2cfa59c7442af115dfc6e50fd8d7d40b96ef"
UPSCAYL_NCNN_MODEL_BASE_URL = (
    "https://raw.githubusercontent.com/upscayl/custom-models/"
    + UPSCAYL_NCNN_MODEL_COMMIT + "/models/"
)

REALESRGAN_MODEL_NAMES = {
    "Photo · Real-ESRGAN x4plus (4× native)": "realesrgan-x4plus",
    "Photo · Real-ESRNet x4plus (4× conservative)": "realesrnet-x4plus",
    "Photo · RealESRGAN General x4 v3 (fast / natural)": "RealESRGAN_General_x4_v3",
    "Photo · RealESRGAN General WDN x4 v3 (noise-aware)": "RealESRGAN_General_WDN_x4_v3",
    "Photo · NMKD Siax 200k (natural detail)": "4x_NMKD-Siax_200k",
    "Photo · LSDIR+ C (natural detail)": "4xLSDIRplusC",
    "Photo · Nomos8k SC (photographic detail)": "4xNomos8kSC",
    "Photo · HFA2k (photographic restoration)": "4xHFA2k",
    "Photo · LSDIR Compact C3 (lightweight)": "4xLSDIRCompactC3",
    "Photo · UniScale Restore": "uniscale_restore",
    "Photo · NMKD Superscale SP (optional / clean source only)": "4x_NMKD-Superscale-SP_178000_G",
    "Anime / illustration · Real-ESRGAN (4× native)": "realesrgan-x4plus-anime",
    "Anime / video · AnimeVideo-v3 (4× native)": "realesr-animevideov3",
    "Anime / video · AnimeVideo-v3 x3 (3× native)": "realesr-animevideov3-x3",
    "Anime / video · AnimeVideo-v3 x2 (2× native)": "realesr-animevideov3-x2",
}
REALESRGAN_MODEL_NATIVE_SCALES: dict[str, int] = {
    "realesrgan-x4plus": 4,
    "realesrnet-x4plus": 4,
    "realesrgan-x4plus-anime": 4,
    "realesr-animevideov3": 4,
    "realesr-animevideov3-x4": 4,
    "realesr-animevideov3-x3": 3,
    "realesr-animevideov3-x2": 2,
    "RealESRGAN_General_x4_v3": 4,
    "RealESRGAN_General_WDN_x4_v3": 4,
    "4x_NMKD-Siax_200k": 4,
    "4xLSDIRplusC": 4,
    "4xNomos8kSC": 4,
    "4xHFA2k": 4,
    "4xLSDIRCompactC3": 4,
    "uniscale_restore": 4,
    "4x_NMKD-Superscale-SP_178000_G": 4,
}


def realesrgan_native_scale(model_name: str) -> int:
    """Return the intrinsic native upscale factor for the selected NCNN model."""
    key = str(model_name or "realesrgan-x4plus").strip()
    return int(REALESRGAN_MODEL_NATIVE_SCALES.get(key, 4))


REALESRGAN_MODEL_ZOO_URL = "https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md"
REALESRGAN_HF_NCNN_BASE = "https://huggingface.co/edgetools/realesrgan/resolve/main/"
REALESRGAN_CATALOG: list[dict[str, Any]] = [
    {
        "id": "realesrgan-x4plus",
        "title": "Real-ESRGAN x4plus · Photo / general images (4× native)",
        "description": "General photographic upscaler with stronger detail recovery; automatically runs at its native 4× model scale.",
        "pytorch_weights_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "backend": "official",
    },
    {
        "id": "realesrnet-x4plus",
        "title": "Real-ESRNet x4plus · Conservative photo restoration (4× native)",
        "description": "A conservative restoration variant that prioritizes source fidelity over aggressive texture synthesis.",
        "pytorch_weights_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "backend": "official",
    },
    {
        "id": "realesrgan-x4plus-anime",
        "title": "Real-ESRGAN x4plus anime · Anime / illustration (4× native)",
        "description": "Optimized for anime-style images and clean linework; automatically runs at its native 4× model scale.",
        "pytorch_weights_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "backend": "official",
    },
    {
        "id": "realesr-animevideov3",
        "title": "AnimeVideo-v3 · Lightweight animation upscaler (4× native)",
        "description": "Lighter anime/video model; automatically runs at its native 4× model scale.",
        "pytorch_weights_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "backend": "official",
    },
    {
        "id": "realesr-animevideov3-x3",
        "title": "AnimeVideo-v3 x3 · Lightweight animation upscaler (3× native)",
        "description": "Lighter animation model; automatically runs at its native 3× model scale.",
        "pytorch_weights_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "backend": "official",
    },
    {
        "id": "realesr-animevideov3-x2",
        "title": "AnimeVideo-v3 x2 · Lightweight animation upscaler (2× native)",
        "description": "Lighter animation model; automatically runs at its native 2× model scale.",
        "pytorch_weights_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "backend": "official",
    },
    {
        "id": "RealESRGAN_General_x4_v3", "title": "RealESRGAN General x4 v3 · Natural photos / fast",
        "description": "Small, fast general-image model for natural and photographic content.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "RealESRGAN_General_x4_v3", "param_size": 5019, "bin_size": 2435272,
        "param_blob_sha1": "221234ab0dc2641a11038f3682669ded6ba9e6d6", "bin_blob_sha1": "e0a8b6ca061931284832a67cdd4112d94a7c98a0",
    },
    {
        "id": "RealESRGAN_General_WDN_x4_v3", "title": "RealESRGAN General WDN x4 v3 · Noise-aware photo",
        "description": "A compact general-photo variant tuned toward denoising, useful when the source has visible noise.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "RealESRGAN_General_WDN_x4_v3", "param_size": 5019, "bin_size": 2435272,
        "param_blob_sha1": "221234ab0dc2641a11038f3682669ded6ba9e6d6", "bin_blob_sha1": "a763b5065f282918ddb1aaa035d4f8b6b69fc39b",
    },
    {
        "id": "4x_NMKD-Siax_200k", "title": "NMKD Siax 200k · Natural / lightly compressed photos",
        "description": "Community model recommended for clean or lightly compressed real-world images.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "4x_NMKD-Siax_200k", "param_size": 108037, "bin_size": 66793352,
        "param_blob_sha1": "c6d62b67aee7d30d8c2a3cc4847727c1c6155d74", "bin_blob_sha1": "f7333bdf6e62c44e723113e2674908f391947baf",
    },
    {
        "id": "4xLSDIRplusC", "title": "LSDIR+ C · High-quality natural-image detail",
        "description": "Community 4× model for detailed natural images; compare against Real-ESRGAN on your source.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "4xLSDIRplusC", "param_size": 108039, "bin_size": 33424520,
        "param_blob_sha1": "aafe22d603d714ff07f3b1b24f5f0149c9767911", "bin_blob_sha1": "0746b5f44d28c90cbf87e438c98f0cdd52ba975a",
    },
    {
        "id": "4xNomos8kSC", "title": "Nomos8k SC · Photographic detail",
        "description": "Community model aimed at detailed high-resolution natural images.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "4xNomos8kSC", "param_size": 108039, "bin_size": 33424520,
        "param_blob_sha1": "aafe22d603d714ff07f3b1b24f5f0149c9767911", "bin_blob_sha1": "ce5cfda16b37a9808d956b748e52564f6e978fd5",
    },
    {
        "id": "4xHFA2k", "title": "HFA2k · Photographic restoration",
        "description": "Community restoration model for natural photo detail and texture.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "4xHFA2k", "param_size": 108039, "bin_size": 33424520,
        "param_blob_sha1": "aafe22d603d714ff07f3b1b24f5f0149c9767911", "bin_blob_sha1": "80d7a5d56afaf06ddb2770c022d44c3677a4d37c",
    },
    {
        "id": "4xLSDIRCompactC3", "title": "LSDIR Compact C3 · Lightweight natural-image model",
        "description": "Compact SRVGG model that trades some capacity for smaller weights and quicker inference.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "4xLSDIRCompactC3", "param_size": 2767, "bin_size": 1247368,
        "param_blob_sha1": "0c02905d51229cfe2c2e9f6562b01ce9dcc0713b", "bin_blob_sha1": "5e9ec6ebb7d1811ccd9286a253e2575ae73b643c",
    },
    {
        "id": "uniscale_restore", "title": "UniScale Restore · Natural photo restoration",
        "description": "Community model intended for general restoration; compare results on your source.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "uniscale_restore", "param_size": 108039, "bin_size": 33424520,
        "param_blob_sha1": "aafe22d603d714ff07f3b1b24f5f0149c9767911", "bin_blob_sha1": "2977b7eebbbdc7557a3a58f372fbb79006a868f8",
    },
    {
        "id": "4x_NMKD-Superscale-SP_178000_G", "title": "NMKD Superscale SP · Clean / artifact-free photo (optional)",
        "description": "Best tested on clean, artifact-free images; can amplify source defects and is deliberately not the default.",
        "model_source_url": "https://github.com/upscayl/custom-models/tree/main/models", "backend": "upscayl",
        "model_file": "4x_NMKD-Superscale-SP_178000_G", "param_size": 108037, "bin_size": 66793352,
        "param_blob_sha1": "c6d62b67aee7d30d8c2a3cc4847727c1c6155d74", "bin_blob_sha1": "a5019b73ffbaede2436e6c99c585f4dcdf2f69a7",
    },
]
REALESRGAN_CATALOG_BY_ID = {item["id"]: item for item in REALESRGAN_CATALOG}


def _realesrgan_model_assets_present(archive_names: set[str]) -> bool:
    """Return true when at least one complete NCNN model pair is present.

    Portable archives may not contain every optional model family. Requiring every
    Real-ESRNet/anime pair caused valid upstream ZIPs to be rejected before install.
    The selected model is checked separately immediately before inference.
    """
    basenames = {Path(name.replace("\\", "/")).name.lower() for name in archive_names}
    for stem in ("realesrgan-x4plus", "realesrnet-x4plus", "realesrgan-x4plus-anime"):
        if f"{stem}.param" in basenames and f"{stem}.bin" in basenames:
            return True
    for scale in (2, 3, 4):
        stem = f"realesr-animevideov3-x{scale}"
        if f"{stem}.param" in basenames and f"{stem}.bin" in basenames:
            return True
    return False


def restore_realesrgan_model_pair_from_archive(archive: Path, model_dir: Path, stem: str) -> bool:
    """Restore one selected NCNN .param/.bin pair from the cached official ZIP.

    Only known model stems and exact basenames are extracted. This repairs a missing
    optional pair such as Real-ESRNet without redownloading a working executable.
    The caller validates minimum model sizes before running inference.
    """
    allowed = {
        "realesrgan-x4plus", "realesrnet-x4plus", "realesrgan-x4plus-anime",
        "realesr-animevideov3-x2", "realesr-animevideov3-x3", "realesr-animevideov3-x4",
    }
    stem = str(stem).lower()
    if stem not in allowed:
        return False
    archive = Path(archive)
    model_dir = Path(model_dir)
    if not archive.is_file():
        return False
    wanted = {f"{stem}.param", f"{stem}.bin"}
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            if zf.testzip() is not None:
                return False
            entries = {}
            for info in zf.infolist():
                base = Path(info.filename.replace("\\", "/")).name.lower()
                if base in wanted and not info.is_dir():
                    entries[base] = info
            if set(entries) != wanted:
                return False
            model_dir.mkdir(parents=True, exist_ok=True)
            for base in sorted(wanted):
                target = model_dir / base
                part = target.with_name(target.name + ".restore.part")
                try:
                    with zf.open(entries[base], "r") as source, part.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    os.replace(part, target)
                finally:
                    part.unlink(missing_ok=True)
        return True
    except (OSError, zipfile.BadZipFile, RuntimeError, KeyError):
        return False



def preferred_preview_mode_for_workflow(workflow: str) -> str:
    """Return a default view mode; source workflows remain user-selectable."""
    if workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:
        return "Side by side"
    return "Enhanced"

DEFAULTS = {
    "workflow_mode": "image_generate",
    "theme": "dark",
    "live_preview_method": "tae",
    "preview_mode": "Side by side",
    "encoder_cpu": False,
    "encoder_cpu_auto": False,
    "gpu_index": 0,
    "gpu_memory_mode": "auto",
    "identity_lock_enabled": True,
    "quality": "Turbo · 8 steps",  # shared default for the selected Turbo checkpoint
    "std_quality": "Turbo · 8 steps",
    "gen_quality": "Turbo · 8 steps",
    "edit_quality": "Turbo · 8 steps",
    "std_prompt_preset_name": "",
    "gen_prompt_preset_name": "",
    "edit_prompt_preset_name": "",
    "std_size_preset": "Scale",
    "std_output_scale": 2,
    "std_output_width": 1024,
    "std_output_height": 1024,
    "gen_prompt": "",
    "gen_size": "1024 × 1024 (1:1 Square)",
    "gen_w": 1024,
    "gen_h": 1024,
    "edit_prompt": "",
    "edit_size": "Original Size",
    "edit_w": 1024,
    "edit_h": 1024,
    "edit_fmt": "png",
    "std_prompt": "",
    "num_inference_steps": 8,
    "gen_count": 1,
    "gen_seed": -1,
    "gen_seed_mode": "Random",
    "edit_seed": -1,
    "edit_seed_mode": "Random",
    "closeup_auto_zoom": True,
    "grounding_profile": "balanced",
    "closeup_zoom": 2.2,
    "closeup_side": 1024,
    "closeup_enhance": True,
    "save_lora_caption": True,
    "lora_trigger_word": "",
    "square_side": 1024,
    "square_enhance": False,
    "retry_diff": 25.0,
    "warn_diff": 18.0,
    "enhance_detail_mode": "fast",  # ultimate / fast (legacy key retained; Ultimate is opt-in)
    "ultimate_qwen_upscale_enabled": False,
    "ultimate_qwen_denoise_strength": 0.20,
    "ultimate_qwen_tile_coverage": 50,
    "ultimate_qwen_passes": 1,
    "ultimate_qwen_steps": 8,
    "ultimate_qwen_scale": 2,
    "auto_realesrgan_enabled": False,
    "watermark_removal_enabled": False,
    "qwen_preview_enabled": True,
    "qwen_detail_profile": "balanced",
    # A structural-difference retry can rerun the entire diffusion model.
    # Keep one attempt by default; additional attempts remain a user option.
    "max_attempts": 1,
    "upscale_enabled": False,
    "upscale_model": "realesrgan-x4plus",
    "upscale_tile": 0,
    "upscale_tta": False,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
QUALITY_PRESETS = ["Turbo · 8 steps", "Fast · 12 steps", "Balanced · 20 steps", "Standard · 28 steps"]
QUALITY_STEPS = {"Turbo · 8 steps": 8, "Fast · 12 steps": 12, "Balanced · 20 steps": 20, "Standard · 28 steps": 28}

RESTORE_PHOTO_PROMPT = (
    "Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, "
    "keeping the same composition, people and colors."
)
RESTORE_OLD_PRINT_PROMPT = (
    "Restore the old photo: remove scratches, dust, fading and grain and recover sharp, natural detail and color, "
    "keeping the same composition and people."
)
FAITHFUL_ENHANCE_PROMPT = (
    "Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, "
    "keeping the same composition, people and colors. "
    "Reconstruct this photograph with Qwen image-to-image as a careful photographic restoration and re-render, "
    "not a simple resize, pixel copy, or sharpening filter. Use the source to anchor scene meaning, identity, geometry "
    "and composition, but actively regenerate the image surface where detail is missing or degraded into a crisp, ultra-detailed "
    "raw studio DSLR photograph. Rebuild believable fine texture and clean 3D shape across the whole image: natural skin microtexture "
    "with realistic pores and fine vellus body hair (peach fuzz) without invented wrinkles or blemishes, individual hair strands, "
    "eyelashes, eyebrows, tactile fabric weave, clothing fuzz/fibers and seams, and realistic wall, wood, stone, metal and background "
    "material texture. Remove random luminance/chroma noise, compression blocks, ringing, pixel speckles and resampling "
    "grit before reconstructing detail; do not turn artifacts into texture. Preserve the same person/object, facial identity, "
    "expression, pose, anatomy, object count, readable text, colors, lighting, perspective and framing. Make a visibly cleaner, "
    "more detailed photographic reconstruction while keeping the original scene recognizable. Never merely enlarge the original "
    "pixels. Do not invent objects, facial features, markings, lettering, wrinkles, acne, folds or unsupported patterns; avoid halos, "
    "crunchy sharpening, plastic skin, recoloring, restyling or composition changes."
)
CLOSEUP_SEMANTIC_AUTO_PROMPT = (
    "This input is already a square close-up crop focused on {target}. Refine this crop faithfully: keep the "
    "target prominent and in the same position, preserve its identity, shape, proportions, markings, material, "
    "colors, and light direction, and recover only source-supported texture and edge clarity. Keep the crop and "
    "composition fixed. Do not recreate the full scene, zoom back out, add margins, change anatomy, or invent detail."
)
CLOSEUP_MANUAL_CROP_PROMPT = (
    "Refine this existing square crop of {target}. Reduce resampling softness and compression artifacts, "
    "and recover edge clarity and texture supported by the source. Preserve the target's identity, geometry, "
    "markings, color, lighting, and current crop. Do not restyle, reframe, or invent detail."
)
SQUARE_OUTPAINT_ONLY_PROMPT = (
    "Extend this image naturally to a square canvas by generating only the scene content that lies beyond the "
    "original image boundaries. Treat the supplied image as the complete, real reference image, not as a padded "
    "canvas. Preserve the existing subjects, objects, readable text, colors, and composition in the original image. "
    "Continue the immediately adjacent scene outward from the exact left and right edge pixels. Extend walls, floors, "
    "furniture, landscape, shadows, reflections, lines, and textures along their existing direction and perspective; "
    "keep horizon and vanishing lines continuous across the seam. Match the original lens, depth of field, exposure, "
    "white balance, grain, and material texture. Generate only the missing side margins; do not recreate the central "
    "image. Never use solid-color panels, blurred bars, frames, borders, mirrored duplicates, or letterboxing. Do not "
    "add unrelated subjects or text."
)
SQUARE_ENHANCE_OUTPAINT_PROMPT = (
    FAITHFUL_ENHANCE_PROMPT + " Extend the original scene naturally to the requested square canvas by generating "
    "only content beyond the original image boundaries. The reference image is not padded with blank margins. "
    "Preserve its subjects, objects, identity, composition, colors, and lighting; continue the scene with matching "
    "perspective, horizon, depth of field, and local texture. Never produce solid-color side panels, blurred bars, "
    "frames, borders, or letterboxing. Do not add unrelated subjects or objects."
)
EDIT_PRESETS = OrderedDict([
    ("Custom instruction (free edit)", ""),
    ("Lighting & Color Grade", "Change only the exposure, white balance, and color grade to: [describe the intended look]. Keep the original subjects, materials, framing, and light direction unless the requested grade requires a change."),
    ("Background Replacement", "Replace only the background with [describe the new background]. Keep the foreground subject and its edges, identity, pose, clothing, and accessories unchanged. Match the new background's perspective, depth of field, and light direction to the subject."),
    ("Clothing / Outfit Edit", "Change only the subject's clothing to [describe the requested outfit]. Preserve the person's identity, face, pose, body proportions, hair, accessories, and background. Make fabric, seams, folds, and shadows consistent with the pose and lighting."),
    ("Object Removal / Clean-up", "Remove only [identify the object or blemish]. Reconstruct the newly revealed area from its immediate surroundings, matching texture, perspective, and lighting. Leave all other objects and the composition unchanged."),
    ("Studio Portrait Retouch", "Retouch the portrait subtly: balance exposure and skin tone and remove only temporary blemishes. Preserve identity, face geometry, age, expression, gaze, hairstyle, makeup, and clothing. Retain realistic skin texture, pores, fine hair, and natural shadows."),
    ("Exact Text Replacement", "Replace the existing text with the exact text \"[type the exact wording]\". Match the original text placement, baseline, perspective, visual weight, and material appearance. Keep every other part of the image unchanged."),
    ("Product Photo Cleanup", "Improve the product photograph's exposure, neutral color balance, and visible surface clarity. Preserve the exact product design, shape, proportions, markings, logo, ports, labels, and object count. Keep the original framing and background unless explicitly requested otherwise."),
])
GEN_PROMPT_PRESETS = OrderedDict([
    ("Custom scene (write your prompt)", ""),
    ("Photorealistic portrait", "A photorealistic portrait of [subject], [pose and expression], wearing [clothing and materials], in [environment]. Frame the subject as [close-up / half-body / full-body], with [camera angle and lens perspective]. Light the scene with [light source, direction, softness, and shadows]. Use [color palette] and natural material detail. Keep the composition balanced with clear separation between subject and background."),
    ("Product photography", "A professional product photograph of [exact product and variant] on [background/surface]. Show [view angle] with the product fully inside the frame and enough negative space around it. Use [softbox / window / studio] lighting from [direction], controlled reflections, accurate color, and realistic material texture. Preserve the exact product design and markings. Do not invent logos or readable labels."),
    ("Poster / readable typography", "A [orientation] poster for [subject or event], with [background and palette]. Place the main headline at [position] in [typeface style, weight, size, and color]. The exact headline reads '[exact text]'. Place the supporting line '[exact text]' at [position]. Keep all lettering legible, correctly spelled, aligned, and visually consistent; do not add any other text. Use a clear hierarchy, balanced margins, and [lighting/illustration style]."),
    ("Landscape / environment", "A [wide / panoramic / vertical] [photorealistic / painterly] landscape of [place and time of day]. In the foreground, [elements and placement]; in the middle distance, [elements]; along the horizon, [distant features]. The light comes from [direction] with [weather, atmospheric depth, shadows, and reflections]. Use [palette and mood], coherent scale and perspective, and a composition with deliberate foreground-to-background depth."),
    ("Illustration / anime", "A [style] illustration of [subject] [action] in [setting]. Use [linework, shading technique, palette, and material treatment]. Place [key elements] at [positions] with a readable silhouette and intentional negative space. Light the scene with [direction and quality]. Keep character design and prop details internally consistent; include no unintended text."),
    ("Transparent sticker / RGBA", "This is an RGBA image with transparency. [Describe the single subject and its pose, silhouette, and style]. Use clean, well-defined outer edges and a cohesive palette. The image has alpha channel and the background is transparent."),
])
PROMPT_PRESETS = OrderedDict([
    ("Custom instruction (manual)", ""),
    ("Faithful enhance (default)", FAITHFUL_ENHANCE_PROMPT),
    ("Restore photo · community benchmark", RESTORE_PHOTO_PROMPT),
    ("Restore old print / scratched photo", RESTORE_OLD_PRINT_PROMPT),
    ("Close-up · target detail cleanup", "Refine only the requested target or crop. Reduce resampling softness and compression artifacts, and restore texture supported by the source. Preserve the target's identity, geometry, markings, colors, lighting, and crop. Do not redesign or invent details."),
    ("Portrait / skin detail", "Refine this portrait's clarity and tonal balance. Preserve the person's identity, facial proportions, age, expression, gaze, hairstyle, and lighting. Recover fine skin, eyelash, eyebrow, hair, and fabric detail only where visible or supported by the source. Avoid beauty-filter smoothing, exaggerated eyes, altered facial structure, or invented features."),
    ("Gentle clean-up (minimal change)", "Apply a restrained technical cleanup. Reduce obvious noise and compression artifacts with only mild sharpening. Preserve all visible content, shape, linework, colors, exposure, lighting, text, and framing. Do not restyle the image or invent detail."),
    ("Strong detail (careful restoration)", "Improve clarity and recover fine, source-consistent texture in skin, hair, fabric, foliage, illustration linework, and surfaces. Preserve identity, anatomy, pose, perspective, composition, colors, and lighting. Avoid halos, crunchy sharpening, plastic smoothing, and synthetic texture."),
    ("Remove compression / noise", "Reduce blocking, ringing, banding, and visible noise while retaining real edge detail and fine texture. Keep the original image style, colors, tonal range, text, composition, and subject identity. Do not blur away detail or invent replacement features."),
    ("Square 1:1 · Outpaint only (No Enhance)", SQUARE_OUTPAINT_ONLY_PROMPT),
    ("Square 1:1 · Enhance + Outpaint", SQUARE_ENHANCE_OUTPAINT_PROMPT),
])

# Which natural-language templates are appropriate for each Qwen workflow.
# Qwen Output Scale (including 1×) is an Enhance size setting and uses the selected
# Enhance prompt. The standalone Real-ESRGAN upscaler is not prompt-driven.
WORKFLOW_PROMPT_PRESETS = {
    "enhance": (
        "Custom instruction (manual)", "Faithful enhance (default)",
        "Restore photo · community benchmark", "Restore old print / scratched photo",
        "Portrait / skin detail", "Gentle clean-up (minimal change)",
        "Strong detail (careful restoration)", "Remove compression / noise",
    ),
    "lora_closeup_1024": (
        "Custom instruction (manual)", "Close-up · target detail cleanup",
        "Portrait / skin detail", "Gentle clean-up (minimal change)",
        "Strong detail (careful restoration)", "Remove compression / noise",
    ),
    "square_1_1": (
        "Custom instruction (manual)", "Faithful enhance (default)",
        "Gentle clean-up (minimal change)", "Strong detail (careful restoration)",
        "Remove compression / noise", "Square 1:1 · Outpaint only (No Enhance)",
        "Square 1:1 · Enhance + Outpaint",
    ),
}


# Curated Hugging Face repositories for Qwen-Image 2.1 Turbo Uncensored, INT8-ConvRot, and Qwen3-VL-8B Text Encoders.
QWEN_LICENSE_URL = "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF"
QWEN_REPO_URL = "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"
QWEN_TURBO_REPO_URL = "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF"
QWEN_VAE_REPO_URL = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1"
QWEN_UNSLOTH_TURBO_REPO_URL = "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-FP8"
QWEN_HERETIC_CONVROT_REPO_URL = "https://huggingface.co/Nic3301/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot"
QWEN_SAKURA_UNCENSORED_REPO_URL = "https://huggingface.co/webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF"
QWEN3VL_ABLITERATED_V2_REPO_URL = "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF"
QWEN3VL_HERETIC_I1_REPO_URL = "https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF"
QWEN3VL_BARTOWSKI_REPO_URL = "https://huggingface.co/bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF"
QWEN3VL_OFFICIAL_GGUF_REPO_URL = "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF"

# For files without an upstream pinned SHA-256, the downloader computes SHA-256 locally,
# records it in model-manifest.json and verifies size/mtime plus that recorded digest on later runs.
MODEL_CATALOG: list[dict[str, Any]] = [
    # --- Top Recommended Diffusion Models (INT8-ConvRot & Uncensored Turbo) ---
    {
        "id": "unsloth_turbo_int8_convrot", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo · Unsloth INT8-ConvRot Safetensors (7.26 GB · recommended best quality)",
        "filename": "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors",
        "url": "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-FP8/resolve/main/Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors",
        "sha256": "", "size_hint": 7_260_000_000, "license": "Qwen Research License", "license_url": QWEN_UNSLOTH_TURBO_REPO_URL,
        "source": "unsloth/Qwen-Image-2.1-Turbo-FP8", "default": True,
        "description": "Unsloth's recommended INT8 W8A8 Hadamard-rotated ConvRot build of Qwen-Image 2.1 Turbo (0.037 LPIPS vs BF16; 8-step Euler schedule).",
    },
    {
        "id": "comfy_turbo_int8_convrot", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo · Comfy-Org INT8-ConvRot Safetensors (7.26 GB · official template)",
        "filename": "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
        "sha256": "", "size_hint": 7_260_000_000, "license": "Qwen Research License", "license_url": QWEN_VAE_REPO_URL,
        "source": "Comfy-Org/Qwen-Image-2.1", "default": False,
        "description": "Official Comfy-Org INT8-ConvRot repackaged Qwen-Image 2.1 Turbo diffusion checkpoint.",
    },
    {
        "id": "sakura_turbo_uc_q8", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo Uncensored · Sakura Q8_0 GGUF (7.07 GB · highest GGUF quality)",
        "filename": "Sakura-Image-2.1-Turbo-Q8_0-7.07GiB.gguf",
        "url": "https://huggingface.co/webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Sakura-Image-2.1-Turbo-Q8_0-7.07GiB.gguf",
        "sha256": "", "size_hint": 7_070_000_000, "license": "Qwen Research License", "license_url": QWEN_SAKURA_UNCENSORED_REPO_URL,
        "source": "webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF", "default": False,
        "description": "Sakura Uncensored release of Qwen-Image 2.1 Turbo Q8_0 GGUF with F32 attention norms for stable-diffusion.cpp and ComfyUI.",
    },
    {
        "id": "sakura_turbo_uc_q5", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo Uncensored · Sakura Q5_K_M GGUF (4.60 GB · 12–16 GB GPU balanced)",
        "filename": "Sakura-Image-2.1-Turbo-Q5_K_M-4.60GiB.gguf",
        "url": "https://huggingface.co/webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Sakura-Image-2.1-Turbo-Q5_K_M-4.60GiB.gguf",
        "sha256": "", "size_hint": 4_600_000_000, "license": "Qwen Research License", "license_url": QWEN_SAKURA_UNCENSORED_REPO_URL,
        "source": "webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF", "default": False,
        "description": "Balanced 5-bit GGUF from the Sakura Uncensored Qwen-Image 2.1 Turbo collection.",
    },
    {
        "id": "unsloth_turbo_q6_k", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo · Unsloth Dynamic 2.0 Q6_K GGUF (5.75 GB)",
        "filename": "qwen-image-2.1-turbo-Q6_K.gguf",
        "url": "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/resolve/main/qwen-image-2.1-turbo-Q6_K.gguf",
        "sha256": "", "size_hint": 5_750_000_000, "license": "Qwen Research License", "license_url": "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF",
        "source": "unsloth/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "Unsloth Dynamic 2.0 sensitivity-calibrated 6-bit GGUF of Qwen-Image 2.1 Turbo.",
    },
    {
        "id": "atomic_turbo_ad_q4", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat AD-Q4_K (4.20 GB · 12 GB GPU GGUF)",
        "filename": "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-AD-Q4_K.gguf",
        "sha256": "4bb73c53cbe284bbd6d69b9fdc59539c531f0389dd2aa567663d777f8130bc59", "size_hint": 4_200_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "Faster, memory-efficient 4-bit GGUF option for 12 GB GPUs; designed for 8 steps with a fixed sigma schedule.",
    },
    {
        "id": "atomic_turbo_ad_q5", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat AD-Q5_K (5.40 GB · quality-balanced GGUF)",
        "filename": "Qwen-Image-2.1-Turbo-AD-Q5_K.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-AD-Q5_K.gguf",
        "sha256": "c56db41508c4c3c0d5782e40e4f153c38a1154ac9466c1c5c50a6e4bd09f6a69", "size_hint": 5_400_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "Higher quality than AD-Q4_K with larger weights; needs more free RAM/VRAM.",
    },
    {
        "id": "atomic_turbo_ad_q6", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat AD-Q6_K (6.71 GB)",
        "filename": "Qwen-Image-2.1-Turbo-AD-Q6_K.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-AD-Q6_K.gguf",
        "sha256": "", "size_hint": 6_710_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "More accurate than AD-Q4_K, with higher memory use.",
    },
    {
        "id": "atomic_turbo_bf16", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat BF16 (14.23 GB · high VRAM only)",
        "filename": "Qwen-Image-2.1-Turbo-BF16.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-BF16.gguf",
        "sha256": "9849f4f5606e7130dafa8ca4fe6d773d91d83b0eed983f6245de3195c9105949", "size_hint": 14_230_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "Reference-quality BF16 weights; about 14.23 GB for weights alone, unsuitable for 12 GB GPUs.",
    },
    {
        "id": "atomic_turbo_q8", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat Q8_0 (7.59 GB)",
        "filename": "Qwen-Image-2.1-Turbo-Q8_0.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-Q8_0.gguf",
        "sha256": "", "size_hint": 7_590_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "The closest published quantized version to BF16; requires substantial RAM/VRAM.",
    },
    {
        "id": "atomic_turbo_ad_q3", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat AD-Q3_K (3.60 GB)",
        "filename": "Qwen-Image-2.1-Turbo-AD-Q3_K.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-AD-Q3_K.gguf",
        "sha256": "", "size_hint": 3_600_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "Smaller footprint; details and faces may shift more than with larger variants.",
    },
    {
        "id": "atomic_turbo_ad_q2", "role": "diffusion", "title": "Qwen-Image 2.1 Turbo · AtomicChat AD-Q2_K (2.55 GB · low memory)",
        "filename": "Qwen-Image-2.1-Turbo-AD-Q2_K.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-AD-Q2_K.gguf",
        "sha256": "8ab2888f605030a894c6238d978f51427bfde6732f4a112c71cff0439e265d04", "size_hint": 2_550_000_000, "license": "Qwen Research License", "license_url": QWEN_LICENSE_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", "default": False,
        "description": "For memory-constrained systems; among AD options, this has the largest visual difference from BF16.",
    },
    {
        "id": "atomic_turbo_uc_q4", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo Uncensored · AtomicChat AD-Q4_K GGUF (5.29 GB)",
        "filename": "Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf",
        "sha256": "", "size_hint": 5_290_000_000, "license": "Apache-2.0", "license_url": QWEN_REPO_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF", "default": False,
        "description": "AtomicChat Qwen-Image-2.1-Turbo-Uncensored AD-Q4_K GGUF checkpoint.",
    },
    {
        "id": "atomic_turbo_uc_q8", "role": "diffusion",
        "title": "Qwen-Image 2.1 Turbo Uncensored · AtomicChat Q8_0 GGUF (8.71 GB)",
        "filename": "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf",
        "sha256": "", "size_hint": 8_710_000_000, "license": "Apache-2.0", "license_url": QWEN_REPO_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF", "default": False,
        "description": "AtomicChat Qwen-Image-2.1-Turbo-Uncensored Q8_0 GGUF checkpoint.",
    },
    # --- Text Encoders: INT8-ConvRot Safetensors & Full Quantized Qwen3-VL-8B GGUF Ladder ---
    {
        "id": "heretic_uc_text_encoder_int8_convrot", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Heretic INT8-ConvRot Safetensors (9.35 GB · no mmproj needed · default)",
        "filename": "qwen3vl_8b_int8_convrot_heretic.safetensors",
        "url": "https://huggingface.co/Nic3301/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot/resolve/main/qwen3vl_8b_int8_convrot_heretic.safetensors",
        "sha256": "", "size_hint": 9_350_000_000, "license": "Apache-2.0", "license_url": QWEN_HERETIC_CONVROT_REPO_URL,
        "source": "Nic3301/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot", "default": True,
        "description": "Refusal-abliterated (Uncensored) Qwen3-VL 8B INT8-ConvRot text encoder with embedded BF16 vision tower. Works in all workflows including Close-up without needing mmproj.",
    },
    {
        "id": "comfy_text_encoder_int8_convrot", "role": "text_encoder",
        "title": "Qwen3-VL 8B · Comfy-Org INT8-ConvRot Safetensors (8.71 GB · no mmproj needed)",
        "filename": "qwen3vl_8b_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "sha256": "", "size_hint": 8_710_000_000, "license": "Apache-2.0", "license_url": QWEN_VAE_REPO_URL,
        "source": "Comfy-Org/Qwen-Image-2.1", "default": False,
        "description": "Official Comfy-Org INT8-ConvRot Qwen3-VL 8B text encoder with embedded BF16 vision weights (no separate mmproj required).",
    },
    {
        "id": "unsloth_text_encoder_int8_convrot", "role": "text_encoder",
        "title": "Qwen-Image 2.1 Text Encoder · Unsloth INT8-ConvRot Safetensors (9.35 GB · no mmproj needed)",
        "filename": "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
        "url": "https://huggingface.co/unsloth/Qwen-Image-2.1-FP8/resolve/main/Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
        "sha256": "", "size_hint": 9_350_000_000, "license": "Apache-2.0", "license_url": "https://huggingface.co/unsloth/Qwen-Image-2.1-FP8",
        "source": "unsloth/Qwen-Image-2.1-FP8", "default": False,
        "description": "Unsloth INT8-ConvRot text encoder with embedded vision tower; works directly without mmproj.",
    },
    {
        "id": "sakura_uc_text_encoder_q8", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Sakura TextEncoder 8-bit Q8_0 GGUF (8.11 GB)",
        "filename": "Sakura-TextEncoder-Qwen3VL-8B-Uncensored-8bit-8.11GiB.gguf",
        "url": "https://huggingface.co/webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Sakura-TextEncoder-Qwen3VL-8B-Uncensored-8bit-8.11GiB.gguf",
        "sha256": "", "size_hint": 8_110_000_000, "license": "Apache-2.0", "license_url": QWEN_SAKURA_UNCENSORED_REPO_URL,
        "source": "webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF", "default": False,
        "description": "Heretic-ablated Uncensored 8-bit Qwen3-VL 8B GGUF text encoder from the Sakura Qwen-Image 2.1 Turbo collection.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q8_0", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q8_0 GGUF (8.71 GB · highest quantized GGUF)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q8_0.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q8_0.gguf",
        "sha256": "", "size_hint": 8_710_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 8-bit Q8_0 GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q6_k", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q6_K GGUF (6.73 GB · high fidelity)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf",
        "sha256": "", "size_hint": 6_730_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 6-bit Q6_K GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q5_k_m", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q5_K_M GGUF (5.85 GB · balanced 5-bit)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q5_K_M.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q5_K_M.gguf",
        "sha256": "", "size_hint": 5_850_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 5-bit Q5_K_M GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q5_k_s", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q5_K_S GGUF (5.72 GB)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q5_K_S.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q5_K_S.gguf",
        "sha256": "", "size_hint": 5_720_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 5-bit Q5_K_S GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q4_k_m", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q4_K_M GGUF (5.03 GB · 12 GB GPU recommended)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf",
        "sha256": "", "size_hint": 5_030_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 4-bit Q4_K_M GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q4_k_s", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q4_K_S GGUF (4.80 GB)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_S.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_S.gguf",
        "sha256": "", "size_hint": 4_800_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 4-bit Q4_K_S GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_iq4_xs", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 IQ4_XS GGUF (4.59 GB)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.IQ4_XS.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.IQ4_XS.gguf",
        "sha256": "", "size_hint": 4_590_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in IQ4_XS GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q3_k_l", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q3_K_L GGUF (4.43 GB)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q3_K_L.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q3_K_L.gguf",
        "sha256": "", "size_hint": 4_430_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 3-bit Q3_K_L GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q3_k_m", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q3_K_M GGUF (4.12 GB)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q3_K_M.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q3_K_M.gguf",
        "sha256": "", "size_hint": 4_120_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in 3-bit Q3_K_M GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_q2_k", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 Q2_K GGUF (3.28 GB · low VRAM)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.Q2_K.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q2_K.gguf",
        "sha256": "", "size_hint": 3_280_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Uncensored Qwen3-VL-8B-Instruct abliterated-v2 text encoder in compact 2-bit Q2_K GGUF.",
    },
    {
        "id": "qwen3vl_abliterated_v2_f16", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Abliterated-v2 F16 GGUF (16.40 GB · unquantized)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.f16.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.f16.gguf",
        "sha256": "", "size_hint": 16_400_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Unquantized 16-bit F16 GGUF text encoder of Qwen3-VL-8B-Instruct-abliterated-v2.",
    },
    {
        "id": "qwen3vl_heretic_i1_q6_k", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Heretic imatrix i1-Q6_K GGUF (6.73 GB · mradermacher)",
        "filename": "Qwen3-VL-8B-Instruct-heretic.i1-Q6_K.gguf",
        "url": "https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF/resolve/main/Qwen3-VL-8B-Instruct-heretic.i1-Q6_K.gguf",
        "sha256": "", "size_hint": 6_730_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_HERETIC_I1_REPO_URL,
        "source": "mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF", "default": False,
        "description": "Weighted imatrix i1-Q6_K GGUF of Qwen3-VL-8B-Instruct-Heretic.",
    },
    {
        "id": "qwen3vl_heretic_i1_q4_k_m", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Heretic imatrix i1-Q4_K_M GGUF (5.03 GB · mradermacher)",
        "filename": "Qwen3-VL-8B-Instruct-heretic.i1-Q4_K_M.gguf",
        "url": "https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF/resolve/main/Qwen3-VL-8B-Instruct-heretic.i1-Q4_K_M.gguf",
        "sha256": "", "size_hint": 5_030_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_HERETIC_I1_REPO_URL,
        "source": "mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF", "default": False,
        "description": "Weighted imatrix i1-Q4_K_M GGUF of Qwen3-VL-8B-Instruct-Heretic.",
    },
    {
        "id": "qwen3vl_heretic_i1_q3_k_s", "role": "text_encoder",
        "title": "Qwen3-VL 8B Uncensored · Heretic imatrix i1-Q3_K_S GGUF (3.77 GB · mradermacher)",
        "filename": "Qwen3-VL-8B-Instruct-heretic.i1-Q3_K_S.gguf",
        "url": "https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF/resolve/main/Qwen3-VL-8B-Instruct-heretic.i1-Q3_K_S.gguf",
        "sha256": "", "size_hint": 3_770_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_HERETIC_I1_REPO_URL,
        "source": "mradermacher/Qwen3-VL-8B-Instruct-Heretic-i1-GGUF", "default": False,
        "description": "Weighted imatrix i1-Q3_K_S GGUF of Qwen3-VL-8B-Instruct-Heretic for lower VRAM budgets.",
    },
    {
        "id": "bartowski_qwen3vl_q8_0", "role": "text_encoder",
        "title": "Qwen3-VL 8B Instruct · Bartowski Q8_0 GGUF (8.71 GB · standard imatrix)",
        "filename": "Qwen3-VL-8B-Instruct-Q8_0.gguf",
        "url": "https://huggingface.co/bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q8_0.gguf",
        "sha256": "", "size_hint": 8_710_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_BARTOWSKI_REPO_URL,
        "source": "bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF", "default": False,
        "description": "Standard Qwen3-VL-8B-Instruct 8-bit Q8_0 GGUF text encoder quantized by bartowski.",
    },
    {
        "id": "bartowski_qwen3vl_q6_k", "role": "text_encoder",
        "title": "Qwen3-VL 8B Instruct · Bartowski Q6_K GGUF (6.73 GB · standard imatrix)",
        "filename": "Qwen3-VL-8B-Instruct-Q6_K.gguf",
        "url": "https://huggingface.co/bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q6_K.gguf",
        "sha256": "", "size_hint": 6_730_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_BARTOWSKI_REPO_URL,
        "source": "bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF", "default": False,
        "description": "Standard Qwen3-VL-8B-Instruct 6-bit Q6_K GGUF text encoder quantized by bartowski.",
    },
    {
        "id": "unsloth_qwen3vl_q5_k_m", "role": "text_encoder",
        "title": "Qwen3-VL 8B Instruct · Unsloth Q5_K_M GGUF (5.85 GB)",
        "filename": "Qwen3-VL-8B-Instruct-Q5_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q5_K_M.gguf",
        "sha256": "", "size_hint": 5_850_000_000, "license": "Apache-2.0", "license_url": "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF",
        "source": "unsloth/Qwen3-VL-8B-Instruct-GGUF", "default": False,
        "description": "Unsloth 5-bit Q5_K_M GGUF build of Qwen3-VL-8B-Instruct.",
    },
    # --- Vision Projectors (Optional — ONLY for GGUF Text Encoders) ---
    {
        "id": "qwen3vl_abliterated_v2_mmproj_f16", "role": "vision_projector",
        "title": "Qwen3-VL 8B Abliterated-v2 Vision Projector · F16 GGUF (1.16 GB · optional, GGUF only)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf",
        "sha256": "", "size_hint": 1_160_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Optional F16 GGUF vision projector for Qwen3-VL-8B GGUF text encoders. Not needed for Safetensors/ConvRot encoders.",
    },
    {
        "id": "qwen3vl_abliterated_v2_mmproj_q8_0", "role": "vision_projector",
        "title": "Qwen3-VL 8B Abliterated-v2 Vision Projector · Q8_0 GGUF (752 MB · optional, GGUF only)",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-Q8_0.gguf",
        "url": "https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-Q8_0.gguf",
        "sha256": "", "size_hint": 752_000_000, "license": "Apache-2.0", "license_url": QWEN3VL_ABLITERATED_V2_REPO_URL,
        "source": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF", "default": False,
        "description": "Compact 8-bit Q8_0 GGUF vision projector for Qwen3-VL-8B GGUF text encoders.",
    },
    {
        "id": "atomic_turbo_uc_mmproj", "role": "vision_projector", "title": "Qwen3-VL 8B Vision Projector · F16 GGUF (1.16 GB · optional, GGUF only)",
        "filename": "mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf",
        "url": "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf",
        "sha256": "", "size_hint": 1_160_000_000, "license": "Apache-2.0", "license_url": QWEN_REPO_URL,
        "source": "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF", "default": False,
        "description": "Optional GGUF vision projector; only used when a GGUF text encoder is selected. Not needed for Safetensors/ConvRot encoders.",
    },
    # --- VAE Models ---
    {
        "id": "madebyollin_texture_fix_vae_bf16", "role": "vae", "title": "Qwen-Image 2.1 Texture-Fix VAE · madebyollin BF16 (676 MB · removes fine grid · default)",
        "filename": "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
        "url": "https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/resolve/main/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
        "sha256": "", "size_hint": 675_509_688, "license": "Qwen Research License", "license_url": "https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1",
        "source": "madebyollin/texture-fix-vae-for-qwen-image-2.1", "default": True,
        "description": "Finetuned Qwen-Image 2.1 VAE decoder by madebyollin that removes the fine checkerboard grid artifact on detailed photographic textures.",
    },
    {
        "id": "atomic_qwen_vae_bf16", "role": "vae", "title": "Qwen-Image 2.1 VAE · Comfy-Org Official BF16 Safetensors (676 MB)",
        "filename": "qwen_image_2.1_vae_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
        "sha256": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9", "size_hint": 675_509_688, "license": "Qwen Research License", "license_url": QWEN_VAE_REPO_URL,
        "source": "Comfy-Org/Qwen-Image-2.1 (identical VAE documented by AtomicChat)", "default": False,
        "description": "Official Qwen-Image-2.1 BF16 VAE compatible with all Qwen-Image 2.1 Turbo checkpoints.",
    },
]
MODEL_CATALOG_BY_ID = {item["id"]: item for item in MODEL_CATALOG}


def model_download_sources(meta: dict[str, Any]) -> tuple[str, ...]:
    """Return official source first, plus an opt-in-compatible Hugging Face mirror fallback.

    The mirror is not operated by Hugging Face. Pinned files still must match their exact
    upstream SHA-256; unpinned files are size-checked and locally hashed as documented.
    """
    primary = str(meta.get("url", "")).strip()
    if not primary:
        return ()
    sources = [primary]
    parsed = urllib.parse.urlsplit(primary)
    if parsed.hostname == "huggingface.co":
        mirror = urllib.parse.urlunsplit(("https", "hf-mirror.com", parsed.path, parsed.query, ""))
        if mirror != primary:
            sources.append(mirror)
    return tuple(sources)


MODEL_ROLE_ORDER = ["diffusion", "text_encoder", "vision_projector", "vae"]
MODEL_ROLE_LABELS = {
    "diffusion": "Image model",
    "text_encoder": "Text encoder",
    "vision_projector": "Vision projector",
    "vae": "VAE",
}
MODEL_ROLE_REQUIRED = {"diffusion", "text_encoder", "vae"}
MODEL_DEFAULTS = {item["role"]: item["id"] for item in MODEL_CATALOG if item.get("default")}
MODEL_DEFAULTS.setdefault("vision_projector", "none")


def _download_ssl_context():
    """Prefer the bundled Mozilla CA bundle, falling back to Python's trust store."""
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _sha256_file(path: Path, cancel_event=None, progress_callback=None) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise Aborted()
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if progress_callback:
                progress_callback(total)
    return digest.hexdigest()


def _human_gb(value: int) -> str:
    return f"{value / 1_000_000_000:.2f} GB"
