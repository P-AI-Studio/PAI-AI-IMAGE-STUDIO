# -*- coding: utf-8 -*-
"""
Local native stable-diffusion.cpp and Real-ESRGAN inference engine, GPU discovery, and Job runner.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageChops, ImageOps

from .constants import (
    APP_VERSION,
    ENHANCE_IMG2IMG_STRENGTH,
    ENHANCE_TILE_IMG2IMG_STRENGTH,
    MAX_INFERENCE_SIDE,
    MAX_OUTPUT_SIDE,
    MODEL_ROLE_LABELS,
    QWEN_DETAIL_PROFILES,
    QWEN_DETAIL_PROFILE_PROMPTS,
    QWEN_DETAIL_PROFILE_TILE_LIMITS,
    REALESRGAN_HF_NCNN_BASE,
    REALESRGAN_MODEL_NAMES,
    REALESRGAN_MODEL_NATIVE_SCALES,
    REALESRGAN_CATALOG_BY_ID,
    UPSCAYL_NCNN_URL,
    UPSCAYL_NCNN_SHA256,
    UPSCAYL_NCNN_ARCHIVE_SIZE,
    UPSCAYL_NCNN_MODEL_BASE_URL,
    realesrgan_native_scale,
    REALESRGAN_NCNN_URLS,
    _download_ssl_context,
    _realesrgan_model_assets_present,
    _sha256_file,
    apply_qwen_detail_profile_prompt,
    qwen_detail_profile_values,
    qwen_detail_refinement_metadata,
    qwen_enhance_reconstruction_strengths,
    restore_realesrgan_model_pair_from_archive,
)
from .grounding import (
    crop_closeup_from_bbox, detect_target_bbox_local, expand_qwen3vl_target_descriptions,
    locate_target_bbox_non_gguf,
    resolve_closeup_grounding_gguf_pair,
    translate_prompt_non_gguf_fallback,
    translate_prompt_to_english as translate_prompt_via_local_encoder,
    _shutdown_qwen3vl_grounder,
)
from .grounding_strategy import GroundingResult, run_grounding_strategy
from .performance import enhance_detail_mode_for_ultimate
from .image_ops import (
    Aborted,
    build_square_outpaint_inputs,
    composite_square_outpaint_result,
    crop_closeup_square,
    clean_iso_grain_and_chroma_noise,
    crossfade_tile_seams,
    blend_ultimate_tile_by_denoise,
    match_tile_color_to_reference,
    mean_difference,
    plan_square_outpaint,
    save_image_with_metadata,
)
from .validation import (
    validate_archive_member_path,
    validate_loopback_url,
    validate_lora_file,
)
from .prompts import (
    apply_identity_lock,
    apply_watermark_removal_prompt,
    build_closeup_prompt,
    build_enhance_prompt,
    build_square_core_enhance_prompt,
    build_square_prompt,
)
from .workspace import (
    APP_DIR,
    CACHE_DIR,
    CLOSEUPS_DIR,
    DATA_ROOT,
    EDITED_DIR,
    GENERATED_DIR,
    MODEL_DIR,
    OUTPAINT_DIR,
    OUTPUT_DIR,
    _model_role_mismatch,
    _text_encoder_embeds_vision,
    prepare_native_prompt,
    install_directory_transactionally,
)


# Qwen-Image 2.1 Turbo is distilled for this exact eight-step Euler schedule.
# Alternate counts/sigma interpolation are not validated for the released checkpoint.
QWEN_TURBO_SIGMAS = (
    1.0, 0.978453, 0.95418, 0.926626, 0.89508,
    0.845148, 0.704534, 0.414568, 0.0,
)


def qwen_turbo_sampling_params() -> dict[str, Any]:
    """Return the checkpoint-required Turbo sampling parameters (8 steps, CFG 1, Euler)."""
    return {
        "sample_steps": 8,
        "custom_sigmas": list(QWEN_TURBO_SIGMAS),
        "guidance": {"txt_cfg": 1.0},
    }


def qwen_turbo_tile_sampling_params(steps: int = 8, strength: Any = None, compensate_img2img_steps: bool = False) -> dict[str, Any]:
    """Return the Turbo sampling parameters (CFG 1, Euler), interpolating strictly along the non-zero distilled
    QWEN_TURBO_SIGMAS knots (1.0 .. 0.414568) and keeping the final step as a single clean jump (0.414568 -> 0.0)
    so the distilled Turbo denoiser is never evaluated at unvalidated sigmas in (0, 0.414568)."""
    target_steps = max(1, min(100, int(steps or 8)))
    schedule_steps = target_steps
    eff_s: Optional[float] = None
    if compensate_img2img_steps and strength is not None:
        eff_s = max(0.05, min(1.0, float(strength)))
        if eff_s < 0.999:
            schedule_steps = max(target_steps, int(round(target_steps / eff_s)))
            while int(schedule_steps * eff_s) < target_steps and schedule_steps < 1200:
                schedule_steps += 1
            while schedule_steps > target_steps and int((schedule_steps - 1) * eff_s) >= target_steps:
                schedule_steps -= 1
    base = list(QWEN_TURBO_SIGMAS)
    if schedule_steps == len(base) - 1:
        sigmas = base
    else:
        # Interpolate the first (schedule_steps - 1) intervals across the 7 valid non-zero knots base[0..7]
        # (from 1.0 down to 0.414568), then jump directly from 0.414568 to 0.0 on the final step.
        non_zero_knots = base[:-1]
        n_nz_intervals = len(non_zero_knots) - 1
        denom = float(max(1, schedule_steps - 1))
        sigmas = []
        for i in range(schedule_steps):
            pos = (i / denom) * n_nz_intervals
            idx = min(n_nz_intervals - 1, int(pos))
            frac = pos - idx
            val = non_zero_knots[idx] + frac * (non_zero_knots[idx + 1] - non_zero_knots[idx])
            sigmas.append(round(max(float(non_zero_knots[-1]), val), 6))
        sigmas[0] = float(base[0])
        sigmas[-1] = float(non_zero_knots[-1])
        sigmas.append(0.0)
    return {
        "sample_steps": schedule_steps,
        "custom_sigmas": sigmas,
        "guidance": {"txt_cfg": 1.0},
    }


def plan_qwen_memory_placement(
    *, free_vram_gb: float, total_vram_gb: float, diffusion_gb: float,
    vae_gb: float, component_file_gb: float, text_encoder_on_cpu: bool = False,
    memory_mode: str = "auto", auto_offload_override: bool = False,
) -> dict[str, Any]:
    """Choose explicit component placement; GPU-only mode never silently falls back to CPU."""
    reserve_gb = 2.8
    free_vram_gb = max(0.0, float(free_vram_gb))
    total_vram_gb = max(0.0, float(total_vram_gb))
    diffusion_gb = max(0.0, float(diffusion_gb))
    vae_gb = max(0.0, float(vae_gb))
    component_file_gb = max(0.0, float(component_file_gb))
    mode = str(memory_mode or "auto").strip().lower()

    if mode == "full_gpu":
        # Strict policy: every inference component is assigned to CUDA. Never let a
        # capacity estimate silently put the text encoder on CPU or enable offload.
        return {
            "activation_reserve_gb": reserve_gb,
            "force_cpu_te_for_capacity": False,
            "text_backend": "cuda",
            "offload_to_cpu": False,
            "auto_fit_mode": "off",
        }

    force_cpu_te = bool(
        not text_encoder_on_cpu and (
            free_vram_gb <= 0
            or free_vram_gb < component_file_gb + reserve_gb
        )
    )
    text_backend = "cpu" if (text_encoder_on_cpu or force_cpu_te) else "cuda"
    if auto_offload_override or mode == "low_vram":
        offload_to_cpu = True
    else:
        offload_to_cpu = (
            total_vram_gb < 10.5
            or (total_vram_gb > 0 and diffusion_gb >= total_vram_gb * 0.52)
            or (free_vram_gb > 0 and free_vram_gb < 6.5)
        )
    if force_cpu_te and (
        free_vram_gb <= 0 or free_vram_gb < diffusion_gb + vae_gb + reserve_gb
    ):
        offload_to_cpu = True
    auto_fit_mode = "off" if (
        offload_to_cpu or text_backend == "cpu"
        or (free_vram_gb > 0 and free_vram_gb >= component_file_gb + reserve_gb)
    ) else "on"
    return {
        "activation_reserve_gb": reserve_gb,
        "force_cpu_te_for_capacity": force_cpu_te,
        "text_backend": text_backend,
        "offload_to_cpu": offload_to_cpu,
        "auto_fit_mode": auto_fit_mode,
    }

@dataclass
class Job:
    workflow: str
    input_path: Optional[str] = None
    prompt: str = ""
    output_dir: str = ""
    width: int = 1024
    height: int = 1024
    quality: str = "High"
    seed: int = -1
    seed_mode: str = "Random"
    output_format: str = "png"
    target: str = ""
    auto_zoom: bool = True
    zoom_factor: float = 2.2
    focus_point: Optional[tuple[float, float]] = None
    enhance: bool = True
    side: int = 1024
    save_caption: bool = False
    trigger_word: str = ""
    preserve_composition: bool = True
    lock_identity: bool = True
    batch_index: int = 0
    batch_count: int = 1
    custom: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex, repr=False)

    def validate(self, require_existing_input: bool = False) -> "Job":
        from .validation import validate_job
        return validate_job(self, require_existing_input=require_existing_input)


def _find_nvidia_smi() -> Optional[Path]:
    """Find nvidia-smi from the installed NVIDIA driver (no CUDA Toolkit required)."""
    found = shutil.which("nvidia-smi")
    if found:
        return Path(found).resolve()
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files")))
        candidates = [
            program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def detect_nvidia_gpus(timeout: float = 5.0) -> list[dict[str, Any]]:
    """Return NVIDIA GPUs and VRAM reported by the installed driver.

    This deliberately does not use torch. The end-user package needs only the NVIDIA
    display driver plus the bundled CUDA runtime. An empty result disables CPU fallback.
    """
    smi = _find_nvidia_smi()
    if not smi:
        return []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        result = subprocess.run(
            [str(smi), "--query-gpu=index,name,memory.total,memory.free,driver_version,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False, creationflags=creationflags,
        )
        if result.returncode != 0:
            return []
        gpus: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                index = int(parts[0])
                total_mb = int(parts[2])
                free_mb = int(parts[3])
            except (ValueError, TypeError):
                continue
            gpus.append({
                "index": index,
                "name": parts[1],
                "memory_total_mb": total_mb,
                "memory_free_mb": free_mb,
                "driver_version": parts[4] if len(parts) > 4 else "unknown",
                "uuid": parts[5] if len(parts) > 5 else "",
            })
        return sorted(gpus, key=lambda gpu: gpu["index"])
    except (OSError, subprocess.SubprocessError):
        return []

def _parse_sdcpp_preview_status(status_obj: dict[str, Any], max_side: int = 512) -> tuple[Optional[Image.Image], int, int, int]:
    """Read the native sdcpp async-job preview frame and sampler counters safely.

    Current sd-server returns ``preview: {step, total_steps, pass, b64_json}``.
    A few older or locally patched builds use preview_info/preview_base64 instead,
    so those shapes are accepted too. No tensor weights or model assets are read here.
    """
    if not isinstance(status_obj, dict):
        return None, 0, 0, 0

    preview = status_obj.get("preview")
    if not isinstance(preview, dict):
        preview = status_obj.get("preview_info")
    if not isinstance(preview, dict):
        preview = {}

    def _positive_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0

    step = _positive_int(preview.get("step", status_obj.get("preview_step", 0)))
    total = _positive_int(preview.get("total_steps", status_obj.get("preview_total_steps", 0)))
    sample_pass = _positive_int(preview.get("pass", status_obj.get("preview_pass", 0)))

    encoded = None
    for container in (preview, status_obj):
        for key in ("b64_json", "image_base64", "preview_base64", "preview_b64", "image", "preview_image"):
            candidate = container.get(key) if isinstance(container, dict) else None
            if isinstance(candidate, dict):
                candidate = candidate.get("b64_json") or candidate.get("base64") or candidate.get("data")
            if isinstance(candidate, str) and candidate.strip():
                encoded = candidate.strip()
                break
        if encoded:
            break

    if not encoded:
        return None, step, total, sample_pass
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=False)
        with Image.open(io.BytesIO(raw)) as frame:
            small = ImageOps.exif_transpose(frame).convert("RGB")
            limit = max(32, int(max_side))
            small.thumbnail((limit, limit), Image.Resampling.BICUBIC)
            if max(small.size) < limit:
                scale = limit / max(1, max(small.size))
                target = (max(1, int(round(small.width * scale))), max(1, int(round(small.height * scale))))
                small = small.resize(target, Image.Resampling.BICUBIC)
            return small.copy(), step, total, sample_pass
    except Exception:
        return None, step, total, sample_pass


class LocalEngine:
    """Bundled native Qwen-Image runtime; diffusion and VAE must use the selected NVIDIA CUDA GPU."""
    def __init__(self, log_callback=None, status_callback=None):
        self.log = log_callback or (lambda message: None)
        self.status_callback = status_callback or (lambda message: None)
        self.force_stop = threading.Event()
        self.stop_after_current = threading.Event()
        self.output_durations: dict[str, float] = {}
        self.model_path = str(MODEL_DIR)
        self.lora_path = ""
        self.lora_enabled = False
        self.lora_strength = 1.0
        self.text_encoder_on_cpu = False
        self.gpu_index = 0
        self.gpu_memory_mode = "auto"
        self.auto_offload_override = False
        self._last_offload_to_cpu = False
        self.gpu_info: Optional[dict[str, Any]] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.server_base_url = ""
        self.server_log_path: Optional[Path] = None
        self._server_log_handle = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_profile_signature: Optional[str] = None
        self._loaded_parts: dict[str, Path] = {}
        self.latest_preview_image: Optional[Image.Image] = None
        self.latest_preview_revision = 0
        self._latest_preview_marker: Optional[tuple[Any, ...]] = None
        # During per-tile inference, keep showing the last complete canvas instead of
        # publishing native sampler previews that make the canvas appear as a mosaic.
        self._suppress_live_preview = False
        self._prompt_translation_cache: dict[tuple[str, str], str] = {}
        self.latest_preview_step = 0
        self.latest_preview_total_steps = 0
        self.latest_preview_pass = 0
        self.live_preview_method = "tae"
        self.live_preview_interval = 1
        # Context is set around nested Qwen tile refinements so UI progress can distinguish
        # first-pass diffusion from Ultimate/detail-tile inference.
        self._active_pipeline_context = ""
        self._lock = threading.RLock()
        # Set by the UI worker while a cancellable Close-up job is active.
        self.grounding_review_callback = None

    def _locate_closeup_target(
        self,
        image: Image.Image,
        target: str,
        model_path: str | Path,
        mmproj_path: str | Path,
        profile: str = "balanced",
    ) -> GroundingResult:
        """Run the bounded, cancellable semantic grounding strategy for Close-up."""
        def locator(region, description, log_callback=None, timeout_seconds=240.0):
            return detect_target_bbox_local(
                region, description, None, model_path=model_path,
                mmproj_path=mmproj_path, timeout_seconds=timeout_seconds, detailed=True,
                shutdown_after_request=False,
            )

        def expand_target(description, timeout_seconds):
            return expand_qwen3vl_target_descriptions(
                description, model_path=model_path, mmproj_path=mmproj_path,
                timeout_seconds=timeout_seconds, max_candidates=3, shutdown_after_request=False,
            )

        try:
            res = run_grounding_strategy(
                image, target, locator=locator, profile=profile,
                should_cancel=self.force_stop.is_set, log_callback=self.log,
                expand_target=expand_target,
            )
            if (res.status != "found" or res.bbox_xyxy is None) and res.status != "cancelled":
                non_gguf_res = locate_target_bbox_non_gguf(image, target, log_callback=self.log)
                if non_gguf_res.status == "found" and non_gguf_res.bbox_xyxy is not None:
                    return GroundingResult(
                        status="found", bbox_xyxy=non_gguf_res.bbox_xyxy, target_text=str(target or ""),
                        attempts=max(1, res.attempts), reason_code="ok",
                        diagnostic_summary="Target located via local visual target grounding fallback.",
                        elapsed_ms=res.elapsed_ms, candidates=(non_gguf_res.bbox_xyxy,),
                    )
            return res
        finally:
            # Multiple short grounding requests (expansion + candidates + tiles) should reuse
            # one local server process, then release its GPU resources at the strategy boundary.
            _shutdown_qwen3vl_grounder()

    def configure(self, model_path: str, lora_path: str = "", lora_enabled: bool = False,
                  lora_strength: float = 1.0, text_encoder_on_cpu: bool = True, gpu_index: int = 0,
                  gpu_memory_mode: str = "auto"):
        new_model_path = str(Path(model_path).expanduser().resolve()) if model_path else ""
        new_memory_mode = str(gpu_memory_mode or "auto").strip().lower()
        if new_memory_mode == "fast":  # migrate in-memory callers from v1.7.3
            new_memory_mode = "full_gpu"
        if new_memory_mode not in {"auto", "full_gpu", "low_vram"}:
            new_memory_mode = "auto"
        with self._lock:
            new_profile_signature = self._model_profile_signature(Path(new_model_path)) if new_model_path else ""
            profile_changed = bool(new_profile_signature and new_profile_signature != self._loaded_profile_signature)
            path_changed = bool(new_model_path and new_model_path != self.model_path)
            memory_mode_changed = new_memory_mode != self.gpu_memory_mode
            hardware_changed = (bool(text_encoder_on_cpu) != self.text_encoder_on_cpu) or (int(gpu_index) != self.gpu_index)
            new_lora_path = str(Path(lora_path).expanduser().resolve()) if lora_path else ""
            lora_runtime_changed = new_lora_path != self.lora_path or bool(lora_enabled) != self.lora_enabled
            if self.server_process is not None and (path_changed or profile_changed or hardware_changed or memory_mode_changed or lora_runtime_changed):
                self.unload_model()
            if path_changed or profile_changed or memory_mode_changed or hardware_changed:
                self.auto_offload_override = False
            self.model_path = new_model_path
            self.lora_path = new_lora_path
            self.lora_enabled = bool(lora_enabled)
            self.lora_strength = max(0.0, min(2.0, float(lora_strength)))
            self.text_encoder_on_cpu = bool(text_encoder_on_cpu)
            self.gpu_index = int(gpu_index)
            self.gpu_memory_mode = new_memory_mode

    @staticmethod
    def _is_process_alive(proc) -> bool:
        return proc is not None and proc.poll() is None

    @staticmethod
    def _human_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"

    @staticmethod
    def _windows_cuda_dependency_dirs() -> list[Path]:
        """Known Windows DLL search folders, including an existing CUDA Toolkit/PATH install.

        This is deliberately a bounded search: never walk the whole drive. The bundled
        runtime remains preferred, but a matching CUDA 12 DLL already installed in Windows
        or discoverable on PATH should satisfy the dependency without another download.
        """
        if os.name != "nt":
            return []
        candidates: list[Path] = []
        candidates.extend(Path(item.strip().strip('"')) for item in os.environ.get("PATH", "").split(os.pathsep) if item.strip())
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        candidates.extend((system_root / "System32", system_root))
        for key, value in os.environ.items():
            if key.upper() == "CUDA_PATH" or key.upper().startswith("CUDA_PATH_V"):
                if value:
                    candidates.append(Path(value) / "bin")
        program_files = Path(os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files")))
        cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
        try:
            candidates.extend(version_dir / "bin" for version_dir in cuda_root.iterdir() if version_dir.is_dir())
        except OSError:
            pass
        # Some Python distributions install split NVIDIA runtime wheels outside PATH.
        for site_root in (Path(sys.prefix) / "Lib" / "site-packages", Path(sys.prefix) / "lib" / "site-packages"):
            candidates.extend((
                site_root / "nvidia" / "cuda_runtime" / "bin",
                site_root / "nvidia" / "cublas" / "bin",
            ))
        result: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
                key = str(resolved).casefold()
                if resolved.is_dir() and key not in seen:
                    seen.add(key)
                    result.append(resolved)
            except OSError:
                continue
        return result

    def _ensure_cuda_runtime_payload(self, runtime_root: Path) -> None:
        """Ensure the CUDA backend DLL and CUDA 12 runtime dependencies are available.

        stable-diffusion.cpp publishes the CUDA backend binary and cudart dependency
        archive separately. The main *cuda12* ZIP alone is not a portable CUDA runtime.
        The upstream stable-diffusion.cpp release pins the main ZIP but does not publish a
        SHA-256 for the companion CUDA ZIP, so we pin its release-tag URL, validate ZIP
        CRCs, and require all three expected DLLs before accepting it.
        """
        if os.name != "nt":
            return
        runtime_root = runtime_root.resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)

        def find_runtime_file(filename: str) -> Optional[Path]:
            try:
                return next((p for p in runtime_root.rglob(filename) if p.is_file()), None)
            except OSError:
                return None

        system_cuda_dirs = self._windows_cuda_dependency_dirs()

        def find_file(filename: str) -> Optional[Path]:
            # Prefer the pinned package, but reuse dependencies already installed in Windows.
            found = find_runtime_file(filename)
            if found is not None:
                return found
            for directory in system_cuda_dirs:
                candidate = directory / filename
                if candidate.is_file():
                    return candidate.resolve()
            return None

        # ggml-cuda.dll is an engine-specific plugin, not a generic system dependency.
        cuda_backend = find_runtime_file("ggml-cuda.dll")
        if cuda_backend is None:
            raise RuntimeError(
                "The bundled native engine is missing ggml-cuda.dll; this runtime is likely CPU-only. "
                "Rebuild the package using the official CUDA 12 stable-diffusion.cpp runtime."
            )

        required_dlls = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
        found_in_windows = [
            name for name in required_dlls
            if find_runtime_file(name) is None and any((directory / name).is_file() for directory in system_cuda_dirs)
        ]
        def _stage_beside_backend() -> None:
            backend_dir = cuda_backend.parent
            for dll_name in required_dlls:
                src_dll = find_file(dll_name)
                dst_dll = backend_dir / dll_name
                if src_dll is not None and src_dll.is_file() and not dst_dll.is_file():
                    try:
                        shutil.copy2(src_dll, dst_dll)
                    except OSError:
                        pass

        missing = [name for name in required_dlls if find_file(name) is None]
        if not missing:
            _stage_beside_backend()
            if found_in_windows:
                self.log("Reusing CUDA 12 dependencies already installed in Windows/PATH; no CUDA companion download needed: " + ", ".join(found_in_windows) + ".")
            else:
                self.log("CUDA 12 dependencies are already present in the app runtime; no download needed.")
            return
        if os.name != "nt":
            raise RuntimeError("Windows CUDA 12 dependencies were not found in this runtime.")

        runtime_tag = "master-964-1d0a58a"
        runtime_url = (
            "https://github.com/leejet/stable-diffusion.cpp/releases/download/"
            f"{runtime_tag}/cudart-sd-bin-win-cu12-x64.zip"
        )
        runtime_urls = (
            runtime_url,
            f"https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/{runtime_tag}/cudart-sd-bin-win-cu12-x64.zip/download",
        )
        cache_dir = CACHE_DIR / "runtime-downloads"
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "cudart-sd-bin-win-cu12-x64.zip"
        part = archive.with_suffix(archive.suffix + ".part")

        def validate_archive(path: Path) -> bool:
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    if zf.testzip() is not None:
                        return False
                    names = {Path(name.replace("\\", "/")).name.lower() for name in zf.namelist()}
                    return all(name.lower() in names for name in required_dlls)
            except (OSError, zipfile.BadZipFile, RuntimeError):
                return False

        archive_valid = archive.is_file() and validate_archive(archive)
        if not archive_valid:
            archive.unlink(missing_ok=True)
            free = shutil.disk_usage(runtime_root).free
            if free < 1_200_000_000:
                raise RuntimeError(
                    "About 600 MB of free disk space is required to download CUDA 12 dependencies; "
                    "Free up disk space and try again."
                )
            last_error = None
            for source_index, source_url in enumerate(runtime_urls):
                part.unlink(missing_ok=True)
                for attempt in range(1, 3):
                    if self.force_stop.is_set():
                        raise Aborted()
                    offset = part.stat().st_size if part.exists() else 0
                    headers = {"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    try:
                        self.status_callback("Downloading cudart-sd-bin-win-cu12-x64.zip · preparing CUDA 12 GPU dependencies…")
                        self.log(f"Downloading CUDA dependency companion from source {source_index + 1}/{len(runtime_urls)}: {source_url}")
                        req = urllib.request.Request(source_url, headers=headers)
                        with urllib.request.urlopen(req, timeout=180, context=_download_ssl_context()) as response:
                            status = getattr(response, "status", response.getcode())
                            if offset and status != 206:
                                offset = 0
                            mode = "ab" if offset and status == 206 else "wb"
                            current = offset
                            expected = response.headers.get("Content-Length")
                            try:
                                expected_total = current + int(expected) if expected is not None else 0
                            except (ValueError, TypeError):
                                expected_total = 0
                            with part.open(mode) as handle:
                                while True:
                                    if self.force_stop.is_set():
                                        handle.flush(); os.fsync(handle.fileno())
                                        raise Aborted()
                                    block = response.read(4 * 1024 * 1024)
                                    if not block:
                                        break
                                    handle.write(block)
                                    current += len(block)
                                    if expected_total:
                                        self.status_callback(f"Downloading cudart-sd-bin-win-cu12-x64.zip · {current/1e6:.0f} / {expected_total/1e6:.0f} MB")
                                    else:
                                        self.status_callback(f"Downloading cudart-sd-bin-win-cu12-x64.zip · {current/1e6:.0f} MB downloaded")
                                handle.flush(); os.fsync(handle.fileno())
                        if part.stat().st_size < 100_000_000:
                            raise RuntimeError("The CUDA 12 dependency file is too small or the server response is not a valid ZIP archive.")
                        if not validate_archive(part):
                            raise RuntimeError("The CUDA 12 dependency ZIP is incomplete or is missing required DLLs.")
                        os.replace(part, archive)
                        archive_valid = True
                        break
                    except Aborted:
                        raise
                    except Exception as exc:
                        last_error = exc
                        self.log(f"CUDA companion download failed from {source_url}: {exc}")
                        part.unlink(missing_ok=True)
                        if attempt < 2:
                            time.sleep(float(attempt))
                if archive_valid:
                    break
            if not archive.is_file() or not archive_valid:
                raise RuntimeError(
                    "Failed to download the CUDA 12 companion runtime. The official URL for this engine version must be available. "
                    f"Details: {last_error}"
                ) from last_error
        companion_sha256 = _sha256_file(archive, self.force_stop)
        self.log(f"CUDA companion archive SHA-256 (computed for audit; upstream does not publish a pin): {companion_sha256}")

        # Copy only the dependency DLLs into a dedicated folder. Their directory will
        # be prepended to PATH before starting sd-server.exe. Never execute ZIP paths.
        cuda_dir = runtime_root / "cudart"
        cuda_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            selected: dict[str, str] = {}
            for name in zf.namelist():
                base = Path(name.replace("\\", "/")).name.lower()
                if base in {dll.lower() for dll in required_dlls} and base not in selected:
                    selected[base] = name
            for dll in required_dlls:
                member = selected.get(dll.lower())
                if not member:
                    raise RuntimeError(f"File {dll} was not found in the official CUDA runtime.")
                with zf.open(member, "r") as source_handle, (cuda_dir / dll).open("wb") as out_handle:
                    shutil.copyfileobj(source_handle, out_handle, length=4 * 1024 * 1024)
        still_missing = [name for name in required_dlls if find_file(name) is None]
        if still_missing:
            raise RuntimeError("CUDA 12 dependencies are incomplete after extraction: " + ", ".join(still_missing))
        manifest = {
            "source_url": runtime_url,
            "release_tag": runtime_tag,
            "archive_sha256_computed": companion_sha256,
            "upstream_sha256_pinned": False,
            "required_dlls": list(required_dlls),
        }
        (cuda_dir / "runtime-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _stage_beside_backend()
        self.log("CUDA12 runtime dependencies verified: cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll.")

    def _runtime_executable(self) -> Path:
        """Find the bundled official stable-diffusion.cpp sd-server or download/verify it in user data.

        The release installer normally bundles this runtime. This fallback exists for
        developer/source runs and repairs incomplete extractions without asking the end
        user to install Python, CMake, Visual Studio or CUDA Toolkit.
        """
        runtime_tag = "master-964-1d0a58a"
        runtime_url = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip"
        runtime_urls = (
            runtime_url,
            "https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip/download",
        )
        runtime_sha256 = "2f08572e143f0b15a764197ea937d219dd70f0cc7eefb30eb84c9dae6a7cba6d"
        runtime_size = 338_016_915
        packaged_candidates = [
            APP_DIR / "runtime" / "sd-server.exe",
            APP_DIR / "runtime" / "bin" / "Release" / "sd-server.exe",
            APP_DIR / "bin" / "Release" / "sd-server.exe",
            APP_DIR / "sd-server.exe",
            DATA_ROOT / "runtime" / runtime_tag / "sd-server.exe",
        ]
        runtime_roots = (APP_DIR / "runtime", DATA_ROOT / "runtime" / runtime_tag)

        def payload_root_for(server: Path) -> Path:
            resolved = server.resolve()
            for runtime_root_candidate in runtime_roots:
                try:
                    resolved.relative_to(runtime_root_candidate.resolve())
                    return runtime_root_candidate
                except (ValueError, OSError):
                    continue
            return server.parent

        for candidate in packaged_candidates:
            if candidate.is_file():
                payload_root = payload_root_for(candidate)
                self._ensure_cuda_runtime_payload(payload_root)
                return candidate.resolve()
        for root in runtime_roots:
            if root.is_dir():
                try:
                    found = next((p for p in root.rglob("sd-server.exe") if p.is_file()), None)
                except OSError:
                    found = None
                if found:
                    self._ensure_cuda_runtime_payload(root)
                    return found.resolve()

        # Reuse a runtime from an older extracted P.AI project before downloading it again.
        # This commonly happens when the user unpacks a new ZIP into a new folder: the old
        # engine is still present under Downloads\PAI_AI_IMAGE_STUDIO*\...\data\runtime,
        # but it is outside the new APP_DIR. Search only project-named folders in a few
        # predictable locations; never recursively scan the whole drive.
        search_roots: list[Path] = [APP_DIR.parent, APP_DIR.parent.parent]
        downloads_root = Path.home() / "Downloads"
        if downloads_root not in search_roots:
            search_roots.append(downloads_root)
        program_files_root = Path(os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files")))
        if program_files_root.exists():
            search_roots.append(program_files_root)
        seen_project_roots: set[str] = set()
        for search_root in search_roots:
            try:
                project_roots = [p for p in search_root.glob("PAI_AI_IMAGE_STUDIO*") if p.is_dir()]
                # APP_DIR itself may be the project folder and need not match the glob above.
                if search_root == APP_DIR.parent and APP_DIR.is_dir():
                    project_roots.append(APP_DIR)
            except OSError:
                continue
            for project_root in project_roots:
                try:
                    project_key = str(project_root.resolve()).casefold()
                    if project_key in seen_project_roots:
                        continue
                    seen_project_roots.add(project_key)
                    # Limit recursion to project folders with our recognizable name.
                    found = next((p for p in project_root.glob(f"**/data/runtime/{runtime_tag}/sd-server.exe") if p.is_file()), None)
                    if found is None:
                        found = next((p for p in project_root.glob(f"**/runtime/{runtime_tag}/sd-server.exe") if p.is_file()), None)
                    if found is not None:
                        payload_root = found.parent
                        # If the executable is nested inside the versioned runtime root,
                        # use that root so CUDA DLL validation can find its companion folder.
                        for parent in found.parents:
                            if parent.name == runtime_tag:
                                payload_root = parent
                                break
                        self.log(f"Reusing existing stable-diffusion.cpp runtime from another P.AI installation; skipping sd-master-1d0a58a-bin-win-cuda12-x64.zip: {found.resolve()}")
                        self._ensure_cuda_runtime_payload(payload_root)
                        return found.resolve()
                except (OSError, RuntimeError) as exc:
                    self.log(f"Could not reuse runtime from {project_root}: {exc}")

        # Reuse an independently installed engine if it is already available on PATH.
        # Only bootstrap/download the pinned archive when neither the app, a previous P.AI
        # project folder, nor Windows PATH provides an executable. CUDA compatibility is
        # still validated below.
        path_engine = shutil.which("sd-server.exe") or shutil.which("sd-server")
        if path_engine:
            path = Path(path_engine).resolve()
            self._ensure_cuda_runtime_payload(path.parent)
            self.log(f"Reusing existing native engine found on Windows PATH: {path}")
            return path

        if os.name != "nt":
            raise RuntimeError(
                "The official stable-diffusion.cpp runtime was not found in this Windows x64 package. "
                "For another operating system, package the official runtime for that operating system."
            )

        runtime_root = DATA_ROOT / "runtime" / runtime_tag
        archive_dir = CACHE_DIR / "runtime-downloads"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / "sd-master-1d0a58a-bin-win-cuda12-x64.zip"
        part = archive.with_suffix(archive.suffix + ".part")
        runtime_root.parent.mkdir(parents=True, exist_ok=True)

        # Keep the cached upstream stable-diffusion.cpp archive only when size and SHA-256 match the release pin.
        if archive.is_file():
            try:
                valid = archive.stat().st_size == runtime_size and _sha256_file(archive, self.force_stop) == runtime_sha256
            except Exception:
                valid = False
            if not valid:
                archive.unlink(missing_ok=True)
        if not archive.is_file():
            free = shutil.disk_usage(DATA_ROOT).free
            if free < runtime_size + 1_000_000_000:
                raise RuntimeError(
                    f"Downloading sd-master-1d0a58a-bin-win-cuda12-x64.zip requires about {runtime_size / 1e9:.1f} GB plus extraction space; "
                    "There is not enough free disk space."
                )
            last_error = None
            for source_index, source_url in enumerate(runtime_urls):
                # Start each mirror from a clean part file. SHA-256 and exact-size checks
                # remain mandatory irrespective of which endpoint supplied the bytes.
                part.unlink(missing_ok=True)
                for attempt in range(1, 3):
                    if self.force_stop.is_set():
                        raise Aborted()
                    offset = part.stat().st_size if part.exists() else 0
                    try:
                        self.status_callback("Downloading sd-master-1d0a58a-bin-win-cuda12-x64.zip · 0 / 338 MB")
                        self.log(f"Downloading official stable-diffusion.cpp archive sd-master-1d0a58a-bin-win-cuda12-x64.zip from source {source_index + 1}/{len(runtime_urls)}: {source_url}")
                        req_headers = {"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"}
                        if offset:
                            req_headers["Range"] = f"bytes={offset}-"
                        request = urllib.request.Request(source_url, headers=req_headers)
                        with urllib.request.urlopen(request, timeout=180, context=_download_ssl_context()) as response:
                            status = getattr(response, "status", response.getcode())
                            if offset and status != 206:
                                offset = 0
                            mode = "ab" if offset and status == 206 else "wb"
                            current = offset
                            with part.open(mode) as handle:
                                while True:
                                    if self.force_stop.is_set():
                                        handle.flush(); os.fsync(handle.fileno())
                                        raise Aborted()
                                    block = response.read(4 * 1024 * 1024)
                                    if not block:
                                        break
                                    handle.write(block)
                                    current += len(block)
                                    self.status_callback(f"Downloading sd-master-1d0a58a-bin-win-cuda12-x64.zip · {current / 1e6:.0f} / {runtime_size / 1e6:.0f} MB")
                                handle.flush(); os.fsync(handle.fileno())
                        if part.stat().st_size != runtime_size:
                            raise RuntimeError(f"Engine binary size mismatch: got {part.stat().st_size} bytes; expected {runtime_size} bytes.")
                        actual = _sha256_file(part, self.force_stop)
                        if actual != runtime_sha256:
                            part.unlink(missing_ok=True)
                            raise RuntimeError(f"Engine SHA-256 mismatch; expected {runtime_sha256}, got {actual}.")
                        os.replace(part, archive)
                        break
                    except Aborted:
                        raise
                    except Exception as exc:
                        last_error = exc
                        self.log(f"Native engine download failed from {source_url}: {exc}")
                        part.unlink(missing_ok=True)
                        if attempt < 2:
                            time.sleep(1.0 * attempt)
                if archive.is_file():
                    break
            if not archive.is_file():
                raise RuntimeError(
                    "Automatic download of sd-master-1d0a58a-bin-win-cuda12-x64.zip failed. An internet connection is required for the first launch; "
                    "after a successful download, the application can run offline. Check your connection and try again. "
                    f"Technical details: {last_error}"
                ) from last_error

        # Validate in a sibling staging directory and keep the old runtime until the
        # installed copy also passes required CUDA payload checks.
        extract_root = runtime_root.parent / f"{runtime_tag}.extracting"
        runtime_has_server = runtime_root.exists() and any(p.is_file() for p in runtime_root.rglob("sd-server.exe"))
        if not runtime_has_server:
            shutil.rmtree(extract_root, ignore_errors=True)
            extract_root.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(archive, "r") as zf:
                    bad = zf.testzip()
                    if bad:
                        raise RuntimeError(f"Engine archive failed CRC verification at {bad}.")
                    for member in zf.infolist():
                        validate_archive_member_path(extract_root, member.filename)
                    zf.extractall(extract_root)
                if not any(p.is_file() for p in extract_root.rglob("sd-server.exe")):
                    raise RuntimeError("The official engine archive does not contain sd-server.exe.")

                def validate_staged_runtime(installed_root: Path):
                    if not any(p.is_file() for p in installed_root.rglob("sd-server.exe")):
                        raise RuntimeError("sd-server.exe was not found after runtime installation.")
                    self._ensure_cuda_runtime_payload(installed_root)

                install_directory_transactionally(extract_root, runtime_root, validator=validate_staged_runtime)
            except Exception:
                shutil.rmtree(extract_root, ignore_errors=True)
                raise
        server = next((p for p in runtime_root.rglob("sd-server.exe") if p.is_file()), None)
        if server is None:
            raise RuntimeError(f"Engine was not found at this path after extraction: {runtime_root}")
        self._ensure_cuda_runtime_payload(runtime_root)
        self.status_callback("CUDA engine is ready; checking model files…")
        return server.resolve()

    @staticmethod
    def _model_profile_signature(folder: Path) -> str:
        profile_path = folder / "model-selection.json"
        try:
            return hashlib.sha256(profile_path.read_bytes()).hexdigest()
        except OSError:
            return ""

    def _find_model_parts(self, folder: Path) -> dict[str, Path]:
        if not folder.is_dir():
            raise RuntimeError(f"Model directory was not found: {folder}")
        # Model Catalog may organize components in subfolders. Discover model files
        # recursively instead of silently reporting "missing" when they are present
        # one directory below the selected model root.
        try:
            files = [p for p in folder.rglob("*") if p.is_file()]
        except OSError as exc:
            raise RuntimeError(f"Could not scan model directory: {folder}\\n{exc}") from exc
        lower = {p: p.name.lower() for p in files}

        profile: dict[str, Any] = {}
        profile_mismatches: dict[str, str] = {}
        try:
            loaded = json.loads((folder / "model-selection.json").read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                profile = loaded
        except (OSError, ValueError, TypeError):
            pass

        def selected_path(role: str) -> Optional[Path]:
            entry = profile.get(role)
            if not isinstance(entry, dict):
                return None
            raw_path = entry.get("path")
            if not raw_path and entry.get("filename"):
                raw_path = str(folder / str(entry["filename"]))
            if not raw_path:
                return None
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = folder / candidate
            if candidate.is_file():
                mismatch = _model_role_mismatch(role, candidate.name)
                if mismatch:
                    profile_mismatches[role] = mismatch
                    # Ignore stale/wrong-role assignments and try safe discovery below.
                    return None
                return candidate.resolve()
            if role == "vision_projector":
                # Vision projector (mmproj) is optional and never required for non-GGUF / ConvRot or VAE img2img workflows.
                return None
            raise RuntimeError(
                f"Selected file for {role} was not found:\n{candidate}\n"
                "Open Models / Download and select or download the file again."
            )

        def first(matches):
            return next((p for p in files if matches(p, lower[p])), None)

        diffusion = selected_path("diffusion")
        if diffusion is None:
            diffusion = first(lambda p, n: n in {
                "qwen-image-2.1-uc-q4_k_m.gguf",
                "qwen-image-2.1-uc-q4_0.gguf",
                "qwen-image-2.1-uc-q5_k_m.gguf",
                "qwen-image-2.1-uc-q6_k.gguf",
                "qwen-image-2.1-uc-q8_0.gguf",
                "qwen-image-2.1-uc-nvfp4.safetensors",
            })
        if diffusion is None:
            diffusion = first(lambda p, n: ("qwen" in n or "rayqwen" in n or "sakura" in n or p.stem.lower().endswith("convrot"))
                              and ("uc-" in n or "int8" in n and "convrot" in n or "turbo" in n or "convrot" in n or p.stem.lower().endswith("convrot"))
                              and _model_role_mismatch("diffusion", n) == ""
                              and p.suffix.lower() in {".safetensors", ".gguf"})

        vae = selected_path("vae")
        if vae is None:
            vae = first(lambda p, n: "texture_fix_vae_for_qwen_image_2.1" in n and p.suffix.lower() == ".safetensors")
        if vae is None:
            vae = first(lambda p, n: "qwen_image_2.1_vae" in n and p.suffix.lower() in {".safetensors", ".gguf"})

        llm = selected_path("text_encoder")
        if llm is None:
            # Prefer the curated Turbo-family encoder when a folder has both it and an
            # older generic Qwen3-VL Q4_K_M encoder; the latter can shift visual fidelity.
            llm = first(lambda p, n: "qwen-image-2.1-turbo-uncensored" in n
                        and p.suffix.lower() == ".gguf"
                        and _model_role_mismatch("text_encoder", n) == "")
        if llm is None:
            llm = first(lambda p, n: "qwen3vl-8b-instruct-q4_k_m" in n and p.suffix.lower() == ".gguf")
        if llm is None:
            llm = first(lambda p, n: ("qwen3vl" in n or "text_encoder" in n or "text-encoder" in n or p.stem.lower().endswith("convrot"))
                        and ("convrot" in n or "bf16" in n or "heretic" in n or p.stem.lower().endswith("convrot"))
                        and _model_role_mismatch("text_encoder", n) == ""
                        and p.suffix.lower() == ".safetensors")
        if llm is None:
            llm = first(lambda p, n: (("qwen3vl" in n and ("instruct" in n or "8b" in n)) or "sakura-textencoder" in n or p.stem.lower().endswith("convrot"))
                        and _model_role_mismatch("text_encoder", n) == ""
                        and p.suffix.lower() in {".gguf", ".safetensors"})

        vision_profile_entry = profile.get("vision_projector")
        vision = selected_path("vision_projector")
        # An explicit "none" selection must not be overridden by auto-discovery.
        # Otherwise, prefer an mmproj whose family matches the selected GGUF encoder.
        if vision is None and not (isinstance(vision_profile_entry, dict) and vision_profile_entry.get("id") == "none"):
            llm_name = lower.get(llm, llm.name.lower()) if llm is not None else ""
            if "qwen-image-2.1-turbo-uncensored" in llm_name:
                vision = first(lambda p, n: "mmproj-qwen-image-2.1-turbo-uncensored" in n and p.suffix.lower() == ".gguf")
            elif "qwen3vl-8b-instruct" in llm_name:
                vision = first(lambda p, n: "mmproj-qwen3vl-8b-instruct" in n and p.suffix.lower() == ".gguf")
            if vision is None:
                vision = first(lambda p, n: "mmproj" in n and ("qwen3vl" in n or "owen3vl" in n or "qwen-image" in n) and p.suffix.lower() == ".gguf")
        # Official Qwen Image 2.1 layouts embed the vision branch in the Qwen3-VL safetensors encoder.
        # A separate --llm_vision/mmproj is used with the GGUF encoder layout; passing one alongside
        # the ConvRot safetensors encoder may create an incompatible projector/hidden-size pairing.
        if llm is not None and _text_encoder_embeds_vision(llm):
            vision = None

        missing = []
        if diffusion is None: missing.append("Qwen-Image 2.1 image model (compatible GGUF or Safetensors)")
        if vae is None: missing.append("VAE compatible with Qwen-Image 2.1")
        if llm is None: missing.append("Compatible text encoder, such as the AtomicChat Qwen-Image-2.1-Turbo-Uncensored GGUF encoder")
        if missing:
            details = ""
            if profile_mismatches:
                details = "\n\nA previously saved model was assigned to the wrong role and has been ignored:\n" + "\n".join(
                    f"• {MODEL_ROLE_LABELS.get(role, role)}: {problem}"
                    for role, problem in profile_mismatches.items()
                )
            raise RuntimeError(
                "Required model components were not found:\n• " + "\n• ".join(missing) +
                "\n\nClick Models / Download… and choose an existing file or a downloadable option for each component." + details
            )
        elif profile_mismatches:
            # Surface the correction instead of silently continuing with stale role assignments.
            self.status_callback("Model selection warning: " + " | ".join(profile_mismatches.values()) + " A compatible model was selected from available files.")
        return {"diffusion": diffusion.resolve(), "vae": vae.resolve(), "llm": llm.resolve(), "llm_vision": vision.resolve() if vision else None}

    def _request_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None, timeout: float = 8.0) -> dict[str, Any]:
        if not self.server_base_url:
            raise RuntimeError("The local engine has not started yet.")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.server_base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": f"PAI-Image-Studio/{APP_VERSION}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(1024 * 1024).decode("utf-8", errors="replace")
            raise RuntimeError(f"The local engine returned HTTP {exc.code}: {detail[:1200]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Could not connect to the local engine: {exc}") from exc
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("The local engine did not return valid JSON.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("The local engine returned an invalid response structure.")
        return result

    @staticmethod
    def _read_log_window(path: Optional[Path], offset: int, end: Optional[int] = None,
                         max_bytes: int = 512 * 1024) -> tuple[int, str]:
        """Read newly appended native-engine log bytes without rescanning old runs."""
        if path is None:
            return offset, ""
        try:
            size = path.stat().st_size
            if end is not None:
                size = min(size, int(end))
            if size < offset:
                offset = 0  # file was replaced/rotated
            if size <= offset:
                return offset, ""
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(min(size - offset, max_bytes))
                new_offset = handle.tell()
            return new_offset, data.decode("utf-8", errors="replace")
        except OSError:
            return offset, ""

    @staticmethod
    def _tail_file(path: Optional[Path], max_chars: int = 3500) -> str:
        if not path or not path.is_file():
            return ""
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_chars * 4), os.SEEK_SET)
                return handle.read().decode("utf-8", errors="replace")[-max_chars:]
        except OSError:
            return ""

    def ensure_loaded(self):
        with self._lock:
            if self._is_process_alive(self.server_process):
                try:
                    self._request_json("GET", "/sdcpp/v1/capabilities", timeout=2.0)
                    return True
                except Exception:
                    # The server can be alive while still initializing; fall through and poll below.
                    pass
            if self.server_process is not None:
                self.unload_model()
            # Fail closed: never silently run a CPU-only diffusion backend. Use the selected
            # NVIDIA CUDA device for diffusion and VAE; the text encoder may remain on CPU
            # on 12 GB cards to preserve VRAM for the denoiser.
            gpus = detect_nvidia_gpus()
            self.gpu_info = next((gpu for gpu in gpus if gpu["index"] == self.gpu_index), None)
            if self.gpu_info is None:
                detected = ", ".join(f"{gpu['index']}: {gpu['name']}" for gpu in gpus) or "none"
                raise RuntimeError(
                    "No NVIDIA CUDA GPU was found, or the selected GPU is no longer available. "
                    "The application will not silently fall back to CPU image generation. Check the NVIDIA driver, "
                    "click Refresh GPU, and select a valid GPU. "
                    f"Detected GPUs: {detected}"
                )
            model_dir = Path(self.model_path).expanduser().resolve() if self.model_path else MODEL_DIR
            parts = self._find_model_parts(model_dir)
            executable = self._runtime_executable()
            log_dir = CACHE_DIR / "engine"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.server_log_path = log_dir / "sd-server.log"
            self._server_log_handle = self.server_log_path.open("ab", buffering=0)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
            self.server_base_url = validate_loopback_url(f"http://127.0.0.1:{port}")
            gpu_index = int(self.gpu_info["index"])
            gpu_uuid = str(self.gpu_info.get("uuid", "")).strip()
            # Bind the child process to the selected physical GPU by UUID, then address it
            # consistently as CUDA device 0. This avoids index mismatches on multi-GPU hosts.
            use_uuid_mask = bool(gpu_uuid and len(gpus) > 1)
            cuda_device_index = 0 if use_uuid_mask else gpu_index
            gpu_name = str(self.gpu_info["name"])
            total_vram_gb = float(self.gpu_info["memory_total_mb"]) / 1024.0
            free_vram_gb = max(0.0, float(self.gpu_info.get("memory_free_mb", 0)) / 1024.0)
            diffusion_gb = parts["diffusion"].stat().st_size / (1024 ** 3)
            component_file_gb = diffusion_gb + sum(
                parts[key].stat().st_size / (1024 ** 3)
                for key in ("llm", "vae", "llm_vision")
                if parts.get(key) is not None and parts[key].is_file()
            )
            vae_gb = parts["vae"].stat().st_size / (1024 ** 3)
            placement = plan_qwen_memory_placement(
                free_vram_gb=free_vram_gb,
                total_vram_gb=total_vram_gb,
                diffusion_gb=diffusion_gb,
                vae_gb=vae_gb,
                component_file_gb=component_file_gb,
                text_encoder_on_cpu=self.text_encoder_on_cpu,
                memory_mode=self.gpu_memory_mode,
                auto_offload_override=self.auto_offload_override,
            )
            activation_reserve_gb = float(placement["activation_reserve_gb"])
            force_cpu_te_for_capacity = bool(placement["force_cpu_te_for_capacity"])
            text_backend = str(placement["text_backend"])
            offload_to_cpu = bool(placement["offload_to_cpu"])
            auto_fit_mode = str(placement["auto_fit_mode"])
            self._last_offload_to_cpu = offload_to_cpu
            backend_spec = f"diffusion=cuda{cuda_device_index},te={text_backend},vae=cuda{cuda_device_index}"
            # When the bundle fits, disable auto-fit so it cannot distribute model
            # weights into system RAM anyway. Use auto-fit only if free-VRAM telemetry is missing.
            auto_fit_mode = "off" if (
                self.gpu_memory_mode == "full_gpu" or offload_to_cpu or text_backend == "cpu"
                or (free_vram_gb > 0 and free_vram_gb >= component_file_gb + activation_reserve_gb)
            ) else "on"
            args = [
                str(executable),
                "--diffusion-model", str(parts["diffusion"]),
                "--vae", str(parts["vae"]),
                "--llm", str(parts["llm"]),
                "--listen-ip", "127.0.0.1", "--listen-port", str(port),
                "--backend", backend_spec,
                "--auto-fit", auto_fit_mode,
                "--conditioning-cache-size", "8",
                "--log-level", "info",
            ]
            if offload_to_cpu:
                args.append("--offload-to-cpu")
            # Flash Attention typically helps CUDA diffusion. On 16+ GB cards where
            # the text encoder is also on CUDA, enable it there too: Qwen-Image's prompt
            # path has a separate attention module. Avoid enabling it for CPU text encoding.
            flash_attn_spec = "off,diffusion=on,te=on" if text_backend != "cpu" else "off,diffusion=on"
            args.extend(["--flash-attn", flash_attn_spec])
            if parts.get("llm_vision") is not None:
                args.extend(["--llm_vision", str(parts["llm_vision"])])
            if self.lora_enabled and self.lora_path:
                lora_file = Path(self.lora_path).expanduser().resolve()
                if not lora_file.is_file():
                    raise RuntimeError(f"Selected LoRA file does not exist: {lora_file}")
                if lora_file.suffix.lower() not in {".safetensors", ".sft", ".gguf", ".pt", ".bin"}:
                    raise RuntimeError("Unsupported LoRA file extension. Choose a compatible Qwen Image 2.1 LoRA (.safetensors/.sft/.gguf/.pt/.bin).")
                if lora_file.suffix.lower() in {".safetensors", ".sft", ".gguf"}:
                    validate_lora_file(lora_file)
                args.extend(["--lora-model-dir", str(lora_file.parent)])
            env = os.environ.copy()
            if gpu_uuid and len(gpus) > 1:
                env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            else:
                # Avoid inherited visibility masks causing a mismatch with nvidia-smi indexes.
                env.pop("CUDA_VISIBLE_DEVICES", None)
            # Native DLLs from the pinned runtime package may live in adjacent folders.
            dll_dirs = {str(executable.parent)}
            for runtime_root in (executable.parent, executable.parent.parent, APP_DIR / "runtime", DATA_ROOT / "runtime"):
                if runtime_root.is_dir():
                    try:
                        dll_dirs.update(str(d.parent) for d in runtime_root.rglob("*.dll"))
                    except OSError:
                        pass
            dll_dirs.update(str(path) for path in self._windows_cuda_dependency_dirs())
            env["PATH"] = os.pathsep.join(sorted(dll_dirs)) + os.pathsep + env.get("PATH", "")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self.log("Starting CUDA-enabled stable-diffusion.cpp engine on loopback only; CPU fallback for diffusion is disabled.")
            if self.gpu_memory_mode == "full_gpu":
                self.log(
                    "GPU-only policy enabled: text encoder, vision projector, diffusion and VAE are all assigned to CUDA; "
                    "CPU inference and CPU weight offload are disabled. If VRAM is insufficient, model loading/inference "
                    "will fail instead of silently falling back to CPU. CPU still handles the desktop UI, file I/O and process orchestration."
                )
            if force_cpu_te_for_capacity:
                encoder_gb = sum(
                    parts[key].stat().st_size / (1024 ** 3)
                    for key in ("llm", "llm_vision")
                    if parts.get(key) is not None and parts[key].is_file()
                )
                self.log(
                    f"VRAM safety fallback · model files total {component_file_gb:.1f} GiB + "
                    f"{activation_reserve_gb:.1f} GiB workspace reserve exceed the {free_vram_gb:.1f} GiB free dedicated VRAM budget. "
                    f"Text encoder/projector ({encoder_gb:.1f} GiB) run on CPU; diffusion ({diffusion_gb:.1f} GiB) and VAE ({vae_gb:.1f} GiB) stay on CUDA; auto-fit={auto_fit_mode}. "
                    "This avoids dynamic placement into Windows shared GPU memory."
                )
            self.log(f"Selected NVIDIA GPU: {gpu_name} · NVML index {gpu_index} · {total_vram_gb:.1f} GB total / {free_vram_gb:.1f} GB free · diffusion weight={diffusion_gb:.2f} GiB · model bundle={component_file_gb:.2f} GiB · workspace reserve={activation_reserve_gb:.1f} GiB · memory_policy={self.gpu_memory_mode} · text_encoder_capacity_fallback={force_cpu_te_for_capacity} · parameter_offload={offload_to_cpu} · auto_fit={auto_fit_mode} · CUDA backend={backend_spec} · UUID isolation={bool(gpu_uuid)} · flash_attention={flash_attn_spec} · text conditioning LRU cache=8 prompts")
            other_component_gb = sum(
                parts[key].stat().st_size / (1024 ** 3)
                for key in ("llm", "vae", "llm_vision")
                if parts.get(key) is not None and parts[key].is_file()
            )
            estimated_weight_gb = diffusion_gb + other_component_gb
            if self.gpu_memory_mode == "full_gpu" and not force_cpu_te_for_capacity and free_vram_gb > 0 and free_vram_gb < estimated_weight_gb + 1.5:
                self.log(
                    f"WARNING: Full GPU mode requested, but free dedicated VRAM ({free_vram_gb:.1f} GiB) "
                    f"is below the approximate component-file total plus 1.5 GiB headroom "
                    f"({estimated_weight_gb + 1.5:.1f} GiB). Activation buffers need extra memory; "
                    "Windows may report shared GPU memory usage or CUDA may run out of memory."
                )
            self.status_callback(f"Loading model on NVIDIA GPU: {gpu_name} · {total_vram_gb:.1f} GB VRAM…")
            self.log(f"Engine: {executable}")
            vision_label = (parts["llm_vision"].name if parts.get("llm_vision")
                            else "embedded in Qwen3-VL safetensors" if _text_encoder_embeds_vision(parts.get("llm"))
                            else "disabled (text-to-image only)")
            self.log(f"Diffusion weights: {parts['diffusion'].name}; LLM: {parts['llm'].name}; vision: {vision_label}; VAE: {parts['vae'].name}")
            startup_log_offset = self.server_log_path.stat().st_size if self.server_log_path and self.server_log_path.exists() else 0
            try:
                self.server_process = subprocess.Popen(
                    args, cwd=str(executable.parent), env=env, stdin=subprocess.DEVNULL,
                    stdout=self._server_log_handle, stderr=subprocess.STDOUT,
                    creationflags=creationflags, close_fds=(os.name != "nt"),
                )
            except Exception as exc:
                self.unload_model()
                raise RuntimeError(f"Failed to run the native engine: {exc}") from exc

            deadline = time.monotonic() + 15 * 60
            last_log_size = startup_log_offset
            log_cursor = startup_log_offset
            load_phase_progress = {
                "loading diffusion model from": ("Diffusion model weights", 18),
                "loading llm from": ("Text encoder", 34),
                "loading llm vision from": ("Vision projector", 49),
                "loading vae from": ("VAE weights", 64),
                "total params memory size": ("Allocating model memory", 78),
                "running in flow mode": ("Initializing inference pipeline", 88),
            }
            while time.monotonic() < deadline:
                if self.force_stop.is_set():
                    self.unload_model()
                    raise Aborted()
                if not self._is_process_alive(self.server_process):
                    details = self._tail_file(self.server_log_path)
                    self.unload_model()
                    raise RuntimeError(
                        "The native engine closed while loading the model. Check runtime/model compatibility and update the NVIDIA driver.\n\n"
                        + details[-3500:]
                    )
                try:
                    capabilities = self._request_json("GET", "/sdcpp/v1/capabilities", timeout=2.0)
                    modes = capabilities.get("supported_modes", [])
                    if not modes or "img_gen" in modes:
                        self._loaded_model_path = str(model_dir)
                        self._loaded_parts = dict(parts)
                        self._loaded_profile_signature = self._model_profile_signature(model_dir)
                        self.status_callback(f"GPU ready: {gpu_name} · CUDA device {gpu_index}")
                        self.log(f"Native engine ready on requested NVIDIA CUDA backend {backend_spec}. No external workflow server or hosted inference is used.")
                        return True
                except Exception:
                    pass
                try:
                    log_size = self.server_log_path.stat().st_size if self.server_log_path else 0
                    if log_size != last_log_size or log_cursor < log_size:
                        log_cursor, new_log = self._read_log_window(self.server_log_path, log_cursor, log_size)
                        # Keep polling if a large log chunk required more than one bounded read.
                        last_log_size = log_cursor
                        lower_log = new_log.lower()
                        matched = None
                        # The last matching stage in this chunk is the most recent backend checkpoint.
                        match_pos = -1
                        for marker, spec in load_phase_progress.items():
                            pos = lower_log.rfind(marker)
                            if pos > match_pos:
                                matched, match_pos = spec, pos
                        if matched:
                            self.status_callback(f"Model load · {matched[0]} · {matched[1]}% estimated")
                        else:
                            self.status_callback("Model load · Initializing CUDA and model components · 10% estimated")
                except OSError:
                    pass
                time.sleep(0.35)
            details = self._tail_file(self.server_log_path)
            self.unload_model()
            raise RuntimeError("Model loading exceeded the 15-minute timeout. Log details:\n\n" + details[-3500:])

    def unload_model(self):
        with self._lock:
            proc = self.server_process
            self.server_process = None
            self.server_base_url = ""
            self._loaded_model_path = None
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=6)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=3)
                except Exception as exc:
                    self.log(f"Could not stop local engine cleanly: {exc}")
            if self._server_log_handle is not None:
                try:
                    self._server_log_handle.close()
                except Exception:
                    pass
                self._server_log_handle = None
            self.status_callback("Local model unloaded.")
            self.log("Local inference engine stopped.")

    def _ensure_hf_realesrgan_pair(self, model_dir: Path, stem: str) -> None:
        """Fetch verified-size NCNN model files from the Hugging Face weight mirror.

        The HF mirror supplies upstream `.param`/`.bin` pairs, not the Windows executable.
        The executable is still obtained only from the official GitHub release or its
        SourceForge mirror. This path is used only for `realesrgan-x4plus` and AnimeVideo-v3.
        """
        allowed_stems = {
            "realesrgan-x4plus",
            "realesrgan-x4plus-anime",
            "realesr-animevideov3-x2",
            "realesr-animevideov3-x3",
            "realesr-animevideov3-x4",
        }
        if stem not in allowed_stems:
            raise RuntimeError(
                f"The installed Real-ESRGAN package does not include model '{stem}'. "
                "Choose a model included in the official portable archive, or restore that archive."
            )
        # HF mirrors the standard NCNN model assets under their upstream filenames. The anime
        # x4plus pair is hosted in a separate HF repository and has different capitalisation.
        remote_base = REALESRGAN_HF_NCNN_BASE
        remote_stem = stem
        if stem == "realesrgan-x4plus-anime":
            remote_base = "https://huggingface.co/HNoise7676/RealESRGAN_x4plus_anime_6B_NCNN/resolve/main/"
            remote_stem = "RealESRGAN_x4plus_anime_6B"
        model_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("param", "bin"):
            filename = f"{stem}.{suffix}"
            remote_filename = f"{remote_stem}.{suffix}"
            target = model_dir / filename
            if suffix == "param":
                min_size = 25_000 if stem == "realesrgan-x4plus-anime" else 1000
            elif stem == "realesrgan-x4plus":
                min_size = 25_000_000
            elif stem == "realesrgan-x4plus-anime":
                min_size = 15_000_000
            else:
                min_size = 900_000
            if target.is_file() and target.stat().st_size >= min_size:
                continue
            part = target.with_suffix(target.suffix + ".part")
            part.unlink(missing_ok=True)
            url = remote_base + urllib.parse.quote(remote_filename)
            self.status_callback(f"Downloading {filename} from Hugging Face model mirror…")
            self.log(f"Downloading missing NCNN model asset from the Hugging Face mirror: {url}")
            request = urllib.request.Request(url, headers={"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=180, context=_download_ssl_context()) as response, part.open("wb") as out:
                    expected = response.headers.get("Content-Length")
                    received = 0
                    while True:
                        if self.force_stop.is_set():
                            raise Aborted()
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                    out.flush(); os.fsync(out.fileno())
                if expected and received != int(expected):
                    raise RuntimeError(f"Incomplete Hugging Face response for {filename}: received {received} bytes, expected {expected}.")
                if received < min_size:
                    raise RuntimeError(f"Hugging Face returned an undersized {filename} ({received} bytes).")
                os.replace(part, target)
                self.log(f"Hugging Face model asset saved: {target.name} ({received:,} bytes; SHA-256={_sha256_file(target, self.force_stop)}).")
            except Exception:
                part.unlink(missing_ok=True)
                raise

    def _ensure_realesrgan_model_pair(self, exe: Path, model_name: str, scale: int) -> None:
        model_dir = exe.parent / "models"
        stem = f"realesr-animevideov3-x{scale}" if model_name == "realesr-animevideov3" else model_name
        needed = (model_dir / f"{stem}.param", model_dir / f"{stem}.bin")
        min_param = 25_000 if stem == "realesrgan-x4plus-anime" else 1000
        min_bin = (25_000_000 if stem in {"realesrgan-x4plus", "realesrnet-x4plus"}
                   else 15_000_000 if stem == "realesrgan-x4plus-anime" else 900_000)
        minimum_sizes = (min_param, min_bin)
        def pair_is_valid() -> bool:
            return all(path.is_file() and path.stat().st_size >= minimum
                       for path, minimum in zip(needed, minimum_sizes))
        if pair_is_valid():
            return
        if not pair_is_valid():
            cached_archive = CACHE_DIR / "runtime-downloads" / "realesrgan-ncnn-vulkan-20220424-windows.zip"
            if restore_realesrgan_model_pair_from_archive(cached_archive, model_dir, stem):
                self.log(f"Restored missing Real-ESRGAN model pair '{stem}' from the cached official Windows runtime archive.")
        if not pair_is_valid() and (stem in {"realesrgan-x4plus", "realesrgan-x4plus-anime"} or stem.startswith("realesr-animevideov3-x")):
            try:
                self._ensure_hf_realesrgan_pair(model_dir, stem)
            except Exception as repair_error:
                self.log(f"Hugging Face repair source could not restore '{stem}': {repair_error}")
        if not pair_is_valid():
            raise RuntimeError(
                f"The selected Real-ESRGAN model '{model_name}' is missing or has incomplete NCNN .param/.bin files. "
                "The app checked the cached official Windows archive and available Hugging Face NCNN mirror. "
                "Restore the Real-ESRGAN runtime archive or reinstall the selected model pair before retrying."
            )

    def _ensure_realesrgan_runtime(self) -> Path:
        """Download the official portable Real-ESRGAN Vulkan runtime on first use.

        It is installed in the user's data directory rather than beside the app, so
        upgrades do not remove it and the second use can work offline.
        """
        if os.name != "nt":
            raise RuntimeError("This prebuilt upscaler package currently supports Windows x64 only.")
        runtime_root = DATA_ROOT / "runtime" / "realesrgan-ncnn-vulkan"
        exe = next((p for p in runtime_root.rglob("realesrgan-ncnn-vulkan.exe") if p.is_file()), None) if runtime_root.exists() else None
        if exe is not None:
            model_dir = exe.parent / "models"
            installed_model_assets = {str(path) for path in model_dir.glob("*") if path.is_file()} if model_dir.is_dir() else set()
            if not _realesrgan_model_assets_present(installed_model_assets):
                try:
                    self._ensure_hf_realesrgan_pair(model_dir, "realesrgan-x4plus")
                    installed_model_assets = {str(path) for path in model_dir.glob("*") if path.is_file()}
                except Exception as exc:
                    self.log(f"Could not repair missing Real-ESRGAN model files from Hugging Face: {exc}")
            if _realesrgan_model_assets_present(installed_model_assets):
                return exe
            # The runtime executable is already installed. Do not fetch the 45 MB ZIP again
            # just because a weight pair is absent; the selected-model check below will report
            # or repair that pair independently.
            self.log("Real-ESRGAN Windows executable is already installed; will repair only missing model weights instead of redownloading the runtime ZIP.")
            return exe

        cache_dir = CACHE_DIR / "runtime-downloads"
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "realesrgan-ncnn-vulkan-20220424-windows.zip"
        part = archive.with_suffix(archive.suffix + ".part")
        def valid_archive(path: Path) -> bool:
            try:
                if not path.is_file() or path.stat().st_size < 10_000_000:
                    return False
                with zipfile.ZipFile(path, "r") as zf:
                    if zf.testzip() is not None:
                        return False
                    basenames = {Path(n.replace("\\", "/")).name.lower() for n in zf.namelist()}
                    # Do not reject a valid upstream executable ZIP merely because an optional model family is absent.
                    return "realesrgan-ncnn-vulkan.exe" in basenames
            except (OSError, zipfile.BadZipFile, RuntimeError):
                return False

        if not archive.is_file() or not valid_archive(archive):
            archive.unlink(missing_ok=True)
            part.unlink(missing_ok=True)
            self.status_callback("Downloading realesrgan-ncnn-vulkan-20220424-windows.zip · official source / mirror…")
            last_error = None
            for source_index, source_url in enumerate(REALESRGAN_NCNN_URLS):
                # Do not combine partial data from different mirrors; every accepted ZIP
                # must pass CRC and expected-executable/model-asset validation.
                part.unlink(missing_ok=True)
                self.log(f"Downloading portable Real-ESRGAN Vulkan runtime (source {source_index + 1}/{len(REALESRGAN_NCNN_URLS)}): {source_url}")
                try:
                    request = urllib.request.Request(
                        source_url,
                        headers={"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"},
                    )
                    with urllib.request.urlopen(request, timeout=180, context=_download_ssl_context()) as response, part.open("wb") as out:
                        expected_length = response.headers.get("Content-Length")
                        expected_bytes = int(expected_length) if expected_length and expected_length.isdigit() else 0
                        received = 0
                        while True:
                            if self.force_stop.is_set():
                                raise Aborted()
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            received += len(chunk)
                            if expected_bytes:
                                self.status_callback(f"Downloading realesrgan-ncnn-vulkan-20220424-windows.zip · {received / 1e6:.0f} / {expected_bytes / 1e6:.0f} MB")
                            else:
                                self.status_callback(f"Downloading realesrgan-ncnn-vulkan-20220424-windows.zip · {received / 1e6:.0f} MB downloaded")
                        out.flush(); os.fsync(out.fileno())
                    if expected_length and received != int(expected_length):
                        raise RuntimeError(f"Incomplete response: received {received} bytes, expected {expected_length}.")
                    if not valid_archive(part):
                        raise RuntimeError("The downloaded file is not a valid Real-ESRGAN Windows ZIP (missing executable, invalid ZIP, or failed CRC check).")
                    part.replace(archive)
                    self.log(f"Real-ESRGAN archive accepted from source: {source_url}")
                    break
                except Aborted:
                    part.unlink(missing_ok=True)
                    raise
                except Exception as exc:
                    last_error = exc
                    self.log(f"Real-ESRGAN source failed: {source_url} · {exc}")
                    part.unlink(missing_ok=True)
            if not archive.is_file() or not valid_archive(archive):
                raise RuntimeError(
                    "Failed to download Real-ESRGAN from the official source and fallback mirror. Check internet/proxy settings and free disk space. "
                    f"Latest error details: {last_error}"
                ) from last_error

        runtime_root.mkdir(parents=True, exist_ok=True)
        # Extract manually and reject zip-slip paths before writing files.
        root_resolved = runtime_root.resolve()
        with zipfile.ZipFile(archive, "r") as zf:
            if zf.testzip() is not None:
                raise RuntimeError("The Real-ESRGAN archive CRC is invalid; delete the ZIP and download it again.")
            for info in zf.infolist():
                target = (runtime_root / info.filename).resolve()
                try:
                    target.relative_to(root_resolved)
                except ValueError:
                    raise RuntimeError("An unsafe path was detected inside the upscaler archive.")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

        exe = next((p for p in runtime_root.rglob("realesrgan-ncnn-vulkan.exe") if p.is_file()), None)
        if exe is None:
            raise RuntimeError("The Real-ESRGAN executable was not found in the downloaded package.")
        model_dir = exe.parent / "models"
        installed_model_assets = {str(path) for path in model_dir.glob("*") if path.is_file()} if model_dir.is_dir() else set()
        if not _realesrgan_model_assets_present(installed_model_assets):
            # HF only hosts the model weights, not the Windows executable. Keep official executable provenance,
            # and repair its default image model separately if the upstream ZIP omits model assets.
            self._ensure_hf_realesrgan_pair(model_dir, "realesrgan-x4plus")
            installed_model_assets = {str(path) for path in model_dir.glob("*") if path.is_file()}
        if not _realesrgan_model_assets_present(installed_model_assets):
            raise RuntimeError("The Real-ESRGAN runtime has an executable but no complete NCNN model pair after repair.")
        self.log(f"Real-ESRGAN portable runtime ready · SHA-256={hashlib.sha256(archive.read_bytes()).hexdigest()}")
        return exe

    def _ensure_upscayl_runtime(self) -> Path:
        """Install a pinned Upscayl-NCNN Windows runner for curated custom NCNN models."""
        if os.name != "nt":
            raise RuntimeError("The bundled Upscayl-NCNN runner currently supports Windows x64 only.")
        runtime_root = DATA_ROOT / "runtime" / "upscayl-ncnn"
        exe = next((p for p in runtime_root.rglob("upscayl-bin.exe") if p.is_file()), None) if runtime_root.exists() else None
        if exe is not None:
            return exe

        cache_dir = CACHE_DIR / "runtime-downloads"
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "upscayl-bin-20251207-174704-windows.zip"
        part = archive.with_suffix(archive.suffix + ".part")

        def valid_archive(path: Path) -> bool:
            try:
                if not path.is_file() or path.stat().st_size != UPSCAYL_NCNN_ARCHIVE_SIZE:
                    return False
                digest = _sha256_file(path, self.force_stop)
                if digest.lower() != UPSCAYL_NCNN_SHA256.lower():
                    return False
                with zipfile.ZipFile(path, "r") as zf:
                    if zf.testzip() is not None:
                        return False
                    return any(Path(n.replace("\\", "/")).name.lower() == "upscayl-bin.exe" for n in zf.namelist())
            except (OSError, zipfile.BadZipFile, RuntimeError, Aborted):
                return False

        if not valid_archive(archive):
            archive.unlink(missing_ok=True)
            part.unlink(missing_ok=True)
            self.status_callback("Downloading the pinned Upscayl-NCNN runtime…")
            request = urllib.request.Request(
                UPSCAYL_NCNN_URL,
                headers={"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180, context=_download_ssl_context()) as response, part.open("wb") as out:
                    expected_header = response.headers.get("Content-Length")
                    received = 0
                    while True:
                        if self.force_stop.is_set():
                            raise Aborted()
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        self.status_callback(f"Downloading Upscayl-NCNN runtime · {received / 1e6:.1f} / {UPSCAYL_NCNN_ARCHIVE_SIZE / 1e6:.1f} MB")
                    out.flush()
                    os.fsync(out.fileno())
                if expected_header and expected_header.isdigit() and received != int(expected_header):
                    raise RuntimeError(f"Incomplete Upscayl-NCNN response: received {received} bytes, expected {expected_header}.")
                if not valid_archive(part):
                    raise RuntimeError("The Upscayl-NCNN archive failed its pinned SHA-256, size, executable, or ZIP CRC check.")
                part.replace(archive)
            except Exception:
                part.unlink(missing_ok=True)
                raise
        runtime_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            if zf.testzip() is not None:
                raise RuntimeError("The Upscayl-NCNN runtime ZIP has invalid CRC data.")
            for info in zf.infolist():
                target = validate_archive_member_path(runtime_root, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        exe = next((p for p in runtime_root.rglob("upscayl-bin.exe") if p.is_file()), None)
        if exe is None:
            raise RuntimeError("The verified Upscayl-NCNN archive did not contain upscayl-bin.exe.")
        self.log(f"Upscayl-NCNN runtime installed and verified · SHA-256={UPSCAYL_NCNN_SHA256}.")
        return exe

    @staticmethod
    def _git_blob_sha1(path: Path) -> str:
        """Calculate the Git object SHA-1 (blob header + bytes) used by the pinned model manifest."""
        size = path.stat().st_size
        digest = hashlib.sha1()
        digest.update(("blob %d\\0" % size).encode("ascii"))
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_upscayl_model_pair(self, exe: Path, model_name: str) -> Path:
        """Download a pinned custom NCNN model pair and verify exact sizes and Git blob hashes."""
        meta = REALESRGAN_CATALOG_BY_ID.get(model_name)
        if not meta or meta.get("backend") != "upscayl":
            raise RuntimeError(f"No pinned Upscayl-NCNN model manifest exists for '{model_name}'.")
        model_dir = exe.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        stem = str(meta["model_file"])
        for suffix in ("param", "bin"):
            filename = f"{stem}.{suffix}"
            target = model_dir / filename
            expected_size = int(meta[f"{suffix}_size"])
            expected_blob = str(meta[f"{suffix}_blob_sha1"]).lower()

            def is_valid(path: Path) -> bool:
                try:
                    return path.is_file() and path.stat().st_size == expected_size and self._git_blob_sha1(path).lower() == expected_blob
                except OSError:
                    return False

            if is_valid(target):
                continue
            part = target.with_suffix(target.suffix + ".part")
            part.unlink(missing_ok=True)
            url = UPSCAYL_NCNN_MODEL_BASE_URL + urllib.parse.quote(filename)
            self.status_callback(f"Downloading Ultimate model asset · {filename}…")
            self.log(f"Fetching pinned community NCNN model asset: {url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180, context=_download_ssl_context()) as response, part.open("wb") as out:
                    expected_header = response.headers.get("Content-Length")
                    received = 0
                    while True:
                        if self.force_stop.is_set():
                            raise Aborted()
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        self.status_callback(f"Downloading {filename} · {received / 1e6:.1f} / {expected_size / 1e6:.1f} MB")
                    out.flush()
                    os.fsync(out.fileno())
                if expected_header and expected_header.isdigit() and received != int(expected_header):
                    raise RuntimeError(f"Incomplete {filename} response: received {received} bytes, expected {expected_header}.")
                if received != expected_size or self._git_blob_sha1(part).lower() != expected_blob:
                    raise RuntimeError(f"Downloaded NCNN model asset '{filename}' failed its pinned size or Git blob SHA-1 validation.")
                os.replace(part, target)
            except Exception:
                part.unlink(missing_ok=True)
                raise
        self.log(f"Verified custom NCNN model '{model_name}' installed in {model_dir}.")
        return model_dir

    def _upscale_image_with_selected_model(
        self, image: Image.Image, options: dict[str, Any], target_size: tuple[int, int]
    ) -> Image.Image:
        """Run the selected learned NCNN upscaler on an in-memory Ultimate stage, then fit exact canvas."""
        temp_root = CACHE_DIR / "ultimate-upscale-temp" / uuid.uuid4().hex
        temp_root.mkdir(parents=True, exist_ok=False)
        source_path = temp_root / "ultimate-stage.png"
        try:
            image.convert("RGB").save(source_path, format="PNG")
            result_path = self._run_upscale_file(source_path, options, temp_root)
            with Image.open(result_path) as opened:
                result = ImageOps.exif_transpose(opened).convert("RGB")
            wanted = (max(1, int(target_size[0])), max(1, int(target_size[1])))
            if result.size != wanted:
                self.log(
                    f"Selected AI upscaler produced {result.width}×{result.height}; "
                    f"fitting its learned result to the exact Ultimate canvas {wanted[0]}×{wanted[1]}."
                )
                result = result.resize(wanted, Image.Resampling.LANCZOS)
            return result
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _run_upscale_file(self, input_path: Path, options: dict[str, Any], output_dir: Optional[Path] = None) -> Path:
        """Run the selected learned NCNN photo upscaler locally through Vulkan."""
        input_path = input_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Upscaler input file was not found: {input_path}")
        model_name = str(options.get("upscale_model", "realesrgan-x4plus")).strip()
        allowed_models = set(REALESRGAN_MODEL_NAMES.values()) | set(REALESRGAN_MODEL_NATIVE_SCALES.keys())
        if model_name not in allowed_models:
            raise ValueError(f"Unknown upscaler model: {model_name}")
        scale = realesrgan_native_scale(model_name)
        tile = max(0, min(1024, int(options.get("upscale_tile", 0))))
        if tile not in (0, 128, 200, 256, 384, 512, 600, 768, 1024):
            tile = 256

        model_meta = REALESRGAN_CATALOG_BY_ID.get(model_name, {})
        backend = str(model_meta.get("backend", "official"))
        if backend == "upscayl":
            exe = self._ensure_upscayl_runtime()
            model_dir = self._ensure_upscayl_model_pair(exe, model_name)
            cli_model_name = str(model_meta.get("model_file", model_name))
        else:
            exe = self._ensure_realesrgan_runtime()
            cli_model_name = "realesr-animevideov3" if model_name.startswith("realesr-animevideov3") else model_name
            self._ensure_realesrgan_model_pair(exe, cli_model_name, scale)
            model_dir = exe.parent / "models"

        target_dir = output_dir or input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        target = target_dir / f"{input_path.stem}_AI{scale}x_{stamp}.png"

        temp_input: Optional[Path] = None
        input_for_engine = input_path
        if input_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            temp_root = CACHE_DIR / "upscale-temp"
            temp_root.mkdir(parents=True, exist_ok=True)
            temp_input = temp_root / f"input_{uuid.uuid4().hex}.png"
            with Image.open(input_path) as image:
                ImageOps.exif_transpose(image).convert("RGB").save(temp_input, format="PNG")
            input_for_engine = temp_input

        log_name = "upscayl-ncnn.log" if backend == "upscayl" else "realesrgan.log"
        log_path = CACHE_DIR / "engine" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if backend == "upscayl":
            args = [
                str(exe), "-i", str(input_for_engine), "-o", str(target),
                "-m", str(model_dir), "-n", cli_model_name,
                "-z", str(scale), "-s", str(scale), "-t", str(tile), "-j", "1:2:2",
            ]
        else:
            args = [
                str(exe), "-i", str(input_for_engine), "-o", str(target),
                "-n", cli_model_name, "-s", str(scale), "-t", str(tile), "-j", "1:2:2",
            ]
        if bool(options.get("upscale_tta", False)):
            args.append("-x")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = os.environ.copy()
        self.status_callback(f"Ultimate · {model_name} · native {scale}× · Vulkan GPU running…")
        self.log(
            f"Running selected AI upscaler: {model_name} · backend={backend} · native scale={scale}× · "
            f"tile={tile} · Vulkan."
        )
        try:
            with log_path.open("wb") as log_handle:
                proc = subprocess.Popen(
                    args, cwd=str(exe.parent), stdout=log_handle, stderr=subprocess.STDOUT,
                    env=env, creationflags=creationflags, close_fds=(os.name != "nt"),
                )
                while proc.poll() is None:
                    if self.force_stop.wait(0.35):
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=1.5)
                        raise Aborted()
                return_code = proc.returncode
            if return_code != 0 or not target.is_file() or target.stat().st_size < 1024:
                details = self._tail_file(log_path, 5000)
                raise RuntimeError(
                    f"The selected AI upscaler could not process the image (exit code {return_code}). "
                    "Check the graphics driver / Vulkan support and the selected model files.\n\n" + details
                )
            with Image.open(target) as check_img:
                check_img.verify()
            with Image.open(input_for_engine) as source_check:
                source_w, source_h = source_check.size
            with Image.open(target) as result_check:
                result_w, result_h = result_check.size
            expected_w, expected_h = source_w * scale, source_h * scale
            if (result_w, result_h) != (expected_w, expected_h):
                raise RuntimeError(
                    f"AI upscaler returned unexpected dimensions {result_w}×{result_h}; expected "
                    f"{expected_w}×{expected_h}. The selected model's native scale or model files may be incompatible. "
                    f"See {log_path}."
                )
            label = f"{scale}× native ({result_w}×{result_h})"
            self.status_callback(f"Ultimate · selected AI model complete · {label} · {target.name}")
            self.log(f"AI upscale saved: {target} ({self._human_size(target.stat().st_size)}).")
            return target
        finally:
            if temp_input is not None:
                temp_input.unlink(missing_ok=True)

    @staticmethod
    def _needs_english_translation(text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return False
        if any(ord(ch) > 127 and ch.isalpha() for ch in s):
            return True
        words = set(re.findall(r"[a-zA-Z]+", s.casefold()))
        non_en_markers = {
            "aks", "dokhtar", "pesar", "zan", "mard", "lebas", "cheshm", "soorat",
            "poost", "moo", "sineh", "koon", "kos", "gardan", "lab", "dast",
            "taghir", "bede", "kon", "ba", "dar", "rooye", "rang", "ghermez", "abi",
            "una", "une", "ein", "eine", "avec", "dans", "mujer", "femme", "frau",
        }
        return bool(words & non_en_markers)

    def _translate_image_prompt_to_english(self, prompt_text: str, workflow: str) -> str:
        """Translate workflow prompts and Close-up targets with the selected local text encoder, fail-safe.

        Cache translations only in memory. If a compatible GGUF encoder is not
        configured or the local encoder fails, use the built-in non-GGUF fallback or
        preserve the user's original prompt instead of aborting the image job.
        """
        original = str(prompt_text or "").strip()
        if not original or workflow not in {"image_generate", "image_edit", "lora_closeup_1024", "enhance", "square_1_1"}:
            return original
        if original in {"the main subject detail", "main subject"}:
            return original
        id_lock_suffix = "Preserve the main subject’s identity, defining features, proportions, and expression."
        comp_lock_suffix = "Change only what is requested. Keep all unmentioned content, framing, perspective, lighting, and background unchanged."
        has_id_lock = id_lock_suffix in original
        has_comp_lock = comp_lock_suffix in original
        core_text = original
        if has_id_lock or has_comp_lock:
            core_text = core_text.replace(id_lock_suffix, "").replace(comp_lock_suffix, "").strip()
            if not core_text:
                return original

        def _reattach_locks(translated_core: str) -> str:
            res = translated_core.strip()
            if not (has_id_lock or has_comp_lock):
                return res
            parts_out = [res.rstrip(" .") + "."] if res else []
            if has_id_lock:
                parts_out.append(id_lock_suffix)
            if has_comp_lock:
                parts_out.append(comp_lock_suffix)
            return " ".join(p for p in parts_out if p.strip())

        try:
            folder = Path(self.model_path).expanduser().resolve()
            loaded_path = getattr(self, "_loaded_model_path", None)
            loaded_parts = getattr(self, "_loaded_parts", {})
            if loaded_path and Path(loaded_path).expanduser().resolve() == folder and loaded_parts:
                parts = dict(loaded_parts)
            else:
                parts = self._find_model_parts(folder)
            encoder = parts.get("llm")
            projector = parts.get("llm_vision")
            if encoder is None or Path(str(encoder)).suffix.lower() != ".gguf":
                resolved_m, resolved_v = resolve_closeup_grounding_gguf_pair(encoder, projector)
                if resolved_m is not None:
                    encoder = resolved_m
                    if projector is None and resolved_v is not None:
                        projector = resolved_v
            if encoder is None or Path(str(encoder)).suffix.lower() != ".gguf":
                fallback_en = translate_prompt_non_gguf_fallback(core_text)
                if fallback_en and fallback_en != core_text:
                    self.log(f"Translated {workflow.replace('_', ' ')} prompt to English with the built-in ConvRot/non-GGUF translator.")
                    self.status_callback("Prompt translation complete · continuing with image generation")
                    return _reattach_locks(fallback_en)
                self.log(
                    "Prompt translation skipped: the selected Model Catalog needs a compatible GGUF Qwen3-VL encoder "
                    "and matching vision projector. Original prompt will be used."
                )
                self.status_callback("Prompt translation unavailable · using the original prompt")
                return original
            cache_key = (str(Path(encoder).resolve()), core_text)
            cached = self._prompt_translation_cache.get(cache_key)
            if cached:
                self._prompt_translation_cache.pop(cache_key, None)
                self._prompt_translation_cache[cache_key] = cached
                self.log("Reused cached local English prompt translation.")
                return _reattach_locks(cached)
            self.status_callback("Local text encoder · translating prompt to English…")
            translated = translate_prompt_via_local_encoder(
                core_text, model_path=encoder, mmproj_path=projector,
                timeout_seconds=90.0, shutdown_after_request=True, gpu_layers=0,
            ).strip()
            if not translated:
                raise RuntimeError("the local text encoder returned an empty translation")
            self._prompt_translation_cache[cache_key] = translated
            while len(self._prompt_translation_cache) > 64:
                self._prompt_translation_cache.pop(next(iter(self._prompt_translation_cache)))
            self.log(f"Translated {workflow.replace('_', ' ')} prompt to English with the selected local text encoder.")
            self.status_callback("Prompt translation complete · continuing with image generation")
            return _reattach_locks(translated)
        except Exception as exc:
            self.log(
                f"Local prompt translation failed ({type(exc).__name__}); preserving the original prompt. "
                "Check that the selected GGUF encoder, matching mmproj, and llama-server runtime are available."
            )
            self.status_callback("Prompt translation unavailable · using the original prompt")
            return original

    def _publish_live_preview(self, image: Image.Image, marker: Any = None) -> None:
        """Publish a compact, real image snapshot between inference pipeline stages.

        Native sd-server sampler previews are still used when available. This extra
        path guarantees that the completed base reconstruction and progressively
        blended detail tiles reach the UI even on server builds that omit preview
        frames for img2img or Turbo jobs.
        """
        if image is None or bool(getattr(self, "_suppress_live_preview", False)):
            return
        try:
            preview = image.convert("RGB").copy()
            preview.thumbnail((512, 512), Image.Resampling.BICUBIC)
            self.latest_preview_image = preview
            self.latest_preview_revision = int(getattr(self, "latest_preview_revision", 0) or 0) + 1
            self._latest_preview_marker = ("stage", marker, self.latest_preview_revision)
            self.latest_preview_step = 0
            self.latest_preview_total_steps = 0
            self.latest_preview_pass = 0
        except Exception as exc:
            self.log(f"Live preview snapshot skipped ({type(exc).__name__}).")

    @staticmethod
    def _inference_size(size: tuple[int, int]) -> tuple[int, int]:
        w, h = int(size[0]), int(size[1])
        scale = min(1.0, MAX_INFERENCE_SIDE / max(w, h))
        # Qwen-Image 2.1's dimensioning is safest when both sides are multiples of 32.
        def m32(value):
            return max(64, int(round((value * scale) / 32.0) * 32))
        return m32(w), m32(h)

    @staticmethod
    def _ultimate_stage1_size(target_size: tuple[int, int], ultimate_enabled: bool, ultimate_scale: int = 2) -> tuple[int, int]:
        target_w, target_h = max(64, int(target_size[0])), max(64, int(target_size[1]))
        if not ultimate_enabled:
            return target_w, target_h
        scale = max(1, min(4, int(ultimate_scale or 2)))
        if scale <= 1:
            return target_w, target_h
        s1_w = max(64, int(round((target_w / float(scale)) / 32.0) * 32))
        s1_h = max(64, int(round((target_h / float(scale)) / 32.0) * 32))
        return s1_w, s1_h

    def _infer(self, prompt: str, source: Optional[Image.Image], size: tuple[int, int],
               seed: int, quality: str, steps: int = 8,
               fit_source_to_output: bool = True,
               init_image: Optional[Image.Image] = None,
               mask_image: Optional[Image.Image] = None,
               strength: float = 0.95,
               compensate_img2img_steps: bool = False,
               skip_ref_images: bool = False) -> Image.Image:
        if self.force_stop.is_set():
            raise Aborted()
        self.ensure_loaded()
        width, height = self._inference_size(size)
        condition = None
        if source is not None and not skip_ref_images:
            source_rgb = ImageOps.exif_transpose(source).convert("RGB")
            # Keep reference conditioning (ref_images) up to native 2K (<=2048 px and never upscaled above source_rgb)
            # on the 32-pixel grid so Qwen3-VL mmproj and VAE ref_latents retain full 2K photographic detail.
            ref_max_side = min(MAX_INFERENCE_SIDE, max(width, height), max(source_rgb.width, source_rgb.height))
            ref_scale = min(1.0, ref_max_side / max(1, source_rgb.width, source_rgb.height))
            ref_w = max(64, int(round((source_rgb.width * ref_scale) / 32.0) * 32))
            ref_h = max(64, int(round((source_rgb.height * ref_scale) / 32.0) * 32))
            if fit_source_to_output:
                if abs((source_rgb.width / max(1, source_rgb.height)) - (width / max(1, height))) <= 0.05:
                    condition = source_rgb if source_rgb.size == (ref_w, ref_h) else source_rgb.resize((ref_w, ref_h), Image.Resampling.BICUBIC)
                else:
                    fitted = ImageOps.contain(source_rgb, (ref_w, ref_h), Image.Resampling.BICUBIC)
                    background = source_rgb.resize((1, 1), Image.Resampling.BICUBIC).getpixel((0, 0))
                    condition = Image.new("RGB", (ref_w, ref_h), background)
                    condition.paste(fitted, ((ref_w - fitted.width) // 2, (ref_h - fitted.height) // 2))
            else:
                # Outpainting must supply the genuine original aspect ratio, not synthesize flat bars.
                condition = source_rgb if source_rgb.size == (ref_w, ref_h) else source_rgb.resize((ref_w, ref_h), Image.Resampling.BICUBIC)

        seed_value = int(seed) if int(seed) >= 0 else random.randint(0, 2**31 - 1)
        diffusion_name = Path(str(self._loaded_parts.get("diffusion", ""))).name.lower()
        is_turbo = "turbo" in diffusion_name
        requested_steps = max(4, min(100, int(steps)))
        steps = requested_steps
        sample_params: dict[str, Any] = {"sample_method": "euler", "sample_steps": steps}
        if is_turbo:
            if requested_steps != 8 and not compensate_img2img_steps:
                self.log(
                    f"Qwen Turbo schedule lock · requested {requested_steps} steps overridden to the required 8-step Euler schedule (CFG=1.0)."
                )
                steps = 8
            sample_params.update(qwen_turbo_sampling_params())
            if compensate_img2img_steps:
                sample_params.update(
                    qwen_turbo_tile_sampling_params(
                        steps=requested_steps,
                        strength=strength if init_image is not None else None,
                        compensate_img2img_steps=bool(init_image is not None),
                    )
                )
        else:
            # Non-Turbo local denoisers use the standard adjustable Qwen-Image-2.1 schedule.
            quality_lower = str(quality).casefold()
            if not compensate_img2img_steps:
                if quality_lower.startswith("fast"):
                    steps = min(steps, 12)
                elif quality_lower.startswith("balanced"):
                    steps = min(steps, 20)
            schedule_steps = steps
            if compensate_img2img_steps and init_image is not None:
                eff_s = max(0.05, min(1.0, float(strength)))
                if eff_s < 0.999:
                    schedule_steps = max(steps, int(round(steps / eff_s)))
                    while int(schedule_steps * eff_s) < steps and schedule_steps < 1200:
                        schedule_steps += 1
                    while schedule_steps > steps and int((schedule_steps - 1) * eff_s) >= steps:
                        schedule_steps -= 1
            sample_params["sample_steps"] = schedule_steps
            # Keep prompt importance at 1.0 for all local Qwen checkpoints as requested.
            sample_params["guidance"] = {"txt_cfg": 1.0}

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": "",
            "width": width,
            "height": height,
            "seed": seed_value,
            "batch_count": 1,
            "embed_image_metadata": True,
            # Native sd-server can return a base64 preview frame from each sampled step.
            # TAE is the clearer default; projection/VAE remain explicit speed/quality alternatives.
            "preview": str(self.live_preview_method or "tae") if str(self.live_preview_method or "tae") in {"proj", "tae", "vae"} else "none",
            "preview_interval": max(1, min(20, int(self.live_preview_interval or 1))) if str(self.live_preview_method or "tae") != "none" else 1,
            "sample_params": sample_params,
        }
        if condition is not None or init_image is not None:
            llm_path = self._loaded_parts.get("llm")
            has_embedded_vision = _text_encoder_embeds_vision(llm_path)
            if not self._loaded_parts.get("llm_vision") and not has_embedded_vision:
                if init_image is None and condition is not None:
                    init_image = condition
                condition = None
                self.log(
                    "Vision projector (mmproj) is not active; conditioning image through VAE init_image without requiring mmproj."
                )
            if condition is not None:
                buffer = io.BytesIO()
                condition.save(buffer, format="PNG", compress_level=1)
                payload["ref_images"] = [base64.b64encode(buffer.getvalue()).decode("ascii")]
            if init_image is not None:
                # Qwen-Image 2.1 supports RGBA conditioning. Keep alpha transparent in
                # the new canvas margins so those pixels are not interpreted as image content.
                init = ImageOps.exif_transpose(init_image)
                if init.mode not in ("RGBA", "LA"):
                    init = init.convert("RGB")
                if init.size != (width, height):
                    init = init.resize((width, height), Image.Resampling.BICUBIC)
                buffer = io.BytesIO()
                init.save(buffer, format="PNG", compress_level=1)
                payload["init_image"] = base64.b64encode(buffer.getvalue()).decode("ascii")
                payload["strength"] = max(0.0, min(1.0, float(strength)))
                if mask_image is not None:
                    mask = ImageOps.exif_transpose(mask_image).convert("L")
                    if mask.size != (width, height):
                        mask = mask.resize((width, height), Image.Resampling.NEAREST)
                    buffer = io.BytesIO()
                    mask.save(buffer, format="PNG")
                    payload["mask_image"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        if self.lora_enabled and self.lora_path:
            lora = Path(self.lora_path)
            if not lora.is_file():
                raise RuntimeError(f"Local LoRA file was not found: {lora}")
            if "restore" in lora.name.lower():
                self.log(
                    f"Qwen Restore LoRA active ({lora.name}, strength={self.lora_strength:.2f}) · "
                    "recommended: 1.0 for in-focus compressed photos, 0.75 for extra Qwen texture, 0.50 for blurry/out-of-focus photos."
                )
            payload["lora"] = [{"path": lora.name, "multiplier": float(self.lora_strength), "is_high_noise": False}]

        pipeline_context = str(getattr(self, "_active_pipeline_context", "") or "").strip()
        context_prefix = (pipeline_context + " · ") if pipeline_context else ""
        if condition is not None or init_image is not None:
            self.status_callback(f"{context_prefix}VAE encode / text conditioning · preparing {width}×{height} · {steps} steps")
        else:
            self.status_callback(f"{context_prefix}Text encoder · prompt conditioning · preparing {width}×{height} · {steps} steps")
        log_cursor = self.server_log_path.stat().st_size if self.server_log_path and self.server_log_path.exists() else 0
        try:
            job = self._request_json("POST", "/sdcpp/v1/img_gen", payload, timeout=20.0)
        except RuntimeError as exc:
            if self.lora_enabled and self.lora_path and ("HTTP 400" in str(exc) or "invalid lora" in str(exc).lower()):
                raise RuntimeError(
                    "The local engine rejected the selected LoRA. P.AI now sends a relative filename through --lora-model-dir; "
                    "if this error remains, the LoRA weights are not compatible with the loaded Qwen Image 2.1 checkpoint "
                    "or this sd-server build. Try a Qwen Image 2.1-compatible LoRA and inspect the sd-server log.\n\n" + str(exc)
                ) from exc
            raise
        job_id = str(job.get("id", ""))
        if not job_id:
            raise RuntimeError(f"The local engine did not return a job ID: {job}")
        self.log(f"Submitted local generation job {job_id} ({width}×{height}, seed {seed_value}).")
        poll_path = "/sdcpp/v1/jobs/" + urllib.parse.quote(job_id, safe="")
        inference_deadline = time.monotonic() + 3 * 60 * 60
        while True:
            if time.monotonic() >= inference_deadline:
                raise RuntimeError("Image generation exceeded the 3-hour timeout. Check the engine log.")
            if self.force_stop.is_set():
                try:
                    self._request_json("POST", poll_path + "/cancel", {}, timeout=3.0)
                except Exception:
                    pass
                raise Aborted()
            try:
                status_obj = self._request_json("GET", poll_path, timeout=1.5)
            except RuntimeError as exc:
                if self.force_stop.is_set():
                    try:
                        self._request_json("POST", poll_path + "/cancel", {}, timeout=1.0)
                    except Exception:
                        pass
                    raise Aborted() from exc
                message = str(exc).lower()
                if "timed out" in message or "timeout" in message or "temporarily unavailable" in message:
                    # Status polling is allowed to time out transiently. A bounded
                    # poll timeout makes Run/Stop respond promptly even on a loaded server.
                    self.force_stop.wait(0.05)
                    continue
                raise
            status = str(status_obj.get("status", "")).lower()
            log_cursor, log_delta = self._read_log_window(self.server_log_path, log_cursor)
            log_lower = log_delta.lower()
            if status in {"failed", "error", "cancelled", "canceled"}:
                detail = status_obj.get("error") or status_obj.get("message") or status_obj
                raise RuntimeError(f"The image engine stopped with status {status}: {detail}")
            if status in {"completed", "succeeded", "success"}:
                result = status_obj.get("result") or {}
                images = result.get("images") or status_obj.get("images") or []
                if not images:
                    # Accept common native-result aliases as pinned server builds evolve.
                    single = result.get("image") or result.get("image_base64") or result.get("b64_json")
                    if single:
                        images = [single]
                encoded = None
                if images:
                    first_image = images[0]
                    if isinstance(first_image, str):
                        encoded = first_image
                    elif isinstance(first_image, dict):
                        encoded = (first_image.get("b64_json") or first_image.get("base64")
                                   or first_image.get("data") or first_image.get("image_base64"))
                if not encoded:
                    raise RuntimeError(
                        "The engine reported completion but did not return a readable image. "
                        "Check the engine log and API version. Response: " + str(status_obj)[:1800]
                    )
                if isinstance(encoded, str) and encoded.startswith("data:") and "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                try:
                    raw = base64.b64decode(encoded, validate=False)
                    with Image.open(io.BytesIO(raw)) as img:
                        result_image = ImageOps.exif_transpose(img).convert("RGB")
                except Exception as exc:
                    raise RuntimeError(f"Failed to read the output image from the local engine: {exc}") from exc
                if result_image.size != size:
                    result_image = result_image.resize(size, Image.Resampling.BICUBIC)
                self.status_callback(f"{context_prefix}VAE decode complete · image ready · seed {seed_value}.")
                self.log(f"Local inference finished · {width}×{height} → {size[0]}×{size[1]} · seed {seed_value}.")
                self._publish_live_preview(result_image, marker=("inference-complete", job_id))
                return result_image
            # Native sd-server status uses preview={step,total_steps,pass,b64_json}; older
            # forks may expose preview_info or top-level preview_base64. Decode the actual
            # returned intermediate frame and retain its counters for the live-preview HUD.
            small_preview, step_num, total_num, preview_pass = _parse_sdcpp_preview_status(status_obj)
            if is_turbo and total_num == steps - 1:
                total_num = steps
                step_num = min(steps, step_num + 1) if step_num > 0 else 0
            if small_preview is not None and not bool(getattr(self, "_suppress_live_preview", False)):
                marker = (job_id, int(step_num or 0), int(total_num or 0), int(preview_pass or 0))
                # Native servers often return the same preview image in every poll.
                # Publish only when sampler metadata advances, avoiding redundant GUI
                # conversions while retaining each actual new sampler frame.
                if self.latest_preview_image is None or marker != self._latest_preview_marker:
                    self.latest_preview_image = small_preview
                    self.latest_preview_revision += 1
                    self._latest_preview_marker = marker
            if step_num:
                self.latest_preview_step = step_num
            if total_num:
                self.latest_preview_total_steps = total_num
            if preview_pass:
                self.latest_preview_pass = preview_pass
            if step_num and total_num:
                if step_num and total_num and step_num >= total_num:
                    self.status_callback(f"{context_prefix}VAE decode · denoising complete ({step_num}/{total_num}); decoding image…")
                elif step_num and total_num:
                    self.status_callback(f"{context_prefix}Qwen diffusion · denoising step {step_num}/{total_num}…")
                else:
                    self.status_callback(f"{context_prefix}Text encoder / prompt conditioning · backend is preparing the sampler…")
            elif any(marker in log_lower for marker in ("decode_first_stage", "latent 1 decoded", "vae decode")):
                self.status_callback(f"{context_prefix}VAE decode · converting latent to image…")
            elif any(marker in log_lower for marker in ("encode_first_stage", "target t_enc")):
                self.status_callback(f"{context_prefix}VAE encode · preparing image latents…")
            elif "conditioning cache hit" in log_lower:
                self.status_callback(f"{context_prefix}Text conditioning cache · reusing encoded prompt…")
            elif any(marker in log_lower for marker in ("conditioning", "tokeniz", "prompt")):
                self.status_callback(f"{context_prefix}Text encoder · prompt conditioning…")
            else:
                self.status_callback(f"{context_prefix}Text encoder / prompt conditioning · backend {status or 'working'}…")
            self.force_stop.wait(0.10)

    @staticmethod
    def _looks_like_cuda_oom(exc: Exception, log_tail: str = "") -> bool:
        message = (str(exc) + "\n" + str(log_tail)).casefold()
        return any(token in message for token in (
            "out of memory", "cuda_error_out_of_memory", "cuda out-of-memory",
            "failed to allocate", "allocation failed", "cudamalloc", "not enough memory",
        ))

    def _infer_with_auto_offload_retry(self, prompt: str, source: Optional[Image.Image], size: tuple[int, int],
                                       seed: int, quality: str, steps: int = 8,
                                       fit_source_to_output: bool = True,
                                       init_image: Optional[Image.Image] = None,
                                       mask_image: Optional[Image.Image] = None,
                                       strength: float = 0.95,
                                       compensate_img2img_steps: bool = False,
                                       skip_ref_images: bool = False) -> Image.Image:
        try:
            return self._infer(prompt, source, size, seed, quality, steps, fit_source_to_output,
                               init_image, mask_image, strength,
                               compensate_img2img_steps=compensate_img2img_steps,
                               skip_ref_images=skip_ref_images)
        except Aborted:
            raise
        except Exception as exc:
            if self.force_stop.is_set() or self.gpu_memory_mode != "auto" or self._last_offload_to_cpu or self.auto_offload_override:
                raise
            log_tail = self._tail_file(self.server_log_path, 5000) if self.server_log_path else ""
            if not self._looks_like_cuda_oom(exc, log_tail):
                raise
            self.auto_offload_override = True
            self.log("CUDA memory allocation failed. Retrying this image once with --offload-to-cpu; diffusion/VAE computation remain on the selected CUDA GPU, but this retry can be slower.")
            self.status_callback("CUDA memory shortage · restarting with low-VRAM weight offload and retrying once…")
            self.unload_model()
            self.ensure_loaded()
            return self._infer(prompt, source, size, seed, quality, steps, fit_source_to_output,
                               init_image, mask_image, strength,
                               compensate_img2img_steps=compensate_img2img_steps,
                               skip_ref_images=skip_ref_images)

    def enhance_once(self, source: Image.Image, prompt: str, width: int, height: int,
                     seed: int, quality: str = "Turbo · 8 steps", steps: int = 8,
                     use_img2img: bool = True, strength: Optional[float] = None) -> Image.Image:
        """Qwen enhancement using both source reference conditioning and bounded Img2Img.

        The original is sent as `ref_images` and `init_image`: the reference helps retain identity,
        framing and semantics, while denoising at the selected profile strength gives the model room to recover visible detail.
        This is a calibrated starting point, not a pixel-perfect identity guarantee.
        """
        strength = ENHANCE_IMG2IMG_STRENGTH if strength is None else max(0.0, min(0.95, float(strength)))
        if use_img2img:
            self.log(
                f"Qwen Enhance · source used as reference + init image · "
                f"Img2Img strength={strength:.2f}."
            )
            return self._infer_with_auto_offload_retry(
                prompt, source, (width, height), seed, quality, steps,
                fit_source_to_output=True, init_image=source,
                strength=strength,
            )
        return self._infer_with_auto_offload_retry(
            prompt, source, (width, height), seed, quality, steps,
            fit_source_to_output=True,
        )

    def _enhance_with_retries(self, original: Image.Image, prompt: str, width: int, height: int,
                              seed: int, quality: str, max_attempts: int = 1,
                              retry_diff: float = 25.0, warn_diff: float = 18.0,
                              steps: int = 8, use_img2img: bool = True,
                              strength: Optional[float] = None) -> Image.Image:
        baseline = ImageOps.exif_transpose(original).convert("RGB")
        best, best_diff = None, float("inf")
        for attempt in range(max(1, min(8, max_attempts))):
            if self.force_stop.is_set():
                raise Aborted()
            this_seed = seed if attempt == 0 and seed >= 0 else random.randint(0, 2**31 - 1)
            result = self.enhance_once(original, prompt, width, height, this_seed, quality, steps, use_img2img=use_img2img, strength=strength)
            diff = mean_difference(baseline, result)
            self.log(f"Enhance attempt {attempt + 1}/{max_attempts}: structural difference {diff:.2f} (seed {this_seed})")
            if diff < best_diff:
                best, best_diff = result, diff
            if diff <= retry_diff:
                break
        if best is None:
            raise RuntimeError("No enhancement attempt produced an image.")
        if best_diff > warn_diff:
            self.log(f"Warning: best structural difference is {best_diff:.2f}, above warning threshold {warn_diff:.2f}.")
        self.log("Qwen-Image 2.1 full-frame enhancement pass complete; external Real-ESRGAN/Real-ESRNet is not chained into this workflow.")
        return best

    @staticmethod
    def _tile_starts(length: int, tile_length: int, max_tiles: Optional[int] = None,
                     min_overlap_ratio: float = 0.0) -> list[int]:
        """Return evenly distributed starts with full coverage and optional minimum overlap."""
        length = max(1, int(length))
        tile_length = max(64, min(length, int(tile_length)))
        if length <= tile_length:
            return [0]
        overlap_ratio = max(0.0, min(0.75, float(min_overlap_ratio)))
        max_stride = max(1, int(math.floor(tile_length * (1.0 - overlap_ratio))))
        last = length - tile_length
        # Pick enough starts that neighboring tiles overlap by at least the requested
        # fraction of their tile width/height, preventing unprocessed seams at high scales.
        count = max(2, int(math.ceil(last / max_stride)) + 1, int(math.ceil(length / tile_length)))
        if max_tiles is not None:
            # max_tiles is a minimum hint, never permission to leave uncovered pixels.
            count = max(count, max(2, int(max_tiles)))
        return sorted(set(int(round(i * last / (count - 1))) for i in range(count)))

    @staticmethod
    def _tile_fade_mask(size: tuple[int, int], fade_left: int, fade_top: int) -> Image.Image:
        """Smooth cosine leading-edge feather mask used to blend adjacent Qwen-refined tiles without linear seam lines."""
        width, height = size
        mask = Image.new("L", size, 255)
        if fade_left > 1:
            horizontal = Image.new("L", (width, 1), 255)
            values = [
                min(255, int(round(127.5 * (1.0 - math.cos(math.pi * x / (fade_left - 1))))))
                if x < fade_left else 255
                for x in range(width)
            ]
            horizontal.putdata(values)
            mask = ImageChops.multiply(mask, horizontal.resize(size))
        if fade_top > 1:
            vertical = Image.new("L", (1, height), 255)
            values = [
                min(255, int(round(127.5 * (1.0 - math.cos(math.pi * y / (fade_top - 1))))))
                if y < fade_top else 255
                for y in range(height)
            ]
            vertical.putdata(values)
            mask = ImageChops.multiply(mask, vertical.resize(size))
        return mask

    def _qwen_detail_refine_large_output(self, image: Image.Image, prompt: str, seed: int,
                                         quality: str, steps: int, detail_mode: str = "max_detail",
                                         use_img2img: bool = True, tile_strength: Optional[float] = None,
                                         detail_profile: str = "balanced", force: bool = False) -> Image.Image:
        """Use Qwen itself for local detail refinement after its full-frame pass exceeds 2048 px.

        This avoids using Real-ESRGAN in the Enhance workflow. A global Qwen pass sets the
        composition; overlapping local passes then restore texture at the requested output size.
        """
        image = image.convert("RGB")
        width, height = image.size
        detail_mode = str(detail_mode or "ultimate").strip().lower()
        if detail_mode not in {"fast", "balanced", "max_detail", "ultimate"}:
            detail_mode = "ultimate"
        if detail_mode == "fast":
            if max(width, height) > MAX_INFERENCE_SIDE:
                self.log("Ultimate-style Qwen refinement is OFF · skipping tiled denoise pass.")
            return image
        if max(width, height) <= MAX_INFERENCE_SIDE and not force:
            return image
        # The selected Qwen profile controls prompt guidance/Img2Img strength. The optional
        # Ultimate uses an overlap-aware grid of Qwen crops; larger output scales may need more than four.
        profile_key = str(detail_profile or "balanced").strip().lower()
        if profile_key not in QWEN_DETAIL_PROFILES:
            profile_key = "balanced"
        if detail_mode == "ultimate":
            # Ultimate Tile Refinement 4-tile (2x2 grid) mode:
            # Directly honor the Model Catalog Tile coverage + overlap setting (45%..75%) without hidden inflation.
            coverage = max(0.45, min(0.75, float(getattr(self, "_active_ultimate_tile_coverage", 0.50))))
            tile_w = min(width, max(64, int(round(width * coverage))))
            tile_h = min(height, max(64, int(round(height * coverage))))
            if width >= 128 and tile_w >= width:
                tile_w = max(64, int(round(width * coverage)))
            if height >= 128 and tile_h >= height:
                tile_h = max(64, int(round(height * coverage)))
            configured_tile_strength = float(getattr(self, "_active_ultimate_denoise_strength", 0.20))
            profile_tile_strength = float(tile_strength) if tile_strength is not None else 0.20
            if "_active_ultimate_denoise_strength" in getattr(self, "__dict__", {}):
                effective_tile_strength = max(0.05, min(0.95, configured_tile_strength))
            else:
                effective_tile_strength = max(0.05, min(0.95, max(configured_tile_strength, profile_tile_strength)))
        else:
            coverage_limit = MAX_INFERENCE_SIDE if detail_mode == "balanced" else 1536
            profile_limit = QWEN_DETAIL_PROFILE_TILE_LIMITS[profile_key]
            tile_limit = min(coverage_limit, profile_limit)
            tile_w = min(width, tile_limit)
            tile_h = min(height, tile_limit)
        if detail_mode == "ultimate":
            # Full-coverage starts: 50% coverage remains a 2×2 grid, while smaller
            # coverage automatically adds rows/columns instead of leaving untreated gaps.
            x_starts = self._tile_starts(width, tile_w)
            y_starts = self._tile_starts(height, tile_h)
            refinement_passes = max(1, min(2, int(getattr(self, "_active_ultimate_refinement_passes", 1))))
            tile_label = f"Ultimate Tile Refinement · full-coverage {len(x_starts)}×{len(y_starts)} Qwen grid · denoise {effective_tile_strength:.2f} · coverage {coverage:.0%} · passes {refinement_passes}"
        else:
            x_starts = self._tile_starts(width, tile_w)
            y_starts = self._tile_starts(height, tile_h)
            refinement_passes = 2 if profile_key == "maximum" else 1
            tile_label = "Balanced" if detail_mode == "balanced" else "Maximum detail"
        tiles_per_pass = len(x_starts) * len(y_starts)
        self.log(f"Enhance detail mode={tile_label} · profile={profile_key} · tile={tile_w}×{tile_h} · grid={len(x_starts)}×{len(y_starts)} · tiles/pass={tiles_per_pass} · repeat passes={refinement_passes} · total Qwen calls={tiles_per_pass * refinement_passes}.")
        result = image.copy()
        # Keep the original workflow prompt attached as semantic context. The text explicitly
        # forbids executing edit/outpaint instructions again inside a finished detail crop.
        task_context = str(prompt or "").strip()
        for previous_guidance in QWEN_DETAIL_PROFILE_PROMPTS.values():
            task_context = task_context.replace(previous_guidance, "").strip()
        if len(task_context) > 900:
            task_context = task_context[:900].rstrip() + "…"
        context_line = (
            " Original workflow prompt, for subject/material consistency only—not a command to redo the task inside this crop: "
            + task_context + "."
        ) if task_context else ""
        watermark_requested = "WATERMARK REMOVAL TASK:" in str(prompt or "")
        preserve_rule = (
            "Do not simply sharpen, add halos, overemphasize grain, outpaint, remove scene objects, redraw subjects, "
            "restyle, relocate or invent objects or facial features. The only permitted content removal is a visible "
            "watermark overlay explicitly requested by the WATERMARK REMOVAL TASK; reconstruct the obscured pixels "
            "from adjacent source texture without smearing. Do not change this crop or composition."
            if watermark_requested else
            "Do not simply sharpen, add halos, overemphasize grain, outpaint, remove, redraw, restyle, relocate or invent "
            "objects or facial features. Do not change this crop or composition."
        )
        local_prompt = apply_qwen_detail_profile_prompt((
            "Faithful local denoise and natural high-resolution detail restoration of this already-composed image crop. The crop is finished. "
            "First remove stochastic luminance/chroma noise, compression blocks/ringing, resampling grit, tiny pixel speckles, "
            "and jagged/aliased edges; do not turn those artifacts into stronger texture. Keep smooth skin gradients naturally smooth, "
            "supple, and clean—do not add artificial skin pores, peach fuzz, pixelated stippling, or plastic texture that is not supported by the source. "
            "Recover only genuine source-supported edge clarity, individual hair/eyelash strands, clean textile weave, and natural surface transitions. "
            "Preserve this exact framing, visible content, object count, identity, geometry, text, perspective, colors, lighting and composition. "
            + preserve_rule + context_line
        ), detail_profile)
        total = tiles_per_pass * refinement_passes
        tile_quality = quality
        tile_steps = steps
        if detail_mode == "ultimate":
            tile_steps = max(4, min(60, int(getattr(self, "_active_ultimate_steps", 8))))
            tile_quality = f"Ultimate · {tile_steps} steps"
        if detail_mode != "ultimate":
            effective_tile_strength = tile_strength if tile_strength is not None else ENHANCE_TILE_IMG2IMG_STRENGTH
        effective_tile_strength = max(0.0, min(0.95, float(effective_tile_strength)))
        for pass_index in range(refinement_passes):
            # Each serial pass starts from the complete result of the previous pass; this makes
            # Maximum a real extra reconstruction pass rather than just a different label.
            pass_input = result.copy()
            pass_result = pass_input.copy()
            if pass_index:
                pass_prompt = local_prompt + "\n\nSECOND CLEANUP PASS: remove residual noise and remaining broken/pixelated edge transitions left after the first reconstruction. Improve natural material texture modestly; preserve the first pass's correct details and do not invent new content."
            else:
                pass_prompt = local_prompt
            for yi, top in enumerate(y_starts):
                for xi, left in enumerate(x_starts):
                    if self.force_stop.is_set():
                        raise Aborted()
                    right, bottom = min(width, left + tile_w), min(height, top + tile_h)
                    # Like Ultimate SD Upscale (USDURedraw), crop each sequential tile from pass_result
                    # so overlapping borders include the already-refined pixels from previous tiles, while
                    # keeping source_tile from pass_input for reference conditioning and color matching.
                    source_tile = pass_input.crop((left, top, right, bottom))
                    tile = pass_result.crop((left, top, right, bottom))
                    fade_left = max(0, x_starts[xi - 1] + tile_w - left) if xi else 0
                    fade_top = max(0, y_starts[yi - 1] + tile_h - top) if yi else 0
                    # Refine each output tile with the same Qwen model at native resolution (with full reference conditioning).
                    tile_index = pass_index * tiles_per_pass + yi * len(x_starts) + xi + 1
                    is_ultimate_pass = detail_mode == "ultimate"
                    stage_name = "Ultimate Qwen refinement" if is_ultimate_pass else "Qwen detail refinement"
                    self._active_pipeline_context = stage_name
                    self.status_callback(f"{stage_name} · pass {pass_index + 1}/{refinement_passes} · tile {tile_index}/{total} · {width}×{height}")
                    self.log(f"Qwen native detail tile {tile_index}/{total} · profile={profile_key} · pass={pass_index + 1}/{refinement_passes} · steps={tile_steps} · strength={effective_tile_strength:.2f} · overlap=({fade_left},{fade_top}) · crop={left},{top},{right},{bottom}.")
                    previous_preview_suppression = bool(getattr(self, "_suppress_live_preview", False))
                    if is_ultimate_pass:
                        self._suppress_live_preview = False
                        self._publish_live_preview(tile, marker=("ultimate-tile-active", pass_index + 1, tile_index, total))
                    else:
                        self._suppress_live_preview = True
                    try:
                        if use_img2img:
                            refined = self._infer_with_auto_offload_retry(
                                pass_prompt, tile, tile.size, int(seed) + tile_index, tile_quality, tile_steps,
                                fit_source_to_output=True, init_image=tile, strength=effective_tile_strength,
                                compensate_img2img_steps=True, skip_ref_images=False,
                            )
                        else:
                            refined = self._infer_with_auto_offload_retry(
                                pass_prompt, tile, tile.size, int(seed) + tile_index, tile_quality, tile_steps,
                                fit_source_to_output=True,
                            )
                    finally:
                        self._suppress_live_preview = previous_preview_suppression
                        self._active_pipeline_context = ""
                    color_matcher = globals().get("match_tile_color_to_reference")
                    if callable(color_matcher):
                        refined = color_matcher(
                            refined, source_tile, strength=0.90,
                            overlap_tile=tile if (xi > 0 or yi > 0) else None,
                            overlap_left=fade_left, overlap_top=fade_top,
                        )
                    if is_ultimate_pass:
                        tile_blender = globals().get("blend_ultimate_tile_by_denoise")
                        if callable(tile_blender):
                            refined = tile_blender(source_tile, refined, effective_tile_strength)
                        elif effective_tile_strength < 0.65:
                            blend_alpha = max(0.10, min(1.0, float(effective_tile_strength) / 0.65))
                            refined = Image.blend(source_tile, refined, blend_alpha)
                    if is_ultimate_pass:
                        self._publish_live_preview(refined, marker=("ultimate-tile-complete", pass_index + 1, tile_index, total))
                    mask_fade_left = fade_left if fade_left > 0 else (min(24, max(4, tile.width // 16)) if xi > 0 else 0)
                    mask_fade_top = fade_top if fade_top > 0 else (min(24, max(4, tile.height // 16)) if yi > 0 else 0)
                    mask = self._tile_fade_mask(tile.size, mask_fade_left, mask_fade_top)
                    pass_result.paste(refined, (left, top), mask)
            result = pass_result
            # Publish only the completely blended canvas. Intermediate tiles are not
            # valid whole-image previews and look like a checkerboard/mosaic in the UI.
            self._publish_live_preview(result, marker=("tile-pass-complete", pass_index + 1, tiles_per_pass))
        return result

    def _enhance_qwen_output(self, original: Image.Image, prompt: str, width: int, height: int,
                             seed: int, quality: str, max_attempts: int, retry_diff: float,
                             warn_diff: float, steps: int, detail_mode: str = "fast",
                             use_img2img: bool = True, strength: Optional[float] = None,
                             tile_strength: Optional[float] = None, detail_profile: str = "balanced",
                             ultimate_scale: int = 2,
                             upscale_options: Optional[dict[str, Any]] = None) -> Image.Image:
        detail_mode_norm = str(detail_mode or "fast").strip().lower()
        stage_width, stage_height = int(width), int(height)
        requested_width = min(MAX_OUTPUT_SIDE, max(64, int(width)))
        requested_height = min(MAX_OUTPUT_SIDE, max(64, int(height)))
        requested_ultimate_scale = max(1, min(4, int(ultimate_scale or 2))) if detail_mode_norm == "ultimate" else 1
        effective_ultimate_scale = float(requested_ultimate_scale)
        if detail_mode_norm == "ultimate" and requested_ultimate_scale > 1:
            stage_width = max(64, int(round((requested_width / float(requested_ultimate_scale)) / 32.0) * 32))
            stage_height = max(64, int(round((requested_height / float(requested_ultimate_scale)) / 32.0) * 32))
        else:
            stage_width, stage_height = requested_width, requested_height
        native_width, native_height = self._inference_size((stage_width, stage_height))
        if max(requested_width, requested_height) > MAX_INFERENCE_SIDE:
            if detail_mode_norm == "fast":
                self.log(
                    f"Enhance Fast · native diffusion pass capped at {native_width}×{native_height}; "
                    f"the final file will retain the requested {requested_width}×{requested_height} dimensions. "
                    "Fast mode does not add detail-refinement tiles."
                )
            elif detail_mode_norm == "ultimate":
                self.log(
                    f"Enhance Ultimate · Pass 1 Qwen at {native_width}×{native_height} "
                    f"(1/{requested_ultimate_scale} of requested {requested_width}×{requested_height}); "
                    f"then selected learned AI upscaler ({str((upscale_options or {}).get('upscale_model', 'realesrgan-x4plus'))}) "
                    f"to {requested_width}×{requested_height} and run overlapping Qwen detail tiles."
                )
            else:
                self.log(
                    f"Enhance · global native pass capped at {native_width}×{native_height}; "
                    f"final target remains {requested_width}×{requested_height} before Qwen detail refinement."
                )
        self._publish_live_preview(original, marker=("enhance-source", stage_width, stage_height, requested_width, requested_height))
        profile_key = str(detail_profile or "balanced").strip().lower()
        if detail_mode_norm == "ultimate":
            base = self._enhance_with_retries(
                original, prompt, native_width, native_height, seed, quality,
                max_attempts, retry_diff, warn_diff, steps,
                use_img2img=use_img2img, strength=strength,
            )
            # Preserve the native Qwen reconstruction for the learned upscaler. Do not enlarge
            # it with a fixed interpolation kernel before AI super-resolution.
            result = self._upscale_image_with_selected_model(
                base, upscale_options or {}, (requested_width, requested_height)
            )
            self._publish_live_preview(result, marker=("ultimate-stage-1-ai-upscale", native_width, native_height, requested_width, requested_height))
            self.log(
                f"Ultimate two-stage upscale · Qwen stage-1 reconstruction at {native_width}×{native_height} → "
                f"selected AI model {str((upscale_options or {}).get('upscale_model', 'realesrgan-x4plus'))} → "
                f"{requested_width}×{requested_height} → overlapping Qwen detail refinement. "
                f"Main strength={float(strength if strength is not None else ENHANCE_IMG2IMG_STRENGTH):.2f}."
            )
        else:
            result = self._enhance_with_retries(original, prompt, requested_width, requested_height, seed, quality,
                                                max_attempts, retry_diff, warn_diff, steps, use_img2img=use_img2img, strength=strength)
            if result.size != (requested_width, requested_height):
                result = result.resize((requested_width, requested_height), Image.Resampling.BICUBIC)
        needs_highres_tiles = max(requested_width, requested_height) > MAX_INFERENCE_SIDE and detail_mode_norm != "fast"
        needs_standard_refinement = max(requested_width, requested_height) <= MAX_INFERENCE_SIDE and detail_mode_norm != "fast"
        if needs_highres_tiles or needs_standard_refinement:
            if detail_mode_norm == "ultimate":
                self.log("Ultimate · running full-coverage Qwen detail tiles on the learned AI-upscaled canvas.")
            elif needs_standard_refinement:
                self.log(
                    f"Qwen profile={profile_key} · second denoise/detail pass enabled at {requested_width}×{requested_height}; "
                    f"tile strength={float(tile_strength if tile_strength is not None else ENHANCE_TILE_IMG2IMG_STRENGTH):.2f}."
                )
            else:
                self.log("Requested output exceeds native Qwen inference dimensions; running full-coverage Qwen detail tiles after the global pass.")
            result = self._qwen_detail_refine_large_output(
                result, prompt, seed + 1000, quality, steps,
                detail_mode_norm,
                use_img2img=use_img2img, tile_strength=tile_strength,
                detail_profile=profile_key, force=needs_standard_refinement,
            )
        if result.size != (requested_width, requested_height):
            result = result.resize((requested_width, requested_height), Image.Resampling.BICUBIC)
        return result

    def process(self, job: Job) -> list[Path]:
        process_started_at = time.monotonic()
        self.output_durations = {}
        try:
            self._active_ultimate_denoise_strength = max(0.05, min(0.95, float(job.custom.get("ultimate_qwen_denoise_strength", 0.20))))
        except (TypeError, ValueError):
            self._active_ultimate_denoise_strength = 0.20
        try:
            self._active_ultimate_tile_coverage = max(0.45, min(0.75, float(job.custom.get("ultimate_qwen_tile_coverage", 0.50))))
        except (TypeError, ValueError):
            self._active_ultimate_tile_coverage = 0.50
        try:
            self._active_ultimate_refinement_passes = max(1, min(2, int(job.custom.get("ultimate_qwen_passes", 1))))
        except (TypeError, ValueError):
            self._active_ultimate_refinement_passes = 1
        try:
            self._active_ultimate_steps = max(4, min(60, int(job.custom.get("ultimate_qwen_steps", 8))))
        except (TypeError, ValueError):
            self._active_ultimate_steps = 8
        workflow_dirs = {
            "image_generate": GENERATED_DIR,
            "image_edit": EDITED_DIR,
            "enhance": OUTPUT_DIR,
            "lora_closeup_1024": CLOSEUPS_DIR,
            "square_1_1": OUTPAINT_DIR,
            "upscale_only": OUTPUT_DIR,
        }
        out_dir = Path(job.output_dir) if job.output_dir else workflow_dirs[job.workflow]
        out_dir.mkdir(parents=True, exist_ok=True)
        # A dedicated super-resolution task must never load or invoke Qwen diffusion.
        if job.workflow == "upscale_only":
            if not job.input_path:
                raise ValueError("Select an image in the queue before upscaling.")
            output = self._run_upscale_file(Path(job.input_path), job.custom, out_dir)
            self.output_durations[str(output)] = time.monotonic() - process_started_at
            return [output]
        steps = int(job.custom.get("num_inference_steps", 8))
        source_file = Image.open(job.input_path) if job.input_path else None
        source_original = ImageOps.exif_transpose(source_file).copy() if source_file else None
        outputs: list[Path] = []
        seed_mode = str(job.seed_mode or "Random")
        # Fixed/Increment with -1 still gets one random base seed. Random generates
        # a fresh explicit seed for every item so metadata records the actual seed.
        base_seed = (None if seed_mode == "Random" else
                     int(job.seed) if int(job.seed) >= 0 else random.randint(0, 2**31 - 1))

        def seed_for(index: int = 0) -> int:
            if seed_mode == "Random":
                return random.randint(0, 2**31 - 1)
            if seed_mode == "Increment":
                return int(base_seed) + index
            return int(base_seed)

        # Ultimate and its Lanczos scaling run strictly when Ultimate is enabled by the user.
        if job.workflow in {"enhance", "lora_closeup_1024", "square_1_1"}:
            if job.prompt and self._needs_english_translation(job.prompt):
                job.prompt = self._translate_image_prompt_to_english(job.prompt, job.workflow)
            if job.workflow == "lora_closeup_1024" and job.target and self._needs_english_translation(job.target):
                job.target = self._translate_image_prompt_to_english(job.target, job.workflow)

        try:
            if job.workflow == "image_generate":
                profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                translated_prompt = self._translate_image_prompt_to_english(job.prompt, job.workflow)
                effective_prompt = apply_qwen_detail_profile_prompt(prepare_native_prompt(translated_prompt, self.log), profile_key)
                profile_strength, profile_tile_strength = qwen_detail_profile_values(profile_key)
                ultimate_enabled = bool(job.custom.get("ultimate_qwen_upscale_enabled", False))
                ult_scale = max(1, min(4, int(job.custom.get("ultimate_qwen_scale", 2) or 2))) if ultimate_enabled else 1
                gen_stage1_size = self._ultimate_stage1_size((job.width, job.height), ultimate_enabled, ult_scale)
                self.log(f"Image Generate · Qwen detail profile={profile_key} · Ultimate={'ON' if ultimate_enabled else 'OFF'} · prompt profile and detail-pass strength={profile_tile_strength:.2f} · stage1={gen_stage1_size[0]}×{gen_stage1_size[1]} · final={job.width}×{job.height}.")
                for index in range(max(1, min(100, job.batch_count))):
                    if self.force_stop.is_set():
                        raise Aborted()
                    image_started_at = time.monotonic()
                    current_seed = seed_for(index)
                    self.log(f"Image {index + 1}: seed={current_seed} ({seed_mode})")
                    result = self._infer_with_auto_offload_retry(effective_prompt, None, gen_stage1_size, current_seed, job.quality, steps)
                    if max(job.width, job.height) > MAX_INFERENCE_SIDE or profile_key in {"balanced", "strong", "maximum"}:
                        if ultimate_enabled:
                            target_sz = (int(job.width), int(job.height))
                            if target_sz != result.size:
                                self.log(
                                    f"Image Generate Ultimate two-stage upscale · Pass 1 Qwen at {gen_stage1_size[0]}×{gen_stage1_size[1]} → "
                                    f"Lanczos {ult_scale}× upscale to {target_sz[0]}×{target_sz[1]} → 4-tile Ultimate refinement."
                                )
                                result = result.resize(target_sz, Image.Resampling.LANCZOS)
                                self._publish_live_preview(result, marker=("generate-stage-1-then-scale", gen_stage1_size[0], gen_stage1_size[1], target_sz[0], target_sz[1]))
                        result = self._qwen_detail_refine_large_output(
                            result, effective_prompt, current_seed + 1000, job.quality, steps,
                            detail_mode="ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast", use_img2img=True, tile_strength=profile_tile_strength,
                            detail_profile=profile_key, force=ultimate_enabled or max(job.width, job.height) <= MAX_INFERENCE_SIDE,
                        )
                    path = out_dir / f"generate_{time.strftime('%Y%m%d_%H%M%S')}_{index+1:03d}_{uuid.uuid4().hex[:5]}.png"
                    save_image_with_metadata(result, path, metadata={
                        "prompt": effective_prompt,
                        "original_prompt": job.prompt,
                        "translated_prompt": translated_prompt if translated_prompt != job.prompt else "",
                        "workflow": job.workflow,
                        "seed": current_seed,
                        "seed_mode": seed_mode,
                        **qwen_detail_refinement_metadata(profile_key, "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast"),
                        "qwen_first_pass_mode": "text-to-image (no init image)",
                    })
                    outputs.append(path)
                    self.output_durations[str(path)] = time.monotonic() - image_started_at
                    self.log(f"Saved {path}")
                    if self.stop_after_current.is_set():
                        self.log("Stop current job requested; finishing the active image and ending the generation batch.")
                        break
            elif job.workflow == "image_edit":
                if source_original is None:
                    raise ValueError("Image Edit requires an input image. Add an image to the queue and select it.")
                edit_seed = seed_for(0)
                profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                translated_prompt = self._translate_image_prompt_to_english(job.prompt, job.workflow)
                effective_prompt = apply_qwen_detail_profile_prompt(prepare_native_prompt(translated_prompt, self.log), profile_key)
                watermark_enabled = bool(job.custom.get("watermark_removal_enabled", False))
                if watermark_enabled:
                    effective_prompt = apply_watermark_removal_prompt(effective_prompt, True)
                    self.log("Watermark removal ON · Qwen will target visible watermark overlays and preserve the rest of the image.")
                self.log(f"Edit seed={edit_seed} ({seed_mode}) · Qwen detail profile={profile_key}")
                profile_strength, profile_tile_strength = qwen_detail_profile_values(profile_key)
                ultimate_enabled = bool(job.custom.get("ultimate_qwen_upscale_enabled", False))
                ult_scale = max(1, min(4, int(job.custom.get("ultimate_qwen_scale", 2) or 2))) if ultimate_enabled else 1
                edit_stage1_size = self._ultimate_stage1_size((job.width, job.height), ultimate_enabled, ult_scale)
                result = self._infer_with_auto_offload_retry(
                    effective_prompt, source_original, edit_stage1_size, edit_seed, job.quality, steps,
                    init_image=source_original, strength=profile_strength,
                )
                if max(job.width, job.height) > MAX_INFERENCE_SIDE or profile_key in QWEN_DETAIL_PROFILES:
                    if ultimate_enabled:
                        target_sz = (int(job.width), int(job.height))
                        if target_sz != result.size:
                            self.log(
                                f"Image Edit Ultimate two-stage upscale · Pass 1 Qwen at {edit_stage1_size[0]}×{edit_stage1_size[1]} → "
                                f"Lanczos {ult_scale}× upscale to {target_sz[0]}×{target_sz[1]} → 4-tile Ultimate refinement."
                            )
                            result = result.resize(target_sz, Image.Resampling.LANCZOS)
                            self._publish_live_preview(result, marker=("edit-stage-1-then-scale", edit_stage1_size[0], edit_stage1_size[1], target_sz[0], target_sz[1]))
                    result = self._qwen_detail_refine_large_output(
                        result, effective_prompt, edit_seed + 1000, job.quality, steps,
                        detail_mode="ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast", use_img2img=True, tile_strength=profile_tile_strength,
                        detail_profile=profile_key, force=ultimate_enabled or max(job.width, job.height) <= MAX_INFERENCE_SIDE,
                    )
                path = out_dir / f"edit_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{job.output_format}"
                save_image_with_metadata(result, path, source_original, {
                    "prompt": effective_prompt, "original_prompt": job.prompt,
                    "translated_prompt": translated_prompt if translated_prompt != job.prompt else "",
                    "workflow": job.workflow, "seed": edit_seed, "seed_mode": seed_mode, "img2img_strength": profile_strength,
                    **qwen_detail_refinement_metadata(profile_key, "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast"),
                })
                outputs.append(path)
                self.output_durations[str(path)] = time.monotonic() - process_started_at
            elif job.workflow == "enhance":
                if source_original is None:
                    raise ValueError("Enhance requires an input image.")
                profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                prompt = apply_identity_lock(
                    apply_qwen_detail_profile_prompt(build_enhance_prompt(job.prompt), profile_key),
                    bool(job.custom.get("identity_lock", True))
                )
                watermark_enabled = bool(job.custom.get("watermark_removal_enabled", False))
                if watermark_enabled:
                    prompt = apply_watermark_removal_prompt(prompt, True)
                    self.log("Watermark removal ON · applied to the Enhance reconstruction/detail prompts.")
                self.log("Enhance strategy: texture-first restoration; preserve material-specific detail and avoid smoothing/softening.")
                self.log(
                    f"Enhance settings · source={source_original.width}×{source_original.height} · "
                    f"output={job.width}×{job.height} · identity_lock={bool(job.custom.get('identity_lock', True))} · "
                    f"steps={steps} · attempts={int(job.custom.get('max_attempts', 1))} · "
                    f"detail_mode={job.custom.get('enhance_detail_mode', 'balanced')}."
                )
                effective_seed = seed_for(0)
                profile_strength, profile_tile_strength = qwen_enhance_reconstruction_strengths(profile_key)
                ultimate_enabled = bool(job.custom.get("ultimate_qwen_upscale_enabled", False))
                detail_mode = enhance_detail_mode_for_ultimate(ultimate_enabled)
                if ultimate_enabled:
                    profile_strength, _ = qwen_detail_profile_values(profile_key)
                    configured_tile_strength = max(0.05, min(0.95, float(job.custom.get("ultimate_qwen_denoise_strength", 0.20))))
                    effective_tile_strength = configured_tile_strength
                    ult_scale = max(1, min(4, int(job.custom.get("ultimate_qwen_scale", 2) or 2)))
                    s1_w, s1_h = self._ultimate_stage1_size((job.width, job.height), True, ult_scale)
                    self.log(
                        f"Qwen Enhance reconstruction · profile={profile_key} · Ultimate=ON · Pass 1={s1_w}×{s1_h} → "
                        f"Lanczos Ultimate scale={ult_scale}× → final={job.width}×{job.height} · "
                        f"denoise={effective_tile_strength:.2f} · coverage={self._active_ultimate_tile_coverage:.0%} · "
                        f"tile steps={int(job.custom.get('ultimate_qwen_steps', 8))} · overlapping tiles."
                    )
                else:
                    self.log(
                        f"Qwen Enhance reconstruction · profile={profile_key} · one full-frame strength={profile_strength:.2f} · "
                        f"no Ultimate tiles · detail strength setting={profile_tile_strength:.2f} · detail mode={detail_mode}."
                    )
                result = self._enhance_qwen_output(source_original, prompt, job.width, job.height, effective_seed, job.quality,
                    int(job.custom.get("max_attempts", 1)), float(job.custom.get("retry_diff", 25.0)),
                    float(job.custom.get("warn_diff", 18.0)), steps,
                    detail_mode, strength=profile_strength,
                    tile_strength=profile_tile_strength, detail_profile=profile_key,
                    ultimate_scale=int(job.custom.get("ultimate_qwen_scale", 2) or 2) if ultimate_enabled else 1)
                path = out_dir / f"enhance_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                detail_metadata = qwen_detail_refinement_metadata(profile_key, detail_mode)
                if detail_mode == "ultimate":
                    detail_metadata["qwen_tile_strength"] = max(0.05, min(0.95, float(getattr(self, "_active_ultimate_denoise_strength", 0.20))))
                save_image_with_metadata(result, path, source_original, {
                    "prompt": prompt, "workflow": job.workflow, "seed": effective_seed, "seed_mode": seed_mode,
                    "img2img_strength": None if detail_mode == "ultimate" else profile_strength,
                    **detail_metadata,
                })
                outputs.append(path)
                self.output_durations[str(path)] = time.monotonic() - process_started_at
            elif job.workflow == "lora_closeup_1024":
                if source_original is None:
                    raise ValueError("Close-up requires an input image.")
                requested_side = min(MAX_OUTPUT_SIDE, max(1024, job.side))
                ultimate_enabled = bool(job.custom.get("ultimate_qwen_upscale_enabled", False))
                closeup_ult_scale = max(1, min(4, int(job.custom.get("ultimate_qwen_scale", 2) or 2))) if ultimate_enabled else 1
                final_side = requested_side
                side, _ = self._ultimate_stage1_size((final_side, final_side), ultimate_enabled, closeup_ult_scale)
                closeup_seed = seed_for(0)
                effective_prompt = ""
                crop_box = None
                effective_zoom = None
                # A trigger token belongs in the model prompt only when a real local LoRA
                # adapter is enabled; it is still useful as a caption prefix independently.
                active_trigger = job.trigger_word if self.lora_enabled and self.lora_path else ""
                if job.trigger_word.strip() and not active_trigger:
                    self.log("Close-up trigger word will be saved to the optional caption only; no enabled local LoRA is configured.")
                detected_bbox = None
                if job.focus_point is not None:
                    # Right-clicking an item in the preview is the most precise offline
                    # choice for arbitrary objects; it wins over all automatic heuristics.
                    cropped, crop_box, effective_zoom = crop_closeup_square(
                        source_original, side, job.zoom_factor, job.focus_point
                    )
                    focus_description = "user-selected preview point (takes priority)"
                    self.log("Close-up focus: using exact user-selected preview point; semantic model inference is skipped for target localization.")
                elif job.auto_zoom:
                    # Reuse the exact text encoder and vision projector selected in Model Catalog on the GPU.
                    # Unload sd-server first if active so the 5.5 GB grounding model has full GPU VRAM, then release it immediately after.
                    model_parts = self._find_model_parts(Path(self.model_path).expanduser().resolve())
                    llm_candidate = model_parts.get("llm")
                    vision_candidate = model_parts.get("llm_vision")
                    resolved_gguf_llm, resolved_gguf_mmproj = resolve_closeup_grounding_gguf_pair(
                        llm_candidate, vision_candidate
                    )
                    uses_gguf_grounder = (
                        resolved_gguf_llm is not None
                        and resolved_gguf_mmproj is not None
                        and resolved_gguf_llm.is_file()
                        and resolved_gguf_mmproj.is_file()
                    )
                    # The grounding server requests CUDA layers. Release the image diffusion
                    # process first to avoid both large models competing for the same VRAM.
                    if self.server_process is not None and uses_gguf_grounder:
                        self.log("Close-up target grounding: stopping the image diffusion server first to release GPU memory; it will reload on the next image inference.")
                        self.unload_model()
                    self.status_callback("Close-up Stage 1 · Qwen3-VL GPU target grounding · locating target on NVIDIA CUDA…")
                    try:
                        grounding_result = self._locate_closeup_target(
                            source_original, job.target, model_parts.get("llm"), model_parts.get("llm_vision"),
                            profile=str(job.custom.get("grounding_profile", "balanced")),
                        )
                    except RuntimeError as grounding_error:
                        self.log("Close-up semantic targeting could not start; see diagnostic details.")
                        raise RuntimeError(
                            "Close-up Auto Zoom could not start semantic target localization. No center crop was substituted. "
                            "Select a compatible chat-capable GGUF encoder and matching vision projector in Model Catalog, "
                            "or right-click the intended target in the preview to choose a manual focus point, or turn Auto Zoom off. "
                            f"Details: {grounding_error}"
                        ) from grounding_error
                    if (grounding_result.status != "found" or grounding_result.bbox_xyxy is None) and not uses_gguf_grounder:
                        non_gguf_res = locate_target_bbox_non_gguf(source_original, job.target, log_callback=self.log)
                        if non_gguf_res.status == "found" and non_gguf_res.bbox_xyxy is not None:
                            grounding_result = GroundingResult(
                                status="found", bbox_xyxy=non_gguf_res.bbox_xyxy, target_text=job.target,
                                attempts=max(1, grounding_result.attempts), reason_code="ok",
                                diagnostic_summary="Target located via local visual target grounding.",
                                elapsed_ms=grounding_result.elapsed_ms, candidates=(non_gguf_res.bbox_xyxy,),
                            )
                    if grounding_result.status == "ambiguous":
                        review_callback = self.grounding_review_callback
                        if review_callback is not None:
                            from .ui.grounding_review_dialog import GroundingReviewDecision
                            review_payload = {
                                "image": source_original.copy(),
                                "result": grounding_result,
                                "target": job.target,
                                "job_id": job.job_id,
                                "allow_retry": True,
                            }
                            decision = review_callback(review_payload)
                            if not isinstance(decision, GroundingReviewDecision):
                                raise RuntimeError("Close-up target review returned an invalid decision.")
                            if decision.action == "cancel":
                                raise Aborted("Close-up target review was cancelled.")
                            if decision.action == "accept":
                                grounding_result = GroundingResult(
                                    status="found", bbox_xyxy=decision.bbox_xyxy, target_text=job.target,
                                    attempts=grounding_result.attempts, reason_code="user_confirmed_box",
                                    diagnostic_summary="Target box explicitly confirmed or adjusted by the user.",
                                    elapsed_ms=grounding_result.elapsed_ms, candidates=grounding_result.candidates,
                                )
                            elif decision.action == "retry":
                                retry_target = (
                                    f"{job.target}. Locate one clearly visible instance of this target only. "
                                    "Return one tight bounding box around the target, or state that it was not found."
                                )
                                grounding_result = self._locate_closeup_target(
                                    source_original, retry_target, model_parts.get("llm"), model_parts.get("llm_vision"),
                                    profile=str(job.custom.get("grounding_profile", "balanced")),
                                )
                                if grounding_result.status == "ambiguous":
                                    decision = review_callback({
                                        "image": source_original.copy(), "result": grounding_result,
                                        "target": job.target, "job_id": job.job_id, "allow_retry": False,
                                    })
                                    if not isinstance(decision, GroundingReviewDecision):
                                        raise RuntimeError("Close-up target review returned an invalid decision.")
                                    if decision.action == "cancel":
                                        raise Aborted("Close-up target review was cancelled.")
                                    if decision.action == "accept":
                                        grounding_result = GroundingResult(
                                            status="found", bbox_xyxy=decision.bbox_xyxy, target_text=job.target,
                                            attempts=grounding_result.attempts, reason_code="user_confirmed_box",
                                            diagnostic_summary="Target box explicitly confirmed or adjusted by the user.",
                                            elapsed_ms=grounding_result.elapsed_ms, candidates=grounding_result.candidates,
                                        )
                    if grounding_result.status != "found" or grounding_result.bbox_xyxy is None:
                        self.log(f"Close-up semantic targeting did not produce an accepted box · status={grounding_result.status} · reason={grounding_result.reason_code}.")
                        if grounding_result.status == "ambiguous":
                            message = "Several distinct target candidates were found; review was skipped or no candidate was accepted."
                        elif grounding_result.status == "runtime_error":
                            message = "The local grounding runtime failed. Check Model Catalog components and Diagnostics."
                        elif grounding_result.status == "timeout":
                            message = "Target localization exceeded its time budget. Try Fast profile or manual focus."
                        elif grounding_result.status == "cancelled":
                            raise Aborted("Close-up target localization was cancelled.")
                        elif grounding_result.status == "not_found":
                            message = "The model could not locate the requested visible target. Try a clearer description or manual focus."
                        else:
                            message = "The model response did not contain a reliable bounding box. Try a clearer description or manual focus."
                        raise RuntimeError(
                            "Close-up Auto Zoom could not find a reliable bounding box. No center crop or keyword guess was substituted. "
                            + message + f" Diagnostic reason: {grounding_result.reason_code}."
                        )
                    detected_bbox = grounding_result.bbox_xyxy
                    if detected_bbox is not None:
                        cropped, crop_box, effective_zoom = crop_closeup_from_bbox(
                            source_original, side, detected_bbox, padding=1.55, zoom_factor=job.zoom_factor
                        )
                        matched_description = str(grounding_result.matched_description or "").strip()
                        semantic_match_note = (
                            f" · model-generated related description={matched_description[:180]!r}"
                            if matched_description and matched_description.casefold() != str(job.target or "").strip().casefold()
                            else ""
                        )
                        self.log(
                            f"Close-up Qwen3-VL proposal: target={job.target!r}{semantic_match_note} · "
                            f"bbox_xyxy={tuple(round(v, 4) for v in detected_bbox)} · crop={crop_box} · "
                            f"requested zoom={job.zoom_factor:.2f}× · effective zoom={effective_zoom:.2f}×."
                        )
                        focus_description = "Qwen3-VL open-vocabulary semantic grounding"
                else:
                    cropped, crop_box, effective_zoom = crop_closeup_square(
                        source_original, side, job.zoom_factor, None
                    )
                    focus_description = "manual center (no focus point selected)"
                self._publish_live_preview(cropped, marker=("closeup-crop-ready", side, final_side))
                self.log(
                    f"Close-up mode: {'target-guided crop + Qwen enhance' if job.auto_zoom and job.enhance else 'manual crop + Qwen enhance' if job.enhance else 'crop-only (no AI inference)'} · "
                    f"zoom={effective_zoom:.2f}× · crop={crop_box} · target={job.target!r} · "
                    f"focus={focus_description} · stage1={side}×{side} · final_output={final_side}×{final_side}."
                )
                if job.enhance:
                    # One combined prompt does both target-aware close-up enhancement and
                    # source-faithful cleanup; the crop is not re-generated from a padded frame.
                    profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                    effective_prompt = apply_identity_lock(
                        apply_qwen_detail_profile_prompt(
                            build_closeup_prompt(job.target, job.prompt, True, active_trigger), profile_key),
                        bool(job.custom.get("identity_lock", True)),
                    )
                    if bool(job.custom.get("watermark_removal_enabled", False)):
                        effective_prompt = apply_watermark_removal_prompt(effective_prompt, True)
                        self.log("Watermark removal ON · included in Close-up reconstruction.")
                    # The geometric target crop has already been prepared at stage-1 square size (1024×1024 when Ultimate is active).
                    # Enhance this stable output canvas instead of feeding the raw source crop
                    # back through fitting/resizing, which could change the effective framing.
                    profile_strength, profile_tile_strength = qwen_detail_profile_values(job.custom.get("qwen_detail_profile", "balanced"))
                    self.log(
                        f"Close-up output Enhance · prepared {side}×{side} target crop as the init/output canvas · "
                        f"Qwen profile={job.custom.get('qwen_detail_profile', 'balanced')} · strength={profile_strength:.2f}."
                    )
                    result = self._enhance_qwen_output(
                        cropped, effective_prompt, side, side, closeup_seed, job.quality,
                        int(job.custom.get("max_attempts", 1)), float(job.custom.get("retry_diff", 25.0)),
                        float(job.custom.get("warn_diff", 18.0)), steps, detail_mode="fast",
                        use_img2img=True, strength=profile_strength, tile_strength=profile_tile_strength,
                        detail_profile=str(job.custom.get("qwen_detail_profile", "balanced")),
                    )
                else:
                    result = cropped
                    if bool(job.custom.get("watermark_removal_enabled", False)):
                        removal_prompt = apply_watermark_removal_prompt(
                            "Remove only watermark overlays from this finished Close-up crop. Preserve the crop, identity, geometry, colors, and lighting.",
                            True,
                        )
                        removal_strength = qwen_detail_profile_values(job.custom.get("qwen_detail_profile", "balanced"))[0]
                        self.log("Watermark removal ON · Close-up Enhance is OFF, so running one explicit removal pass on the crop.")
                        result = self._infer_with_auto_offload_retry(
                            removal_prompt, cropped, (side, side), closeup_seed + 1500, job.quality, steps,
                            init_image=cropped, strength=max(0.72, removal_strength),
                        )
                if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)):
                    profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                    _profile_strength, profile_tile_strength = qwen_detail_profile_values(profile_key)
                    if result.size != (final_side, final_side):
                        self.log(
                            f"Close-up Ultimate two-stage upscale · Pass 1 Qwen at {side}×{side} → "
                            f"Lanczos {closeup_ult_scale}× upscale to {final_side}×{final_side} → 4-tile Ultimate refinement & seamless stitch."
                        )
                        result = result.resize((final_side, final_side), Image.Resampling.LANCZOS)
                        self._publish_live_preview(result, marker=("closeup-stage-1-then-scale", side, final_side))
                    self.log(
                        f"Close-up · running Model Catalog Ultimate Qwen refinement "
                        f"(denoise={self._active_ultimate_denoise_strength:.2f}, coverage={self._active_ultimate_tile_coverage:.0%}, "
                        f"steps={self._active_ultimate_steps}, passes={self._active_ultimate_refinement_passes})."
                    )
                    result = self._qwen_detail_refine_large_output(
                        result, effective_prompt or apply_qwen_detail_profile_prompt(build_enhance_prompt(job.prompt), profile_key),
                        closeup_seed + 1000, job.quality, steps, detail_mode="ultimate",
                        tile_strength=self._active_ultimate_denoise_strength, detail_profile=profile_key,
                        force=True,
                    )
                path = out_dir / f"closeup_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                closeup_metadata = {
                    "prompt": effective_prompt or job.prompt, "target": job.target, "workflow": job.workflow,
                    "seed": closeup_seed if job.enhance else None, "seed_mode": seed_mode,
                    "auto_zoom": job.auto_zoom, "enhance": job.enhance, "zoom_factor": effective_zoom,
                    "img2img_strength": qwen_detail_profile_values(job.custom.get("qwen_detail_profile", "balanced"))[0] if job.enhance else None,
                    **qwen_detail_refinement_metadata(job.custom.get("qwen_detail_profile", "balanced"), "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast", enabled=bool(job.enhance) and bool(job.custom.get("ultimate_qwen_upscale_enabled", False))),
                    "crop_box": crop_box, "focus_point": job.focus_point, "detected_bbox_xyxy": detected_bbox, "focus_method": focus_description, "trigger_word": job.trigger_word,
                }
                save_image_with_metadata(result, path, source_original, closeup_metadata)
                outputs.append(path)
                self.output_durations[str(path)] = time.monotonic() - process_started_at
                if job.save_caption:
                    existing = Path(job.input_path).with_suffix(".txt")
                    old_caption = existing.read_text(encoding="utf-8").strip() if existing.exists() else "high quality, detailed photograph"
                    caption = ", ".join(x for x in [job.trigger_word.strip(), f"close-up of {job.target.strip()}", old_caption] if x)
                    path.with_suffix(".txt").write_text(caption, encoding="utf-8")
            elif job.workflow == "square_1_1":
                if source_original is None:
                    raise ValueError("Square 1:1 requires an input image.")
                requested_side = min(MAX_OUTPUT_SIDE, max(1024, job.side))
                ultimate_enabled = bool(job.custom.get("ultimate_qwen_upscale_enabled", False))
                square_ult_scale = max(1, min(4, int(job.custom.get("ultimate_qwen_scale", 2) or 2))) if ultimate_enabled else 1
                final_side = requested_side
                side, _ = self._ultimate_stage1_size((final_side, final_side), ultimate_enabled, square_ult_scale)
                mode, fitted, margins = plan_square_outpaint(source_original, side)
                square_seed = seed_for(0)
                self.log(
                    f"Square 1:1 mode={mode} · enhance={'ON' if job.enhance else 'OFF'} · "
                    f"source={source_original.width}×{source_original.height} · target={side}×{side}."
                )
                if mode in ("copy", "crop") and not job.enhance:
                    self.log("Square 1:1: source is already square/nearly square and base Enhance is OFF; resize/crop only, with the optional Ultimate pass applied afterward to the finished output if enabled.")
                if mode in ("copy", "crop"):
                    if job.enhance:
                        profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                        square_prompt = apply_identity_lock(
                            apply_qwen_detail_profile_prompt(build_square_core_enhance_prompt(job.prompt), profile_key),
                            bool(job.custom.get("identity_lock", True))
                        )
                        if bool(job.custom.get("watermark_removal_enabled", False)):
                            square_prompt = apply_watermark_removal_prompt(square_prompt, True)
                        profile_strength, profile_tile_strength = qwen_detail_profile_values(profile_key)
                        result = self._enhance_qwen_output(
                            fitted, square_prompt, side, side, square_seed, job.quality,
                            int(job.custom.get("max_attempts", 1)), float(job.custom.get("retry_diff", 25.0)),
                            float(job.custom.get("warn_diff", 18.0)), steps, detail_mode="fast",
                            use_img2img=True, strength=profile_strength,
                            tile_strength=profile_tile_strength, detail_profile=profile_key
                        )
                    else:
                        result = fitted
                else:
                    left, top, right, bottom = margins
                    profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                    square_base_prompt = build_square_prompt(job.prompt, job.enhance)
                    if job.enhance:
                        square_base_prompt = apply_qwen_detail_profile_prompt(square_base_prompt, profile_key)
                    prompt = apply_identity_lock(
                        square_base_prompt, bool(job.custom.get("identity_lock", True))
                    )
                    if bool(job.custom.get("watermark_removal_enabled", False)):
                        prompt = apply_watermark_removal_prompt(prompt, True)
                    prompt += (
                        f" The final output canvas is {side} × {side} pixels. The reference input keeps its original "
                        "aspect ratio; expand beyond its original edges to fill the square output. Do not add side bars, "
                        "solid-color panels, blurred borders, or letterboxing."
                    )
                    edge_names = []
                    if left > 0: edge_names.append("left")
                    if right < side: edge_names.append("right")
                    if top > 0: edge_names.append("top")
                    if bottom < side: edge_names.append("bottom")
                    if not edge_names:
                        raise RuntimeError("Square outpaint mode was selected but no missing edge was found.")
                    self.log(
                        "Square 1:1 edge plan (offline): " + ", ".join(edge_names) +
                        ". No separate vision/chat model is used; the native Qwen image engine receives the original reference and a combined mask for all missing margins."
                    )
                    feather = max(8, min(32, side // 48))
                    working_canvas, _full_mask = build_square_outpaint_inputs(
                        source_original, side, margins, feather=feather
                    )
                    # A single masked Qwen inference extends every missing side together. This
                    # avoids running one separate generation for each edge and keeps all margins
                    # contextually consistent. The original source core is restored afterward.
                    edge_summary = ", ".join(edge_names)
                    joint_prompt = (
                        prompt + "\n\nJOINT OUTPAINT PASS: extend all masked missing margins in this ONE generation pass. "
                        f"The editable margins are: {edge_summary}. Continue the same original scene seamlessly "
                        "beyond each corresponding source boundary. Treat opposite margins as one coherent scene, "
                        "matching perspective, geometry, materials, texture, colors, lighting, horizon and shadows. "
                        "Only the masked missing margins may be synthesized; preserve the source interior and all "
                        "subjects exactly. Do not add bars, borders, panels, duplicate subjects, unrelated scenery, "
                        "or a second image. The source photograph is authoritative for identity and scene content."
                    )
                    self.log(
                        f"Square 1:1 joint outpaint · one Qwen generation pass for all missing margins ({edge_summary}); "
                        "one combined mask; the source core is composited back after inference."
                    )
                    if self.force_stop.is_set():
                        raise Aborted()
                    generated_pass = self._infer_with_auto_offload_retry(
                        joint_prompt, source_original, (side, side), square_seed,
                        job.quality, steps, fit_source_to_output=False, init_image=working_canvas,
                        mask_image=_full_mask, strength=0.94,
                    )
                    generated = composite_square_outpaint_result(
                        source_original, generated_pass, side, margins, enhance=False, feather=feather
                    )
                    # Outpaint has completed and the original core is composited. Enhance
                    # this complete square output as one image instead of refining only a
                    # source crop and pasting that crop back into the finished canvas.
                    feather = max(16, min(64, side // 24))
                    result = composite_square_outpaint_result(
                        source_original, generated, side, margins, enhance=False, feather=feather
                    )
                    if job.enhance:
                        profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                        profile_strength, profile_tile_strength = qwen_detail_profile_values(profile_key)
                        square_output_prompt = (
                            build_enhance_prompt("") + " The square 1:1 output and all outpainted margins are already complete. "
                            "Refine this finished square as one image. Do not extend the canvas, change the crop, add new margins, "
                            "or repeat the outpainting task. Keep existing synthesized surroundings coherent with the original scene. "
                            + ("Scene context for material consistency only: " + str(job.prompt).strip() if str(job.prompt or "").strip() else "")
                        )
                        square_output_prompt = apply_identity_lock(
                            apply_qwen_detail_profile_prompt(square_output_prompt, profile_key),
                            bool(job.custom.get("identity_lock", True))
                        )
                        if bool(job.custom.get("watermark_removal_enabled", False)):
                            square_output_prompt = apply_watermark_removal_prompt(square_output_prompt, True)
                        self.log(
                            f"Square 1:1 output Enhance · applying Qwen profile={profile_key} to the completed {side}×{side} square "
                            f"(original core + generated margins) at strength={profile_strength:.2f}."
                        )
                        result = self._enhance_qwen_output(
                            result, square_output_prompt, side, side, square_seed + 1000, job.quality,
                            int(job.custom.get("max_attempts", 1)), float(job.custom.get("retry_diff", 25.0)),
                            float(job.custom.get("warn_diff", 18.0)), steps, detail_mode="fast",
                            use_img2img=True, strength=profile_strength, tile_strength=profile_tile_strength,
                            detail_profile=profile_key,
                        )
                        # Protect joins with a narrow seam blend while retaining enhanced
                        # content across the completed square rather than reverting the core.
                        result = composite_square_outpaint_result(
                            source_original, result, side, margins, enhance=True, feather=feather
                        )
                    elif side > MAX_INFERENCE_SIDE:
                        self.log("Square 1:1 base Enhance is OFF · completed outpaint is preserved; the optional Ultimate output pass still follows its ON/OFF toggle.")
                if bool(job.custom.get("watermark_removal_enabled", False)) and not job.enhance:
                    profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                    removal_prompt = apply_watermark_removal_prompt(
                        "Remove only visible watermark overlays from this completed Square 1:1 image. Preserve all original scene content, generated margins, subjects, text, geometry, colors, lighting, and framing.",
                        True,
                    )
                    removal_prompt = apply_qwen_detail_profile_prompt(removal_prompt, profile_key)
                    removal_strength = max(0.72, qwen_detail_profile_values(profile_key)[0])
                    self.log("Watermark removal ON · Square Enhance is OFF, so running one explicit removal pass on the completed square.")
                    result = self._infer_with_auto_offload_retry(
                        removal_prompt, result, (side, side), square_seed + 1500, job.quality, steps,
                        init_image=result, strength=removal_strength,
                    )
                if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)):
                    if self.force_stop.is_set():
                        raise Aborted()
                    profile_key = str(job.custom.get("qwen_detail_profile", "balanced"))
                    _profile_strength, profile_tile_strength = qwen_detail_profile_values(profile_key)
                    if result.size != (final_side, final_side):
                        self.log(f"Square 1:1 Ultimate two-stage upscale · Stage-1 Qwen output {side}×{side} → Lanczos {square_ult_scale}× upscale to {final_side}×{final_side} → 4-tile refinement.")
                        result = result.resize((final_side, final_side), Image.Resampling.LANCZOS)
                        if mode == "outpaint" and source_original is not None:
                            _, _, final_margins = plan_square_outpaint(source_original, final_side)
                            if final_margins is not None:
                                result = composite_square_outpaint_result(
                                    source_original, result, final_side, final_margins,
                                    enhance=bool(job.enhance), feather=max(16, min(64, final_side // 24)),
                                )
                        self._publish_live_preview(result, marker=("square-stage-1-then-scale", side, final_side))
                    self.log("Square 1:1 · Ultimate Qwen refinement is running on the completed final square output, after joint outpainting and source-core compositing.")
                    self.status_callback("Ultimate Qwen output refinement · final square image · overlapping tile pass…")
                    result = self._qwen_detail_refine_large_output(
                        result,
                        "Completed square 1:1 image; preserve the finished framing, synthesized margins and original source core. " + str(job.prompt or ""),
                        square_seed + 2000, job.quality, steps,
                        detail_mode="ultimate", use_img2img=True,
                        tile_strength=self._active_ultimate_denoise_strength, detail_profile=profile_key, force=True,
                    )
                path = out_dir / f"square_{side}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                save_image_with_metadata(result, path, source_original, {
                    "workflow": job.workflow, "prompt": job.prompt,
                    "seed": square_seed if (mode not in ("copy", "crop") or job.enhance or bool(job.custom.get("ultimate_qwen_upscale_enabled", False))) else None,
                    "seed_mode": seed_mode,
                    "img2img_strength": qwen_detail_profile_values(job.custom.get("qwen_detail_profile", "balanced"))[0] if job.enhance else None,
                    **qwen_detail_refinement_metadata(job.custom.get("qwen_detail_profile", "balanced"), "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast", enabled=bool(job.custom.get("ultimate_qwen_upscale_enabled", False))),
                })
                outputs.append(path)
                self.output_durations[str(path)] = time.monotonic() - process_started_at
            else:
                raise ValueError(f"Unknown workflow: {job.workflow}")
        finally:
            if source_file is not None:
                source_file.close()
        # Optional user-controlled Real-ESRGAN/Real-ESRNet post-process, independent from Qwen.
        # It runs only when the small global toolbar toggle was ON when the job was queued.
        if bool(job.custom.get("auto_realesrgan_enabled", False)) and outputs and job.workflow != "upscale_only":
            original_outputs = list(outputs)
            self.log("Global ESRGAN toggle is ON · running the selected standalone Vulkan upscaler on workflow outputs.")
            for upscale_index, produced in enumerate(original_outputs, start=1):
                if self.force_stop.is_set():
                    raise Aborted()
                self.status_callback(f"ESRGAN · image {upscale_index}/{len(original_outputs)} · starting Vulkan upscaler…")
                upscaled = self._run_upscale_file(Path(produced), job.custom, out_dir)
                outputs.append(upscaled)
                self.output_durations[str(upscaled)] = time.monotonic() - process_started_at
                self.status_callback(f"ESRGAN · image {upscale_index}/{len(original_outputs)} complete · output saved")
        for output in outputs:
            self.output_durations.setdefault(str(output), time.monotonic() - process_started_at)
        self.status_callback("Finalizing output · validating saved files and updating the queue…")
        return outputs

