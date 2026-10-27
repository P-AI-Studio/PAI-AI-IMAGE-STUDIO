# -*- coding: utf-8 -*-
"""
Centralized input validation, file verification, archive security, and settings schema
sanitization for P.AI AI IMAGE STUDIO.
"""
from __future__ import annotations

import math
import re
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps

from .performance import normalize_grounding_profile

from .constants import (
    DEFAULTS,
    FIXED_SQUARE_OUTPUT_SIZES,
    IMAGE_EXTENSIONS,
    MAX_OUTPUT_SIDE,
    QUALITY_PRESETS,
    QWEN_DETAIL_PROFILES,
    REALESRGAN_CATALOG_BY_ID,
    SETTINGS_VERSION,
    VALID_MEMORY_MODES,
    VALID_OUTPUT_FORMATS,
    VALID_PREVIEW_METHODS,
    VALID_SEED_MODES,
    VALID_WORKFLOWS,
)

MAX_PROMPT_LENGTH = 4000
MAX_TARGET_LENGTH = 300
MAX_TRIGGER_LENGTH = 120
MIN_IMAGE_SIDE = 16
MAX_IMAGE_SIDE = 16384
MAX_IMAGE_PIXELS = 100_000_000
MAX_IMAGE_FILE_BYTES = 256 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
ALLOWED_LORA_EXTENSIONS = {".safetensors", ".sft", ".gguf"}
ALLOWED_MODEL_EXTENSIONS = {".gguf", ".safetensors", ".sft", ".bin", ".param"}

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ValidationError(ValueError):
    """Raised when user, file, network, or job inputs fail validation."""


def preview_click_to_normalized(
    x: float, y: float, image_rect: tuple[float, float, float, float]
) -> Optional[tuple[float, float]]:
    """Map a widget click to normalized image coordinates, rejecting pane/letterbox clicks.

    ``image_rect`` is (left, top, right, bottom) in widget coordinates with
    exclusive right/bottom edges, matching Qt's image target rectangle.
    The returned values are in [0, 1); they are never clamped from clicks
    outside the actual displayed image, which could silently select a corner.
    """
    try:
        px, py = float(x), float(y)
        left, top, right, bottom = (float(value) for value in image_rect)
    except (TypeError, ValueError, OverflowError):
        return None
    values = (px, py, left, top, right, bottom)
    if not all(math.isfinite(value) for value in values):
        return None
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None
    if px < left or py < top or px >= right or py >= bottom:
        return None
    # Keep the normalized coordinate far enough below 1.0 that downstream
    # thousand-scale displays/rounding cannot turn an edge click into (1000,1000).
    return (min(0.999, max(0.0, (px - left) / width)),
            min(0.999, max(0.0, (py - top) / height)))


def sanitize_prompt_text(
    text: Any,
    max_length: int = MAX_PROMPT_LENGTH,
    allow_empty: bool = True,
    field_name: str = "prompt",
) -> str:
    """Sanitize user text by stripping control characters and enforcing length limits."""
    if text is None:
        raw = ""
    elif isinstance(text, (bytes, bytearray)):
        raw = bytes(text).decode("utf-8", errors="replace")
    else:
        raw = str(text)
    cleaned = _CONTROL_CHAR_RE.sub("", raw.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    if not allow_empty and not cleaned:
        raise ValidationError(f"{field_name.capitalize()} cannot be empty.")
    return cleaned


def validate_int_range(
    value: Any,
    minimum: int,
    maximum: int,
    field_name: str = "value",
    clamp: bool = False,
) -> int:
    """Validate an integer parameter, rejecting booleans and non-numeric values."""
    if isinstance(value, bool):
        raise ValidationError(f"{field_name} must be an integer, not a boolean.")
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValidationError(f"{field_name} must be a finite integer.")
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be a valid integer (got {value!r}).") from exc
    if clamp:
        return max(minimum, min(maximum, parsed))
    if parsed < minimum or parsed > maximum:
        raise ValidationError(f"{field_name} must be between {minimum} and {maximum} (got {parsed}).")
    return parsed


def validate_float_range(
    value: Any,
    minimum: float,
    maximum: float,
    field_name: str = "value",
    clamp: bool = False,
) -> float:
    """Validate a finite floating-point parameter within [minimum, maximum]."""
    if isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be a valid number (got {value!r}).") from exc
    if math.isnan(parsed) or math.isinf(parsed):
        raise ValidationError(f"{field_name} must be a finite number.")
    if clamp:
        return max(minimum, min(maximum, parsed))
    if parsed < minimum or parsed > maximum:
        raise ValidationError(f"{field_name} must be between {minimum} and {maximum} (got {parsed}).")
    return parsed


def validate_dimensions(
    width: Any,
    height: Any,
    min_side: int = 64,
    max_side: int = MAX_OUTPUT_SIDE,
    snap_multiple: int = 32,
) -> tuple[int, int]:
    """Validate width and height, rejecting non-positive values and snapping to multiples."""
    w = validate_int_range(width, 1, max_side * 4, field_name="width", clamp=False)
    h = validate_int_range(height, 1, max_side * 4, field_name="height", clamp=False)
    if snap_multiple > 1:
        w = int(round(w / float(snap_multiple))) * snap_multiple
        h = int(round(h / float(snap_multiple))) * snap_multiple
    w = max(min_side, min(max_side, w))
    h = max(min_side, min(max_side, h))
    return w, h


def validate_focus_point(focus_point: Any) -> Optional[tuple[float, float]]:
    """Validate optional normalized (x, y) focus coordinates in [0.0, 1.0]."""
    if focus_point is None:
        return None
    if not isinstance(focus_point, (tuple, list)) or len(focus_point) != 2:
        raise ValidationError("focus_point must be a (x, y) pair of normalized coordinates.")
    x = validate_float_range(focus_point[0], 0.0, 1.0, field_name="focus_point.x", clamp=True)
    y = validate_float_range(focus_point[1], 0.0, 1.0, field_name="focus_point.y", clamp=True)
    return (x, y)


def validate_normalized_bbox(value: Any, field_name: str = "bbox") -> tuple[float, float, float, float]:
    """Validate finite normalized XYXY coordinates in the original image frame."""
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValidationError(f"{field_name} must contain four normalized XYXY coordinates.")
    if any(isinstance(item, bool) for item in value):
        raise ValidationError(f"{field_name} coordinates must be numeric, not boolean.")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} coordinates must be finite numbers.") from exc
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        raise ValidationError(f"{field_name} coordinates must be finite numbers.")
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValidationError(f"{field_name} must be ordered XYXY coordinates within [0, 1].")
    return (x0, y0, x1, y1)


def validate_image_file(
    path: str | Path,
    max_bytes: int = MAX_IMAGE_FILE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[Path, tuple[int, int], str]:
    """Validate an input image file path, extension, byte size, header, and pixel bounds."""
    if not path or not str(path).strip():
        raise ValidationError("Image path cannot be empty.")
    resolved = Path(str(path)).expanduser().resolve()
    if not resolved.exists():
        raise ValidationError(f"Image file does not exist: {resolved}")
    if not resolved.is_file():
        raise ValidationError(f"Image path is not a regular file: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValidationError(f"Unsupported image extension '{suffix}'. Allowed: {allowed}")
    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise ValidationError(f"Cannot stat image file {resolved}: {exc}") from exc
    if size_bytes <= 0:
        raise ValidationError(f"Image file is empty (0 bytes): {resolved}")
    if size_bytes > max_bytes:
        raise ValidationError(
            f"Image file exceeds maximum allowed size ({size_bytes / (1024 * 1024):.1f} MiB > "
            f"{max_bytes / (1024 * 1024):.0f} MiB): {resolved}"
        )
    try:
        with Image.open(resolved) as img:
            fmt = str(img.format or suffix.lstrip(".").upper())
            transposed = ImageOps.exif_transpose(img)
            width, height = transposed.size
    except Exception as exc:
        raise ValidationError(f"Corrupt or unreadable image file '{resolved.name}': {exc}") from exc
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ValidationError(
            f"Image dimensions ({width}x{height}) are too small; minimum supported is {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE}."
        )
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise ValidationError(
            f"Image dimensions ({width}x{height}) exceed the maximum supported side ({MAX_IMAGE_SIDE}px)."
        )
    if width * height > max_pixels:
        raise ValidationError(
            f"Image total pixel count ({width * height:,}) exceeds safety limit ({max_pixels:,} pixels)."
        )
    return resolved, (width, height), fmt


def validate_model_file(path: str | Path, role: str = "") -> Path:
    """Validate a local GGUF or Safetensors model checkpoint file and check header magic."""
    if not path or not str(path).strip():
        raise ValidationError("Model file path cannot be empty.")
    resolved = Path(str(path)).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValidationError(f"Model file does not exist: {resolved}")
    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise ValidationError(f"Could not inspect model file {resolved}: {exc}") from exc
    if size_bytes < 16:
        raise ValidationError(f"Model file is empty or truncated ({size_bytes} bytes): {resolved.name}")
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_MODEL_EXTENSIONS:
        raise ValidationError(f"Unsupported model file extension '{suffix}' for {resolved.name}.")
    with resolved.open("rb") as handle:
        head = handle.read(64)
    head_lower = head.lstrip().lower()
    if head_lower.startswith((b"<!doctype", b"<html", b"<?xml", b'{"error"')):
        raise ValidationError(
            f"File '{resolved.name}' contains an HTML/JSON error response instead of binary model weights."
        )
    if suffix == ".gguf":
        if head[:4] != b"GGUF":
            raise ValidationError(f"File '{resolved.name}' does not have a valid GGUF magic header.")
        version = int.from_bytes(head[4:8], "little", signed=False)
        if version not in (1, 2, 3):
            raise ValidationError(f"File '{resolved.name}' has unsupported GGUF version {version}.")
    elif suffix in {".safetensors", ".sft"}:
        header_len = int.from_bytes(head[:8], "little", signed=False)
        if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES or (8 + header_len) > size_bytes:
            raise ValidationError(f"File '{resolved.name}' has an invalid safetensors header length.")
        if head[8:9] != b"{":
            raise ValidationError(f"File '{resolved.name}' safetensors metadata header is not valid JSON.")
    if role:
        from .workspace import _model_role_mismatch
        mismatch = _model_role_mismatch(role, resolved.name)
        if mismatch:
            raise ValidationError(mismatch)
    return resolved


def validate_lora_file(path: str | Path) -> Path:
    """Validate a local LoRA adapter file (.safetensors, .sft, or .gguf)."""
    if not path or not str(path).strip():
        raise ValidationError("LoRA file path cannot be empty.")
    resolved = Path(str(path)).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValidationError(f"LoRA file does not exist: {resolved}")
    if resolved.suffix.lower() not in ALLOWED_LORA_EXTENSIONS:
        raise ValidationError(
            f"Unsupported LoRA file extension '{resolved.suffix}'. Expected .safetensors, .sft, or .gguf."
        )
    return validate_model_file(resolved)


def validate_archive_member_path(root: Path, member_name: str) -> Path:
    """Prevent Zip-Slip / path traversal when extracting runtime or model archives."""
    cleaned = str(member_name or "").replace("\\", "/").strip()
    if not cleaned or "\x00" in cleaned:
        raise ValidationError("Archive contains an empty or null-byte member filename.")
    if cleaned.startswith("/") or re.match(r"^[a-zA-Z]:", cleaned):
        raise ValidationError(f"Unsafe absolute or drive path in archive: {member_name!r}")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValidationError(f"Unsafe parent directory traversal in archive: {member_name!r}")
    root_resolved = root.resolve()
    destination = (root_resolved / Path(*parts)).resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise ValidationError(f"Archive member escapes target root directory: {member_name!r}")
    return destination


def validate_https_url(url: str, allowed_hosts: Optional[set[str]] = None) -> str:
    """Validate that a download URL is well-formed HTTPS and optionally matches allowed hosts."""
    text = str(url or "").strip()
    if not text or _CONTROL_CHAR_RE.search(text) or " " in text:
        raise ValidationError(f"Invalid URL: {url!r}")
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme.lower() != "https":
        raise ValidationError(f"Only HTTPS download URLs are permitted (got scheme '{parsed.scheme}').")
    if not parsed.hostname:
        raise ValidationError(f"URL is missing a valid hostname: {url!r}")
    if parsed.username or parsed.password:
        raise ValidationError("Download URLs must not embed user credentials.")
    if allowed_hosts is not None and parsed.hostname.lower() not in {h.lower() for h in allowed_hosts}:
        raise ValidationError(f"Host '{parsed.hostname}' is not in the allowed download hosts.")
    return text


def validate_loopback_url(url: str) -> str:
    """Verify that a local inference server URL targets loopback (127.0.0.1 or localhost) only."""
    text = str(url or "").strip()
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme.lower() != "http":
        raise ValidationError(f"Local engine URL must use http:// loopback (got {text!r}).")
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValidationError(f"Local engine must bind to loopback only (got host {parsed.hostname!r}).")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValidationError(f"Invalid loopback port in {text!r}") from exc
    if port is None or not (1 <= int(port) <= 65535):
        raise ValidationError(f"Local engine URL requires a valid TCP port (1..65535): {text!r}")
    return text


def validate_sha256_hex(digest: str, allow_empty: bool = False) -> str:
    """Validate a 64-character hexadecimal SHA-256 checksum string."""
    text = str(digest or "").strip().lower()
    if not text and allow_empty:
        return ""
    if not _SHA256_HEX_RE.match(text):
        raise ValidationError(f"Invalid SHA-256 hex digest: {digest!r}")
    return text


def validate_job_custom(custom: Any) -> dict[str, Any]:
    """Validate and normalize the custom dictionary on a Job instance."""
    if custom is None:
        return {}
    if not isinstance(custom, dict):
        raise ValidationError("Job.custom must be a dictionary.")
    out = dict(custom)
    if "num_inference_steps" in out:
        out["num_inference_steps"] = validate_int_range(
            out["num_inference_steps"], 1, 100, "num_inference_steps", clamp=True
        )
    if "max_attempts" in out:
        out["max_attempts"] = validate_int_range(out["max_attempts"], 1, 5, "max_attempts", clamp=True)
    if "retry_diff" in out:
        out["retry_diff"] = validate_float_range(out["retry_diff"], 1.0, 100.0, "retry_diff", clamp=True)
    if "warn_diff" in out:
        out["warn_diff"] = validate_float_range(out["warn_diff"], 1.0, 100.0, "warn_diff", clamp=True)
    if "qwen_detail_profile" in out:
        prof = str(out["qwen_detail_profile"] or "balanced").strip().lower()
        out["qwen_detail_profile"] = prof if prof in QWEN_DETAIL_PROFILES else "balanced"
    if "grounding_profile" in out:
        out["grounding_profile"] = normalize_grounding_profile(out["grounding_profile"])
    if "enhance_detail_mode" in out:
        mode = str(out["enhance_detail_mode"] or "fast").strip().lower()
        out["enhance_detail_mode"] = mode if mode in {"fast", "balanced", "ultimate"} else "fast"
    if "ultimate_qwen_denoise_strength" in out:
        out["ultimate_qwen_denoise_strength"] = validate_float_range(
            out["ultimate_qwen_denoise_strength"], 0.05, 0.95, "ultimate_qwen_denoise_strength", clamp=True
        )
    if "ultimate_qwen_tile_coverage" in out:
        cov = validate_float_range(
            out["ultimate_qwen_tile_coverage"], 0.45, 75.0, "ultimate_qwen_tile_coverage", clamp=True
        )
        if cov > 1.0:
            cov = cov / 100.0
        out["ultimate_qwen_tile_coverage"] = max(0.45, min(0.75, cov))
    if "ultimate_qwen_passes" in out:
        out["ultimate_qwen_passes"] = validate_int_range(
            out["ultimate_qwen_passes"], 1, 2, "ultimate_qwen_passes", clamp=True
        )
    if "ultimate_qwen_steps" in out:
        out["ultimate_qwen_steps"] = validate_int_range(
            out["ultimate_qwen_steps"], 4, 60, "ultimate_qwen_steps", clamp=True
        )
    if "ultimate_qwen_scale" in out:
        out["ultimate_qwen_scale"] = validate_int_range(
            out["ultimate_qwen_scale"], 1, 4, "ultimate_qwen_scale", clamp=True
        )
    out.pop("upscale_scale", None)
    if "upscale_tile" in out:
        tile = validate_int_range(out["upscale_tile"], 0, 2048, "upscale_tile", clamp=True)
        out["upscale_tile"] = 0 if tile < 32 else tile
    if "upscale_model" in out:
        model_id = str(out["upscale_model"] or "realesrgan-x4plus").strip()
        out["upscale_model"] = model_id if model_id in REALESRGAN_CATALOG_BY_ID else "realesrgan-x4plus"
    return out


def validate_job(job: Any, require_existing_input: bool = False) -> Any:
    """Validate all fields of a Job instance in-place and return it."""
    workflow = str(getattr(job, "workflow", "") or "").strip()
    if workflow not in VALID_WORKFLOWS:
        raise ValidationError(f"Unknown workflow: {workflow!r}. Valid workflows: {', '.join(VALID_WORKFLOWS)}")
    job.workflow = workflow

    job.prompt = sanitize_prompt_text(
        getattr(job, "prompt", ""),
        max_length=MAX_PROMPT_LENGTH,
        allow_empty=(workflow != "image_generate"),
        field_name="prompt",
    )
    job.target = sanitize_prompt_text(
        getattr(job, "target", ""),
        max_length=MAX_TARGET_LENGTH,
        allow_empty=True,
        field_name="target",
    )
    job.trigger_word = sanitize_prompt_text(
        getattr(job, "trigger_word", ""),
        max_length=MAX_TRIGGER_LENGTH,
        allow_empty=True,
        field_name="trigger_word",
    )

    if workflow in {"lora_closeup_1024", "square_1_1"}:
        try:
            requested_side = int(getattr(job, "side", 1024))
        except (TypeError, ValueError):
            requested_side = 1024
        side = min(FIXED_SQUARE_OUTPUT_SIZES, key=lambda s: abs(s - requested_side))
        job.side = side
        job.width = side
        job.height = side
    else:
        job.width = validate_int_range(getattr(job, "width", 1024), 64, MAX_OUTPUT_SIDE, "width", clamp=True)
        job.height = validate_int_range(getattr(job, "height", 1024), 64, MAX_OUTPUT_SIDE, "height", clamp=True)

    job.batch_count = validate_int_range(getattr(job, "batch_count", 1), 1, 100, "batch_count", clamp=True)
    job.batch_index = validate_int_range(getattr(job, "batch_index", 0), 0, 10000, "batch_index", clamp=True)
    job.seed = validate_int_range(getattr(job, "seed", -1), -1, 2**31 - 1, "seed", clamp=True)

    seed_mode = str(getattr(job, "seed_mode", "Random") or "Random").strip()
    job.seed_mode = seed_mode if seed_mode in VALID_SEED_MODES else "Random"

    fmt = str(getattr(job, "output_format", "png") or "png").strip().lower().lstrip(".")
    job.output_format = fmt if fmt in VALID_OUTPUT_FORMATS else "png"

    job.zoom_factor = validate_float_range(getattr(job, "zoom_factor", 1.05), 1.05, 6.0, "zoom_factor", clamp=True)
    job.focus_point = validate_focus_point(getattr(job, "focus_point", None))
    job.custom = validate_job_custom(getattr(job, "custom", {}))

    if require_existing_input and workflow != "image_generate":
        input_path = getattr(job, "input_path", None)
        if not input_path or not str(input_path).strip():
            raise ValidationError(f"Workflow '{workflow}' requires an input image.")
        resolved, _size, _fmt = validate_image_file(input_path)
        job.input_path = str(resolved)

    return job


def validate_and_migrate_settings(raw_obj: dict[str, Any], decode_path_fn=None) -> dict[str, Any]:
    """Validate and migrate a raw settings dictionary loaded from settings.json."""
    if not isinstance(raw_obj, dict):
        raise ValidationError("Settings payload must be a JSON object.")
    version = validate_int_range(raw_obj.get("settings_version", 0), 0, 10000, "settings_version")
    if version < 16 or version > SETTINGS_VERSION:
        raise ValidationError(f"Unsupported settings_version {version} (expected 16..{SETTINGS_VERSION}).")

    decoded: dict[str, Any] = {}
    for k, v in raw_obj.items():
        if isinstance(v, str) and decode_path_fn is not None:
            decoded[k] = decode_path_fn(v)
        else:
            decoded[k] = v

    if version < 38:
        decoded["ultimate_qwen_upscale_enabled"] = False
        decoded["enhance_detail_mode"] = "fast"
    if "encoder_cpu_auto" not in raw_obj and "encoder_cpu" in raw_obj:
        decoded["encoder_cpu_auto"] = False
    if version < 21:
        decoded["max_attempts"] = 1
    if version < 24:
        decoded["quality"] = "Turbo · 8 steps"
        decoded["num_inference_steps"] = 8
    # Version 41 changes the default Auto preview to TAE for better readability.
    # Preserve explicit choices other than the old default projection method.
    if version < 41 and str(decoded.get("live_preview_method", "tae")).strip().lower() == "proj":
        decoded["live_preview_method"] = "tae"
    if version < 39:
        for quality_key in ("quality", "std_quality", "gen_quality", "edit_quality"):
            if decoded.get(quality_key) == "Turbo · 9 steps":
                decoded[quality_key] = "Turbo · 8 steps"
        if int(decoded.get("num_inference_steps", 8) or 8) == 9:
            decoded["num_inference_steps"] = 8
        if int(decoded.get("ultimate_qwen_steps", 8) or 8) == 9:
            decoded["ultimate_qwen_steps"] = 8
        if int(decoded.get("ultimate_qwen_tile_coverage", 50) or 50) == 60:
            decoded["ultimate_qwen_tile_coverage"] = 50
    # v43 restores the default Ultimate tile step setting to eight and adds an
    # independent 1×–4× Ultimate output-scale control. Turbo always uses its exact
    # checkpoint-defined eight-step schedule, irrespective of this non-Turbo control.
    if version < 43:
        decoded["ultimate_qwen_steps"] = 8
        decoded.setdefault("ultimate_qwen_scale", 2)
    if version < 44:
        if "ultimate_qwen_scale" not in raw_obj or int(raw_obj.get("ultimate_qwen_scale", 1) or 1) == 1:
            decoded["ultimate_qwen_scale"] = 2
    if version < 44:
        decoded.setdefault("preview_mode", "Side by side")
    valid_preview_modes = {"Enhanced", "Original", "Slider before/after", "Side by side", "Overlay", "Difference", "Flicker"}
    preview_mode = str(decoded.get("preview_mode", "Side by side"))
    decoded["preview_mode"] = preview_mode if preview_mode in valid_preview_modes else "Side by side"
    decoded["ultimate_qwen_scale"] = validate_int_range(
        decoded.get("ultimate_qwen_scale", 2), 1, 4, "ultimate_qwen_scale", clamp=True
    )
    decoded["ultimate_qwen_denoise_strength"] = validate_float_range(
        decoded.get("ultimate_qwen_denoise_strength", 0.20), 0.05, 0.95, "ultimate_qwen_denoise_strength", clamp=True
    )
    cov_val = decoded.get("ultimate_qwen_tile_coverage", 50)
    if isinstance(cov_val, float) and cov_val <= 1.0:
        cov_val = int(round(cov_val * 100))
    decoded["ultimate_qwen_tile_coverage"] = validate_int_range(
        cov_val, 45, 75, "ultimate_qwen_tile_coverage", clamp=True
    )
    decoded["ultimate_qwen_passes"] = validate_int_range(
        decoded.get("ultimate_qwen_passes", 1), 1, 2, "ultimate_qwen_passes", clamp=True
    )
    decoded["ultimate_qwen_steps"] = validate_int_range(
        decoded.get("ultimate_qwen_steps", 8), 4, 60, "ultimate_qwen_steps", clamp=True
    )
    if version < 28:
        decoded["retry_diff"] = 25.0
        decoded["warn_diff"] = 18.0
        decoded["max_attempts"] = 1
        decoded["gpu_memory_mode"] = "auto"
    if version < 33:
        decoded["gpu_memory_mode"] = "auto"
        decoded["encoder_cpu"] = False
        decoded["encoder_cpu_auto"] = False
        decoded["enhance_detail_mode"] = "fast"

    if "std_output_scale" not in raw_obj:
        legacy_size = str(decoded.get("std_size_preset", "Follow input"))
        legacy_dims = {
            "1024 × 1024": (1024, 1024),
            "2048 × 2048": (2048, 2048),
            "1344 × 768 (16:9)": (1344, 768),
            "768 × 1344 (9:16)": (768, 1344),
            "1536 × 1024 (3:2)": (1536, 1024),
            "1024 × 1536 (2:3)": (1024, 1536),
        }
        if legacy_size == "Custom dimensions":
            decoded["std_size_preset"] = "Custom dimensions"
        elif legacy_size in ("Follow input", "Keep Original Size (Auto)"):
            decoded["std_size_preset"] = "Scale"
            decoded["std_output_scale"] = 1
        elif legacy_size in legacy_dims:
            decoded["std_size_preset"] = "Custom dimensions"
            decoded["std_output_width"], decoded["std_output_height"] = legacy_dims[legacy_size]
        else:
            decoded["std_size_preset"] = "Scale"
            decoded["std_output_scale"] = 2

    if decoded.get("std_size_preset") not in ("Scale", "Custom dimensions"):
        decoded["std_size_preset"] = "Scale"
    decoded["std_output_scale"] = validate_int_range(
        decoded.get("std_output_scale", 2), 1, 4, "std_output_scale", clamp=True
    )
    if decoded.get("quality") not in QUALITY_PRESETS:
        decoded["quality"] = "Turbo · 8 steps"
    for quality_key in ("std_quality", "gen_quality", "edit_quality"):
        if quality_key not in raw_obj or decoded.get(quality_key) not in QUALITY_PRESETS:
            decoded[quality_key] = decoded.get("quality", "Turbo · 8 steps")
    if "num_inference_steps" not in raw_obj:
        decoded["num_inference_steps"] = 8
    else:
        decoded["num_inference_steps"] = validate_int_range(
            decoded.get("num_inference_steps", 8), 4, 60, "num_inference_steps", clamp=True
        )
    if "upscale_tile" not in raw_obj:
        decoded["upscale_tile"] = 0
    if "ultimate_qwen_upscale_enabled" not in raw_obj:
        decoded["ultimate_qwen_upscale_enabled"] = str(decoded.get("enhance_detail_mode", "balanced")) != "fast"
    if str(decoded.get("gpu_memory_mode", "auto")).strip().lower() == "fast":
        decoded["gpu_memory_mode"] = "full_gpu"
    if str(decoded.get("gpu_memory_mode", "auto")).strip().lower() not in VALID_MEMORY_MODES:
        decoded["gpu_memory_mode"] = "auto"
    if decoded.get("edit_size") == "Keep Original Size (Auto)":
        decoded["edit_size"] = "Original Size"
    if str(decoded.get("qwen_detail_profile", "balanced")).strip().lower() not in QWEN_DETAIL_PROFILES:
        decoded["qwen_detail_profile"] = "balanced"
    decoded["grounding_profile"] = normalize_grounding_profile(decoded.get("grounding_profile", "balanced"))
    if str(decoded.get("live_preview_method", "tae")).strip().lower() not in VALID_PREVIEW_METHODS:
        decoded["live_preview_method"] = "tae"
    decoded["settings_version"] = SETTINGS_VERSION
    return {**DEFAULTS, **decoded}
