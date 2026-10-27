# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for `pai_studio.validation` (Input Validation & Security).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pai_studio.engine import Job
from pai_studio.validation import (
    ValidationError,
    sanitize_prompt_text,
    validate_and_migrate_settings,
    validate_archive_member_path,
    validate_dimensions,
    validate_float_range,
    validate_focus_point,
    validate_https_url,
    validate_image_file,
    validate_int_range,
    validate_job,
    validate_job_custom,
    validate_loopback_url,
    validate_lora_file,
    validate_model_file,
    validate_sha256_hex,
)


def _expect_validation_error(fn, *args, **kwargs) -> ValidationError:
    try:
        fn(*args, **kwargs)
    except ValidationError as exc:
        return exc
    raise AssertionError(f"Expected ValidationError from {fn.__name__}({args!r}, {kwargs!r})")


def test_sanitize_prompt_strips_control_chars_and_enforces_limits():
    raw = "  portrait\x00 of a \x07cat\r\nin studio light  "
    cleaned = sanitize_prompt_text(raw)
    assert cleaned == "portrait of a cat\nin studio light"
    assert "\x00" not in cleaned and "\x07" not in cleaned

    long_prompt = "a" * 5000
    assert len(sanitize_prompt_text(long_prompt, max_length=4000)) == 4000

    _expect_validation_error(sanitize_prompt_text, "   \x00  ", allow_empty=False, field_name="prompt")


def test_validate_int_and_float_ranges_reject_booleans_and_nan():
    assert validate_int_range(8, 1, 60) == 8
    assert validate_int_range("16", 1, 60) == 16
    assert validate_int_range(120, 1, 60, clamp=True) == 60
    _expect_validation_error(validate_int_range, True, 0, 10)
    _expect_validation_error(validate_int_range, float("nan"), 0, 10)
    _expect_validation_error(validate_int_range, 0, 1, 10, clamp=False)

    assert validate_float_range(0.46, 0.0, 1.0) == 0.46
    assert validate_float_range(2.5, 0.0, 1.0, clamp=True) == 1.0
    _expect_validation_error(validate_float_range, False, 0.0, 1.0)
    _expect_validation_error(validate_float_range, float("inf"), 0.0, 1.0)


def test_validate_dimensions_snaps_to_multiple_of_32_and_clamps():
    assert validate_dimensions(1000, 1030) == (992, 1024)
    assert validate_dimensions(32, 9000) == (64, 8192)
    _expect_validation_error(validate_dimensions, 0, 1024)
    _expect_validation_error(validate_dimensions, -512, 1024)


def test_validate_focus_point_normalizes_and_rejects_bad_shapes():
    assert validate_focus_point(None) is None
    assert validate_focus_point((0.25, 0.75)) == (0.25, 0.75)
    assert validate_focus_point([-0.5, 1.5]) == (0.0, 1.0)
    _expect_validation_error(validate_focus_point, (0.5,))
    _expect_validation_error(validate_focus_point, "center")


def test_validate_image_file_checks_existence_extension_headers_and_bounds(tmp_path):
    valid_img = tmp_path / "sample.png"
    Image.new("RGB", (128, 96), (40, 80, 120)).save(valid_img)
    resolved, size, fmt = validate_image_file(valid_img)
    assert resolved == valid_img.resolve()
    assert size == (128, 96)
    assert fmt == "PNG"

    # Missing file
    _expect_validation_error(validate_image_file, tmp_path / "missing.png")

    # Unsupported extension
    bad_ext = tmp_path / "sample.exe"
    bad_ext.write_bytes(b"MZ")
    _expect_validation_error(validate_image_file, bad_ext)

    # Empty file
    empty_img = tmp_path / "empty.png"
    empty_img.write_bytes(b"")
    _expect_validation_error(validate_image_file, empty_img)

    # Corrupt image payload
    corrupt_img = tmp_path / "corrupt.png"
    corrupt_img.write_bytes(b"not-a-valid-png-header")
    _expect_validation_error(validate_image_file, corrupt_img)

    # Too small image (< 16x16)
    tiny_img = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(tiny_img)
    _expect_validation_error(validate_image_file, tiny_img)

    # Pixel bomb limit
    _expect_validation_error(validate_image_file, valid_img, max_pixels=1000)


def test_validate_model_and_lora_files_verify_gguf_and_safetensors_headers(tmp_path):
    # Valid GGUF v3 header
    gguf_file = tmp_path / "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf"
    gguf_file.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + b"\x00" * 32)
    assert validate_model_file(gguf_file, role="diffusion") == gguf_file.resolve()

    # Role mismatch caught by validate_model_file
    encoder_gguf = tmp_path / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
    encoder_gguf.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + b"\x00" * 32)
    _expect_validation_error(validate_model_file, encoder_gguf, role="diffusion")
    assert validate_model_file(encoder_gguf, role="text_encoder") == encoder_gguf.resolve()

    # HTML error page disguised as GGUF weight file
    html_gguf = tmp_path / "corrupt.gguf"
    html_gguf.write_bytes(b"<!DOCTYPE html><html><body>Rate limited</body></html>")
    _expect_validation_error(validate_model_file, html_gguf)

    # Valid safetensors LoRA file
    header_bytes = json.dumps({"__metadata__": {"modelspec.architecture": "qwen-image-2.1"}}).encode("utf-8")
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + b"\x00" * 32)
    assert validate_lora_file(lora_file) == lora_file.resolve()

    # Corrupt safetensors with bogus header length exceeding file size
    bad_lora = tmp_path / "bad.safetensors"
    bad_lora.write_bytes((999_999).to_bytes(8, "little") + b"{}" + b"\x00" * 16)
    _expect_validation_error(validate_lora_file, bad_lora)


def test_validate_archive_member_blocks_zip_slip_traversal(tmp_path):
    dest = validate_archive_member_path(tmp_path, "bin/sd-server.exe")
    assert dest == (tmp_path / "bin" / "sd-server.exe").resolve()

    _expect_validation_error(validate_archive_member_path, tmp_path, "../escaped.dll")
    _expect_validation_error(validate_archive_member_path, tmp_path, "sub/../../escaped.dll")
    _expect_validation_error(validate_archive_member_path, tmp_path, "/etc/passwd")
    _expect_validation_error(validate_archive_member_path, tmp_path, "C:/Windows/System32/cmd.exe")


def test_validate_urls_and_sha256_digests():
    url = "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/model.gguf"
    assert validate_https_url(url, allowed_hosts={"huggingface.co", "hf-mirror.com"}) == url
    _expect_validation_error(validate_https_url, "http://huggingface.co/model.gguf")
    _expect_validation_error(validate_https_url, "https://evil.example.com/model.gguf", allowed_hosts={"huggingface.co"})

    assert validate_loopback_url("http://127.0.0.1:8080/health") == "http://127.0.0.1:8080/health"
    _expect_validation_error(validate_loopback_url, "http://0.0.0.0:8080/health")
    _expect_validation_error(validate_loopback_url, "http://192.168.1.10:8080/health")

    valid_sha = "2f08572e143f0b15a764197ea937d219dd70f0cc7eefb30eb84c9dae6a7cba6d"
    assert validate_sha256_hex(valid_sha.upper()) == valid_sha
    assert validate_sha256_hex("", allow_empty=True) == ""
    _expect_validation_error(validate_sha256_hex, "not-a-sha256")


def test_validate_job_and_settings_migration(tmp_path):
    img_path = tmp_path / "input.png"
    Image.new("RGB", (256, 256), (10, 20, 30)).save(img_path)

    job = Job(
        workflow="enhance",
        input_path=str(img_path),
        prompt="restore fine texture\x00",
        width=1024,
        height=1024,
        custom={"num_inference_steps": 8, "ultimate_qwen_tile_coverage": 50},
    )
    validated = job.validate(require_existing_input=True)
    assert validated.prompt == "restore fine texture"
    assert validated.custom["ultimate_qwen_tile_coverage"] == 0.50

    # Generate workflow requires a non-empty prompt
    empty_gen = Job(workflow="image_generate", prompt="   ")
    _expect_validation_error( validate_job, empty_gen)

    # Unknown workflow rejected
    bad_wf = Job(workflow="unknown_mode", prompt="test")
    _expect_validation_error(validate_job, bad_wf)

    # Settings migration from v38 to the current schema version
    migrated = validate_and_migrate_settings({
        "settings_version": 38,
        "quality": "Turbo · 9 steps",
        "num_inference_steps": 9,
        "ultimate_qwen_steps": 9,
        "ultimate_qwen_tile_coverage": 60,
    })
    from pai_studio.constants import SETTINGS_VERSION
    assert migrated["settings_version"] == SETTINGS_VERSION
    assert migrated["quality"] == "Turbo · 8 steps"
    assert migrated["num_inference_steps"] == 8
    assert migrated["ultimate_qwen_steps"] == 8
    assert migrated["ultimate_qwen_scale"] == 2
    assert migrated["ultimate_qwen_tile_coverage"] == 50


def test_preview_click_uses_image_rectangle_and_rejects_clicks_outside_the_image():
    from pai_studio.validation import preview_click_to_normalized

    # The rectangle is the actual rendered image in a side-by-side pane, not the whole widget.
    assert preview_click_to_normalized(750, 320, (500, 200, 1000, 600)) == (0.5, 0.3)
    assert preview_click_to_normalized(500, 200, (500, 200, 1000, 600)) == (0.0, 0.0)
    assert preview_click_to_normalized(999, 599, (500, 200, 1000, 600)) == (0.998, 0.9975)
    assert preview_click_to_normalized(499, 320, (500, 200, 1000, 600)) is None
    assert preview_click_to_normalized(750, 600, (500, 200, 1000, 600)) is None


def test_preview_focus_point_never_rounds_to_the_1000_1000_edge_sentinel():
    from pai_studio.validation import preview_click_to_normalized

    point = preview_click_to_normalized(9999, 9999, (0, 0, 10000, 10000))
    assert point is not None
    assert point[0] <= 0.999 and point[1] <= 0.999
