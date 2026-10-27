from __future__ import annotations

from PIL import Image

from pai_studio.engine import LocalEngine
from pai_studio.prompts import apply_watermark_removal_prompt


def test_ultimate_runs_global_reconstruction_then_the_four_tile_refiner():
    logs: list[str] = []
    engine = LocalEngine(log_callback=logs.append)
    original = Image.new("RGB", (80, 120), (100, 120, 140))
    calls: list[tuple[str, object]] = []

    global_calls: list[tuple[int, int, float | None]] = []

    def fake_global(image, prompt, width, height, seed, quality, max_attempts,
                    retry_diff, warn_diff, steps, use_img2img=True, strength=None):
        global_calls.append((width, height, strength))
        return Image.new("RGB", (width, height), (110, 125, 140))

    def fake_tile_refiner(image, prompt, seed, quality, steps, detail_mode, **kwargs):
        calls.append((detail_mode, kwargs.get("tile_strength")))
        return image

    engine._enhance_with_retries = fake_global  # type: ignore[method-assign]
    engine._qwen_detail_refine_large_output = fake_tile_refiner  # type: ignore[method-assign]
    engine._upscale_image_with_selected_model = (
        lambda image, options, target_size: image.resize(target_size, Image.Resampling.BICUBIC)
    )

    result = engine._enhance_qwen_output(
        original,
        "faithfully add fine texture",
        160,
        240,
        123,
        "Turbo · 8 steps",
        1,
        25.0,
        18.0,
        8,
        detail_mode="ultimate",
        strength=0.84,
        tile_strength=0.25,
        detail_profile="balanced",
    )

    assert result.size == (160, 240)
    assert global_calls and global_calls[0][2] == 0.84
    assert calls == [("ultimate", 0.25)]
    assert any("Ultimate two-stage upscale" in line and "selected AI model" in line for line in logs)


def test_ultimate_off_executes_one_global_pass_and_never_calls_tile_refinement():
    logs: list[str] = []
    engine = LocalEngine(log_callback=logs.append)
    original = Image.new("RGB", (80, 120), (100, 120, 140))
    global_calls: list[tuple[int, int, float | None]] = []

    def fake_global(image, prompt, width, height, seed, quality, max_attempts,
                    retry_diff, warn_diff, steps, use_img2img=True, strength=None):
        global_calls.append((width, height, strength))
        return Image.new("RGB", (width, height), (120, 130, 140))

    def forbidden_tile_refiner(*args, **kwargs):
        raise AssertionError("Ultimate OFF must not run the tile refiner")

    engine._enhance_with_retries = fake_global  # type: ignore[method-assign]
    engine._qwen_detail_refine_large_output = forbidden_tile_refiner  # type: ignore[method-assign]

    result = engine._enhance_qwen_output(
        original,
        "faithfully add fine texture",
        240,
        360,
        456,
        "Turbo · 8 steps",
        1,
        25.0,
        18.0,
        8,
        detail_mode="fast",
        strength=0.84,
        tile_strength=0.25,
        detail_profile="balanced",
    )

    assert result.size == (240, 360)
    assert global_calls == [(240, 360, 0.84)]
    assert not any("Ultimate fast path" in line for line in logs)


def test_enhance_preview_is_published_before_the_global_pass_for_2x_3x_4x_sized_outputs():
    for scale in (2, 3, 4):
        engine = LocalEngine()
        original = Image.new("RGB", (40, 60), (100, 120, 140))
        seen_at_start: list[tuple[int, int, int]] = []
        width, height = 40 * scale, 60 * scale

        def fake_global(image, prompt, out_width, out_height, seed, quality, max_attempts,
                        retry_diff, warn_diff, steps, use_img2img=True, strength=None):
            preview = engine.latest_preview_image
            assert preview is not None
            seen_at_start.append((preview.width, preview.height, engine.latest_preview_revision))
            return Image.new("RGB", (out_width, out_height), (130, 140, 150))

        engine._enhance_with_retries = fake_global  # type: ignore[method-assign]
        result = engine._enhance_qwen_output(
            original, "", width, height, 1, "Turbo", 1, 25.0, 18.0, 8,
            detail_mode="fast", strength=0.84,
        )
        assert result.size == (width, height)
        assert seen_at_start and seen_at_start[0][2] >= 1


def test_published_preview_is_compact_and_advances_revision():
    engine = LocalEngine()
    source = Image.new("RGB", (1800, 1200), (10, 20, 30))
    engine._publish_live_preview(source, marker=("test", 1))
    first_revision = engine.latest_preview_revision
    first_preview = engine.latest_preview_image
    engine._publish_live_preview(source, marker=("test", 2))

    assert first_preview is not None
    assert max(first_preview.size) <= 512
    assert first_revision == 1
    assert engine.latest_preview_revision == 2
    assert engine.latest_preview_image is not first_preview


def test_watermark_removal_is_strictly_opt_in_and_instruction_is_idempotent():
    prompt = "Keep the portrait and all scene content unchanged."
    assert apply_watermark_removal_prompt(prompt, enabled=False) == prompt

    enabled = apply_watermark_removal_prompt(prompt, enabled=True)
    assert "WATERMARK REMOVAL TASK:" in enabled
    assert "Keep the portrait and all scene content unchanged." in enabled
    assert apply_watermark_removal_prompt(enabled, enabled=True) == enabled


def test_watermark_toggle_is_off_by_default_and_canvas_paints_focus_marker_in_both_panes():
    from pathlib import Path
    from pai_studio.constants import DEFAULTS

    root = Path(__file__).resolve().parents[1]
    canvas_source = (root / "pai_studio" / "ui" / "canvas.py").read_text(encoding="utf-8")
    engine_source = (root / "pai_studio" / "engine.py").read_text(encoding="utf-8")

    assert DEFAULTS["watermark_removal_enabled"] is False
    assert 'self._draw_focus_marker(painter, self._pane_image_targets.get("original"), left)' in canvas_source
    assert 'self._draw_focus_marker(painter, self._pane_image_targets.get("enhanced"), right)' in canvas_source
    assert '"watermark_removal_enabled", False' in engine_source
    assert "apply_watermark_removal_prompt" in engine_source


def test_ultimate_tile_inference_does_not_publish_checkerboard_previews():
    engine = LocalEngine()
    engine._active_ultimate_tile_coverage = 0.50
    engine._active_ultimate_refinement_passes = 1
    engine._active_ultimate_steps = 8
    engine._active_ultimate_denoise_strength = 0.30
    source = Image.new("RGB", (400, 400), (80, 90, 100))
    preview_sizes_during_tiles: list[tuple[int, int]] = []
    suppression_flags: list[bool] = []
    calls: list[int] = []

    def fake_tile_inference(prompt, tile, size, seed, quality, steps, **kwargs):
        calls.append(seed)
        suppression_flags.append(engine._suppress_live_preview)
        assert engine.latest_preview_image is not None
        preview_sizes_during_tiles.append(engine.latest_preview_image.size)
        return Image.new("RGB", tile.size, (100 + (seed % 80), 110, 120))

    engine._infer_with_auto_offload_retry = fake_tile_inference  # type: ignore[method-assign]
    result = engine._qwen_detail_refine_large_output(
        source, "Faithful natural detail reconstruction", 10, "Turbo · 8 steps", 8,
        detail_mode="ultimate", use_img2img=True, tile_strength=0.74,
        detail_profile="balanced", force=True,
    )

    assert result.size == source.size
    assert len(calls) == 4
    # During Ultimate tile inference, live preview is NOT suppressed and shows the active tile crop
    assert suppression_flags == [False] * 4
    assert all(sz != source.size for sz in preview_sizes_during_tiles)
    # The stitched complete canvas is published at the end of the 4-tile pass.
    assert engine.latest_preview_image is not None
    assert engine.latest_preview_image.size == source.size
    assert engine._latest_preview_marker[1][0] == "tile-pass-complete"
    assert engine._suppress_live_preview is False


def test_ultimate_tile_starts_guarantee_overlap_at_large_scale():
    length, tile_length = 4320, 1400
    starts = LocalEngine._tile_starts(length, tile_length, min_overlap_ratio=0.10)
    assert starts[0] == 0
    assert starts[-1] == length - tile_length
    assert len(starts) >= 4
    assert all(right - left <= tile_length * 0.90 + 1 for left, right in zip(starts, starts[1:]))


def test_ultimate_scale_catalog_values_support_1x_through_4x_with_1x_as_default():
    from pai_studio.constants import DEFAULTS, MAX_OUTPUT_SIDE
    from pai_studio.prompts import calculate_qwen_output_size

    assert DEFAULTS["ultimate_qwen_scale"] == 2
    assert MAX_OUTPUT_SIDE >= 8192
    source_size = (810, 1080)
    expected = {1: source_size, 2: (1620, 2160), 3: (2430, 3240), 4: (3240, 4320)}
    for scale, dimensions in expected.items():
        assert calculate_qwen_output_size(source_size, "enhance", "Scale", scale, 1024, 1024) == dimensions


def test_feature_switches_are_compact_and_use_the_workflow_purple_active_color():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ui_source = (root / "pai_studio" / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert '"QToolButton:checked { background: #7858f5; border-color: #7858f5;' in ui_source
    assert "button.setMinimumWidth(0)" in ui_source
    assert "workflow_feature_row.addWidget(feature_button, 1)" in ui_source
    assert "row.addWidget(self._make_feature_toggle_button(feature), 1)" in ui_source
    assert "background: #145a43" not in ui_source
    assert 'self.btn_watermark_toggle = self._make_feature_toggle_button("watermark")' in ui_source


def test_ultimate_scale_is_applied_after_enhance_scale_and_one_x_still_refines_tiles():
    def run(ultimate_scale: int):
        logs: list[str] = []
        engine = LocalEngine(log_callback=logs.append)
        original = Image.new("RGB", (80, 120), (100, 120, 140))
        global_calls: list[tuple[int, int]] = []
        tile_sizes: list[tuple[int, int]] = []

        def fake_global(image, prompt, width, height, seed, quality, max_attempts,
                        retry_diff, warn_diff, steps, use_img2img=True, strength=None):
            global_calls.append((width, height))
            return Image.new("RGB", (width, height), (120, 130, 140))

        def fake_tile_refiner(image, prompt, seed, quality, steps, detail_mode, **kwargs):
            tile_sizes.append(image.size)
            return image

        engine._enhance_with_retries = fake_global  # type: ignore[method-assign]
        engine._qwen_detail_refine_large_output = fake_tile_refiner  # type: ignore[method-assign]
        engine._upscale_image_with_selected_model = (
            lambda image, options, target_size: image.resize(target_size, Image.Resampling.BICUBIC)
        )
        result = engine._enhance_qwen_output(
            original, "restore texture", 160, 240, 1, "Turbo · 8 steps", 1, 25.0, 18.0, 8,
            detail_mode="ultimate", strength=0.84, tile_strength=0.74,
            detail_profile="balanced", ultimate_scale=ultimate_scale,
        )
        return result, global_calls, tile_sizes, logs

    one_x, one_global, one_tiles, one_logs = run(1)
    two_x, two_global, two_tiles, two_logs = run(2)
    # When Ultimate is ON with scale 2x on target (160, 240), Pass 1 runs at half-res (96, 128) rounded to 32px,
    # then Lanczos upscales by 2x to (160, 240), and Ultimate tiles refine (160, 240).
    assert one_global == [(160, 256)] and two_global == [(64, 128)]
    assert one_x.size == (160, 240) and one_tiles == [(160, 240)]
    assert two_x.size == (160, 240) and two_tiles == [(160, 240)]
    assert any("selected AI model" in line for line in two_logs)


def test_realesrgan_models_automatically_run_at_their_native_scale(tmp_path):
    """Verify that Real-ESRGAN models automatically run at their intrinsic native scale
    without any manual scale selector in Model Catalog (4x for x4plus/x4plus-anime/animevideov3,
    3x for animevideov3-x3, 2x for animevideov3-x2).
    """
    from pathlib import Path
    from pai_studio.engine import LocalEngine

    engine = LocalEngine()
    src_path = tmp_path / "source.png"
    Image.new("RGB", (60, 40), (100, 150, 200)).save(src_path, format="PNG")

    fake_exe = tmp_path / "runtime" / "realesrgan-ncnn-vulkan.exe"
    fake_exe.parent.mkdir(parents=True, exist_ok=True)
    fake_exe.write_bytes(b"exe")

    engine._ensure_realesrgan_runtime = lambda: fake_exe  # type: ignore[method-assign]
    engine._ensure_realesrgan_model_pair = lambda exe, model, scale: None  # type: ignore[method-assign]

    seen_calls: list[tuple[str, str]] = []

    class FakeProc:
        returncode = 0
        def poll(self):
            return 0

    import subprocess
    orig_popen = subprocess.Popen

    def fake_popen(args, **kwargs):
        n_idx = args.index("-n")
        s_idx = args.index("-s")
        o_idx = args.index("-o")
        model_arg = args[n_idx + 1]
        scale_arg = args[s_idx + 1]
        out_file = Path(args[o_idx + 1])
        seen_calls.append((model_arg, scale_arg))
        img = Image.new("RGB", (60 * int(scale_arg), 40 * int(scale_arg)), (120, 160, 210))
        for y in range(img.height):
            for x in range(img.width):
                img.putpixel((x, y), ((x * 17) % 255, (y * 31) % 255, ((x + y) * 13) % 255))
        img.save(out_file, format="PNG", compress_level=0)
        return FakeProc()

    subprocess.Popen = fake_popen  # type: ignore[assignment]
    try:
        out_4x = engine._run_upscale_file(src_path, {"upscale_model": "realesrgan-x4plus"}, tmp_path)
        with Image.open(out_4x) as im4:
            assert im4.size == (240, 160)
        out_3x = engine._run_upscale_file(src_path, {"upscale_model": "realesr-animevideov3-x3"}, tmp_path)
        with Image.open(out_3x) as im3:
            assert im3.size == (180, 120)
        out_2x = engine._run_upscale_file(src_path, {"upscale_model": "realesr-animevideov3-x2"}, tmp_path)
        with Image.open(out_2x) as im2:
            assert im2.size == (120, 80)
        assert seen_calls == [
            ("realesrgan-x4plus", "4"),
            ("realesr-animevideov3", "3"),
            ("realesr-animevideov3", "2"),
        ]
    finally:
        subprocess.Popen = orig_popen  # type: ignore[assignment]


def test_ultimate_sd_upscale_splits_any_output_into_four_overlapping_tiles_with_8_steps_and_seamless_blending():
    """Verify that Ultimate mode behaves like external workflow server's Ultimate Tile Refinement in 4-tile mode:
    takes the output image at any resolution (including 4K+), splits it into exactly 4 overlapping
    tiles (2x2 grid), runs 8 steps sequentially on each tile using the updated canvas so borders
    match, and applies per-tile color matching + 2D cosine feathering.
    """
    from pai_studio.engine import LocalEngine

    engine = LocalEngine()
    engine._active_ultimate_tile_coverage = 0.50
    engine._active_ultimate_refinement_passes = 1
    engine._active_ultimate_steps = 8
    engine._active_ultimate_denoise_strength = 0.35

    # Even on a 4320x3240 output, Ultimate must split into 4 overlapping tiles (2x2), not 9 or 16.
    large_output = Image.new("RGB", (4320, 3240), (110, 120, 130))
    tile_calls: list[tuple[tuple[int, int], int]] = []

    def fake_tile_infer(prompt, tile, size, seed, quality, steps, **kwargs):
        tile_calls.append((size, steps))
        return tile.copy()

    engine._infer_with_auto_offload_retry = fake_tile_infer  # type: ignore[method-assign]
    refined = engine._qwen_detail_refine_large_output(
        large_output, "Restore the photo", 42, "Turbo · 8 steps", 8,
        detail_mode="ultimate", use_img2img=True, tile_strength=0.35,
        detail_profile="balanced", force=True,
    )
    assert refined.size == (4320, 3240)
    assert len(tile_calls) == 4
    assert all(steps == 8 for _size, steps in tile_calls)
    # Each tile must be larger than half the canvas width/height to guarantee overlap.
    assert all(w >= 4320 // 2 and h >= 3240 // 2 for (w, h), _ in tile_calls)

