# -*- coding: utf-8 -*-
"""Regression tests for Int8 ConvRot text encoders, mmproj-optional Close-up grounding,
curated Hugging Face Qwen-Image 2.1 Turbo Uncensored catalog links, and 2x2 Ultimate SD Upscale."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw

from pai_studio.constants import MODEL_CATALOG, MODEL_DEFAULTS
from pai_studio.engine import LocalEngine
from pai_studio.grounding import (
    crop_closeup_from_bbox,
    detect_target_bbox_local,
    locate_target_bbox_non_gguf,
    resolve_closeup_grounding_gguf_pair,
)
from pai_studio.workspace import _model_role_mismatch, _text_encoder_embeds_vision


def test_convrot_end_tag_recognition_in_role_guard_and_embedded_vision():
    # Text encoders ending in convrot
    for encoder_name in (
        "qwen3vl_8b_int8_convrot.safetensors",
        "qwen3vl_8b_int8_convrot_heretic.safetensors",
        "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
        "custom_te_int8_convrot.safetensors",
    ):
        assert _model_role_mismatch("text_encoder", encoder_name) == ""
        assert _text_encoder_embeds_vision(encoder_name) is True
        assert "text encoder" in _model_role_mismatch("diffusion", encoder_name).lower()

    # Diffusion weights ending in convrot
    for diff_name in (
        "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors",
        "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "rayQwen21_turboV13Int8convrot.safetensors",
    ):
        assert _model_role_mismatch("diffusion", diff_name) == ""
        assert "diffusion" in _model_role_mismatch("text_encoder", diff_name).lower()


def test_closeup_target_grounding_succeeds_for_int8_convrot_and_without_mmproj(tmp_path: Path):
    image = Image.new("RGB", (640, 960), (28, 32, 38))
    draw = ImageDraw.Draw(image)
    # Draw a bright subject in the center with warm skin-like and high-contrast features
    draw.rectangle((220, 140, 420, 820), fill=(212, 168, 145))
    draw.ellipse((260, 160, 380, 290), fill=(230, 185, 162))
    draw.rectangle((250, 620, 390, 740), fill=(198, 142, 126))

    convrot_te = tmp_path / "qwen3vl_8b_int8_convrot_heretic.safetensors"
    convrot_te.write_bytes(b"safetensors-stub")

    # 1. Non-GGUF Int8 ConvRot encoder with mmproj=None
    engine = LocalEngine()
    res_convrot = engine._locate_closeup_target(
        image, "face", model_path=convrot_te, mmproj_path=None, profile="fast"
    )
    assert res_convrot.status == "found"
    assert res_convrot.bbox_xyxy is not None
    x1, y1, x2, y2 = res_convrot.bbox_xyxy
    assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0

    # 2. GGUF encoder WITHOUT mmproj (mmproj_path=None)
    gguf_te = tmp_path / "Sakura-TextEncoder-Qwen3VL-8B-Uncensored-8bit-8.11GiB.gguf"
    gguf_te.write_bytes(b"gguf-stub")
    res_gguf_no_mmproj = engine._locate_closeup_target(
        image, "pussy", model_path=gguf_te, mmproj_path=None, profile="fast"
    )
    assert res_gguf_no_mmproj.status == "found"
    assert res_gguf_no_mmproj.bbox_xyxy is not None


def test_find_model_parts_does_not_require_mmproj_even_if_stale_selection_exists(tmp_path: Path):
    diff = tmp_path / "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors"
    vae = tmp_path / "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
    te = tmp_path / "qwen3vl_8b_int8_convrot_heretic.safetensors"
    for f in (diff, vae, te):
        f.write_bytes(b"stub")

    # Even if model-selection.json references a missing mmproj file on disk, _find_model_parts must succeed
    (tmp_path / "model-selection.json").write_text(
        json.dumps({
            "diffusion": {"filename": diff.name},
            "vae": {"filename": vae.name},
            "text_encoder": {"filename": te.name},
            "vision_projector": {"filename": "missing-mmproj.gguf"},
        }),
        encoding="utf-8",
    )
    engine = LocalEngine()
    parts = engine._find_model_parts(tmp_path)
    assert parts["diffusion"] == diff.resolve()
    assert parts["vae"] == vae.resolve()
    assert parts["llm"] == te.resolve()
    assert parts["llm_vision"] is None


def test_model_catalog_includes_curated_uncensored_and_convrot_publishers():
    assert MODEL_DEFAULTS["vision_projector"] == "none"
    diffusion_urls = [item.get("url", "") for item in MODEL_CATALOG if item.get("role") == "diffusion"]
    encoder_urls = [item.get("url", "") for item in MODEL_CATALOG if item.get("role") == "text_encoder"]
    vae_urls = [item.get("url", "") for item in MODEL_CATALOG if item.get("role") == "vae"]

    assert any("unsloth/Qwen-Image-2.1-Turbo-FP8" in u and "ConvRot" in u for u in diffusion_urls)
    assert any("Comfy-Org/Qwen-Image-2.1" in u and "convrot" in u for u in diffusion_urls)
    assert any("webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF" in u for u in diffusion_urls)
    assert any("Nic3301/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot" in u for u in encoder_urls)
    assert any("Comfy-Org/Qwen-Image-2.1" in u and "qwen3vl_8b_int8_convrot.safetensors" in u for u in encoder_urls)
    assert any("webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF" in u for u in encoder_urls)
    assert any("madebyollin/texture-fix-vae-for-qwen-image-2.1" in u for u in vae_urls)


def test_ultimate_always_covers_full_canvas_with_exact_four_quadrants():
    engine = LocalEngine()
    for coverage in (0.45, 0.50, 0.75):
        engine._active_ultimate_tile_coverage = coverage
        engine._active_ultimate_refinement_passes = 1
        engine._active_ultimate_steps = 8
        engine._active_ultimate_denoise_strength = 0.35
        crops = []

        def fake_infer(prompt, source, size, seed, quality, steps=8, **kwargs):
            crops.append(size)
            return Image.new("RGB", size, (120, 140, 160))

        engine._infer_with_auto_offload_retry = fake_infer
        canvas = Image.new("RGB", (2048, 2048), (80, 90, 100))
        out = engine._qwen_detail_refine_large_output(
            canvas, "enhance prompt", 42, "Turbo · 8 steps", 8,
            detail_mode="ultimate", use_img2img=True, tile_strength=0.35,
            detail_profile="balanced", force=True,
        )
        assert out.size == (2048, 2048)
        assert len(crops) == 4
        # Bottom-right corner must have been refined by the 4th quadrant
        assert out.getpixel((2047, 2047)) != (80, 90, 100)


def test_resolve_closeup_grounding_gguf_pair_discovers_installed_gguf_and_mmproj(tmp_path: Path):
    convrot_te = tmp_path / "qwen3vl_8b_int8_convrot_heretic.safetensors"
    gguf_te = tmp_path / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    mmproj = tmp_path / "mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf"
    convrot_te.write_bytes(b"safetensors-stub")
    gguf_te.write_bytes(b"gguf-te-stub")
    mmproj.write_bytes(b"gguf-mmproj-stub")

    resolved_model, resolved_mmproj = resolve_closeup_grounding_gguf_pair(convrot_te, None)
    assert resolved_model == gguf_te.resolve()
    assert resolved_mmproj == mmproj.resolve()


def test_non_gguf_grounding_tight_box_honors_closeup_zoom_factor():
    image = Image.new("RGB", (900, 1200), (24, 28, 34))
    draw = ImageDraw.Draw(image)
    # Central skin-toned figure from x=320..580, y=150..1050
    draw.rectangle((320, 150, 580, 1050), fill=(218, 172, 148))
    # Distinct anatomical high-contrast features at head (y~210) and lower torso/pelvis (y~760)
    draw.ellipse((390, 170, 510, 270), fill=(238, 192, 168), outline=(90, 55, 45), width=4)
    draw.polygon([(420, 730), (480, 730), (450, 795)], fill=(165, 98, 88), outline=(75, 40, 35))

    face_res = locate_target_bbox_non_gguf(image, "face")
    pelvis_res = locate_target_bbox_non_gguf(image, "pussy")
    assert face_res.status == "found" and face_res.bbox_xyxy is not None
    assert pelvis_res.status == "found" and pelvis_res.bbox_xyxy is not None

    fx1, fy1, fx2, fy2 = face_res.bbox_xyxy
    px1, py1, px2, py2 = pelvis_res.bbox_xyxy
    # Face center should be in upper region, pelvis center in lower region
    assert 0.10 <= (fy1 + fy2) * 0.5 <= 0.35
    assert 0.52 <= (py1 + py2) * 0.5 <= 0.78
    assert 0.35 <= (px1 + px2) * 0.5 <= 0.65

    # Tight bbox ensures requested zoom=2.30x is honored without clamping down
    _, crop_box, eff_zoom = crop_closeup_from_bbox(image, 1024, pelvis_res.bbox_xyxy, zoom_factor=2.30)
    assert abs(eff_zoom - 2.30) < 0.08


def test_source_workflows_auto_switch_to_side_by_side_and_locator_is_in_model_catalog():
    root = Path(__file__).resolve().parent.parent / "pai_studio" / "ui"
    mw_src = (root / "main_window.py").read_text(encoding="utf-8")
    dlg_src = (root / "dialogs.py").read_text(encoding="utf-8")
    canvas_src = (root / "canvas.py").read_text(encoding="utf-8")

    # 1. Verify Locator Runtime card is hosted in ModelDownloadDialog (Model Catalog), not lora_group
    assert "self.grounding_runtime_group = QGroupBox(" in mw_src
    assert "lora_layout.addWidget(self.btn_qwen_grounding_setup)" not in mw_src
    assert "getattr(app_parent, \"grounding_runtime_group\", None)" in dlg_src
    assert "addWidget(grounding_card)" in dlg_src

    # 2. Verify all 5 source workflows switch to Side by side automatically on execution
    assert 'if job.workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:' in mw_src
    assert 'if active_workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:' in mw_src
    assert 'self.preview_mode.blockSignals(True)' in mw_src

    # 3. Verify PreviewCanvas supports Side by side with single original image and maps closeup_crop_box focus points
    assert 'self.mode == "Side by side" and (self.original is not None or self.enhanced is not None)' in canvas_src
    assert "self.closeup_crop_box" in canvas_src


def test_prompt_translation_works_across_generate_edit_and_closeup_even_without_mmproj(monkeypatch, tmp_path: Path):
    from pai_studio.engine import Job

    encoder = tmp_path / "Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf"
    encoder.write_bytes(b"gguf-stub")

    engine = LocalEngine()
    engine.model_path = str(tmp_path)
    engine._find_model_parts = lambda _folder: {
        "diffusion": tmp_path / "diff.gguf",
        "vae": tmp_path / "vae.safetensors",
        "llm": encoder,
        "llm_vision": None,  # No projector selected/downloaded
    }

    seen_translations = []

    def fake_translate(text, **kwargs):
        seen_translations.append((text, kwargs.get("model_path"), kwargs.get("mmproj_path")))
        return f"english({text})"

    monkeypatch.setattr("pai_studio.engine.translate_prompt_via_local_encoder", fake_translate)

    # 1. Image Generate
    gen_en = engine._translate_image_prompt_to_english("یک دختر زیبا در ساحل", "image_generate")
    assert gen_en == "english(یک دختر زیبا در ساحل)"

    # 2. Image Edit (with build_edit_prompt lock suffixes preserved)
    from pai_studio.prompts import build_edit_prompt
    edit_raw = build_edit_prompt("لباس را قرمز کن", lock_identity=True, preserve_composition=True)
    edit_en = engine._translate_image_prompt_to_english(edit_raw, "image_edit")
    assert edit_en.startswith("english(لباس را قرمز کن.).")
    assert "Preserve the main subject" in edit_en

    # 3. Close-up target & prompt
    cl_target_en = engine._translate_image_prompt_to_english("چشم چپ", "lora_closeup_1024")
    assert cl_target_en == "english(چشم چپ)"
    assert all(m == encoder and v is None for _, m, v in seen_translations)


def test_two_stage_1024_then_ultimate_2048_across_closeup_enhance_generate_edit_and_square(tmp_path: Path):
    from pai_studio.engine import Job

    src_path = tmp_path / "input.png"
    src_img = Image.new("RGB", (900, 1200), (100, 120, 140))
    draw = ImageDraw.Draw(src_img)
    draw.rectangle((300, 150, 600, 1050), fill=(215, 170, 145))
    draw.ellipse((380, 180, 520, 300), fill=(235, 190, 165))
    src_img.save(src_path)

    convrot_te = tmp_path / "qwen3vl_8b_int8_convrot.safetensors"
    convrot_te.write_bytes(b"stub")

    for wf in ("lora_closeup_1024", "enhance", "image_generate", "image_edit", "square_1_1"):
        engine = LocalEngine()
        engine.model_path = str(tmp_path)
        engine._find_model_parts = lambda _folder: {
            "diffusion": tmp_path / "diff.safetensors",
            "vae": tmp_path / "vae.safetensors",
            "llm": convrot_te,
            "llm_vision": None,
        }
        infer_sizes = []

        def fake_infer(prompt, source, size, seed, quality, steps=8, **kwargs):
            infer_sizes.append(size)
            return Image.new("RGB", size, (130, 145, 160))

        engine._infer_with_auto_offload_retry = fake_infer
        job = Job(
            workflow=wf,
            input_path=str(src_path) if wf != "image_generate" else None,
            prompt="یک پرتره زیبا" if wf in ("image_generate", "image_edit") else "",
            target="صورت" if wf == "lora_closeup_1024" else "",
            output_dir=str(tmp_path / wf),
            width=2048,
            height=2048,
            side=2048,
            enhance=True,
            auto_zoom=True,
            custom={"num_inference_steps": 8, "qwen_detail_profile": "balanced", "ultimate_qwen_upscale_enabled": True, "ultimate_qwen_scale": 2},
        )
        outs = engine.process(job)
        assert len(outs) == 1
        with Image.open(outs[0]) as out_im:
            assert out_im.size == (2048, 2048)
        # First pass (Stage 1) MUST be 1024x1024, followed by 4 Ultimate tiles!
        assert infer_sizes[0] == (1024, 1024), f"{wf} first pass was {infer_sizes[0]}, expected (1024, 1024)"
        assert len(infer_sizes) >= 5, f"{wf} expected 1 global pass + 4 Ultimate tiles, got {len(infer_sizes)}"


def test_ultimate_denoise_strength_spinner_directly_controls_tile_denoise_up_and_down():
    """Verify raising or lowering Denoise strength (ultimate_qwen_denoise_strength) directly changes tile img2img strength."""
    engine = LocalEngine()
    engine._active_ultimate_tile_coverage = 0.50
    engine._active_ultimate_refinement_passes = 1
    engine._active_ultimate_steps = 8

    for user_denoise in (0.10, 0.22, 0.45, 0.68):
        engine._active_ultimate_denoise_strength = user_denoise
        recorded_strengths = []

        def fake_infer(prompt, source, size, seed, quality, steps=8, **kwargs):
            recorded_strengths.append(kwargs.get("strength"))
            return source.copy()

        engine._infer_with_auto_offload_retry = fake_infer
        canvas = Image.new("RGB", (1024, 1024), (90, 100, 110))
        out = engine._qwen_detail_refine_large_output(
            canvas, "enhance prompt", 42, "Turbo · 8 steps", 8,
            detail_mode="ultimate", use_img2img=True, tile_strength=0.74,
            detail_profile="maximum", force=True,
        )
        assert out.size == (1024, 1024)
        assert len(recorded_strengths) == 4
        assert all(abs(s - user_denoise) < 1e-6 for s in recorded_strengths)


def test_model_catalog_text_encoders_include_quantized_qwen3vl_ladder_and_exclude_atomic_image_models():
    """Verify text_encoder role contains real Qwen3-VL-8B quantized encoders (Q2_K..Q8_0, F16, ConvRot) and not AtomicChat image models."""
    encoder_items = [item for item in MODEL_CATALOG if item.get("role") == "text_encoder"]
    encoder_urls = [item.get("url", "") for item in encoder_items]
    encoder_filenames = [item.get("filename", "") for item in encoder_items]

    # AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF entries are moved out of text_encoder into diffusion
    assert not any("AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF" in u for u in encoder_urls)
    diffusion_urls = [item.get("url", "") for item in MODEL_CATALOG if item.get("role") == "diffusion"]
    assert any("AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF" in u for u in diffusion_urls)

    # Full quantization ladder of Qwen3-VL-8B text encoders is present
    for quant_tag in ("Q2_K", "Q3_K_M", "Q3_K_L", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "f16", "convrot"):
        assert any(quant_tag.lower() in fn.lower() for fn in encoder_filenames), f"Missing {quant_tag} in text_encoder catalog"


def test_resolve_closeup_grounding_gguf_pair_ignores_diffusion_gguf_and_persian_target_grounds_accurately(tmp_path: Path):
    """Ensure diffusion GGUF files in the model folder are never mistaken for GGUF text encoders during Close-up grounding."""
    convrot_te = tmp_path / "qwen3vl_8b_int8_convrot_heretic.safetensors"
    diff_gguf_1 = tmp_path / "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf"
    diff_gguf_2 = tmp_path / "Sakura-Image-2.1-Turbo-Q8_0-7.07GiB.gguf"
    mmproj = tmp_path / "mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf"
    for f in (convrot_te, diff_gguf_1, diff_gguf_2, mmproj):
        f.write_bytes(b"stub")

    resolved_model, _ = resolve_closeup_grounding_gguf_pair(convrot_te, None)
    assert resolved_model is None  # Must NOT pick diffusion GGUF!

    # Also verify Persian target terms ("صورت", "چشم", "کس") locate the right anatomical zones in non-GGUF grounding
    image = Image.new("RGB", (900, 1200), (24, 28, 34))
    draw = ImageDraw.Draw(image)
    draw.rectangle((320, 150, 580, 1050), fill=(218, 172, 148))
    draw.ellipse((390, 170, 510, 270), fill=(238, 192, 168), outline=(90, 55, 45), width=4)
    draw.polygon([(420, 730), (480, 730), (450, 795)], fill=(165, 98, 88), outline=(75, 40, 35))

    face_fa = locate_target_bbox_non_gguf(image, "صورت")
    groin_fa = locate_target_bbox_non_gguf(image, "کس")
    assert face_fa.status == "found" and face_fa.bbox_xyxy is not None
    assert groin_fa.status == "found" and groin_fa.bbox_xyxy is not None
    assert 0.10 <= (face_fa.bbox_xyxy[1] + face_fa.bbox_xyxy[3]) * 0.5 <= 0.35
    assert 0.52 <= (groin_fa.bbox_xyxy[1] + groin_fa.bbox_xyxy[3]) * 0.5 <= 0.78


def test_ultimate_half_res_lanczos_pipeline_when_on_and_disabled_when_off(tmp_path: Path):
    """Verify:
    1) When Ultimate is ON with scale 2x:
       - 1024 -> Pass 1 at 512x512 -> Lanczos 2x to 1024x1024 -> 4 Ultimate tiles
       - 2048 -> Pass 1 at 1024x1024 -> Lanczos 2x to 2048x2048 -> 4 Ultimate tiles
       - 4096 -> Pass 1 at 2048x2048 -> Lanczos 2x to 4096x4096 -> 4 Ultimate tiles
    2) When Ultimate is OFF (even with ultimate_qwen_scale=2 in custom):
       - Runs directly at requested resolution without half-res, Lanczos, or Ultimate tiles.
    """
    from pai_studio.engine import Job, qwen_turbo_tile_sampling_params

    # Verify img2img step compensation prevents 3-step truncation in sd-server
    for user_steps in (4, 8, 12, 16):
        for user_denoise in (0.20, 0.25, 0.35, 0.50):
            params = qwen_turbo_tile_sampling_params(steps=user_steps, strength=user_denoise, compensate_img2img_steps=True)
            sched_steps = params["sample_steps"]
            assert int(sched_steps * user_denoise) == user_steps
            assert len(params["custom_sigmas"]) == sched_steps + 1

    # Verify Tile Coverage directly controls tile crop dimensions
    engine = LocalEngine()
    for cov, expected_side in ((0.50, 1024), (0.60, 1229), (0.75, 1536)):
        engine._active_ultimate_tile_coverage = cov
        engine._active_ultimate_denoise_strength = 0.25
        engine._active_ultimate_steps = 12
        engine._active_ultimate_refinement_passes = 1
        seen_tiles = []

        def fake_tile_infer(prompt, source, size, seed, quality, steps=8, **kwargs):
            seen_tiles.append((size, steps, kwargs.get("strength"), kwargs.get("compensate_img2img_steps"), kwargs.get("skip_ref_images")))
            return Image.new("RGB", size, (120, 130, 140))

        engine._infer_with_auto_offload_retry = fake_tile_infer
        canvas = Image.new("RGB", (2048, 2048), (90, 100, 110))
        out = engine._qwen_detail_refine_large_output(
            canvas, "test", 1, "Turbo · 8 steps", 8, detail_mode="ultimate", use_img2img=True, force=True
        )
        assert out.size == (2048, 2048)
        assert len(seen_tiles) == 4
        assert all(sz == (expected_side, expected_side) for sz, _, _, _, _ in seen_tiles)
        assert all(st == 12 and abs(s - 0.25) < 1e-6 and comp is True and skip is False for _, st, s, comp, skip in seen_tiles)

    src_path = tmp_path / "src.png"
    src_img = Image.new("RGB", (900, 1200), (100, 120, 140))
    draw = ImageDraw.Draw(src_img)
    draw.rectangle((300, 150, 600, 1050), fill=(215, 170, 145))
    draw.ellipse((380, 180, 520, 300), fill=(235, 190, 165))
    src_img.save(src_path)

    convrot_te = tmp_path / "qwen3vl_8b_int8_convrot.safetensors"
    convrot_te.write_bytes(b"stub")

    # 1. Ultimate ON: 1024 -> 512, 2048 -> 1024, 4096 -> 2048
    for target_res, expected_pass1 in ((1024, 512), (2048, 1024), (4096, 2048)):
        eng = LocalEngine()
        eng.model_path = str(tmp_path)
        eng._find_model_parts = lambda _f: {"diffusion": tmp_path / "d.safetensors", "vae": tmp_path / "v.safetensors", "llm": convrot_te, "llm_vision": None}
        calls = []

        def fake_inf(prompt, source, size, seed, quality, steps=8, **kwargs):
            calls.append((size, steps, kwargs.get("strength")))
            return Image.new("RGB", size, (130, 140, 150))

        eng._infer_with_auto_offload_retry = fake_inf
        job = Job(
            workflow="lora_closeup_1024",
            input_path=str(src_path),
            target="face",
            output_dir=str(tmp_path / f"on_{target_res}"),
            width=target_res, height=target_res, side=target_res,
            enhance=True, auto_zoom=True,
            custom={
                "num_inference_steps": 8,
                "qwen_detail_profile": "maximum",
                "ultimate_qwen_upscale_enabled": True,
                "ultimate_qwen_scale": 2,
                "ultimate_qwen_denoise_strength": 0.25,
                "ultimate_qwen_tile_coverage": 0.50,
                "ultimate_qwen_steps": 10,
                "ultimate_qwen_passes": 1,
            },
        )
        outs = eng.process(job)
        with Image.open(outs[0]) as im:
            assert im.size == (target_res, target_res)
        # Pass 1 is half resolution (expected_pass1), followed by 4 tiles of 50% coverage (expected_pass1)
        assert calls[0][0] == (expected_pass1, expected_pass1)
        assert len(calls) == 5
        for tile_call in calls[1:]:
            assert tile_call[0] == (expected_pass1, expected_pass1)
            assert tile_call[1] == 10
            assert abs(tile_call[2] - 0.25) < 1e-6

    # 2. Ultimate OFF: 1024 runs directly at 1024 with NO tiles and NO Lanczos half-res
    eng_off = LocalEngine()
    eng_off.model_path = str(tmp_path)
    eng_off._find_model_parts = lambda _f: {"diffusion": tmp_path / "d.safetensors", "vae": tmp_path / "v.safetensors", "llm": convrot_te, "llm_vision": None}
    off_calls = []

    def fake_inf_off(prompt, source, size, seed, quality, steps=8, **kwargs):
        off_calls.append(size)
        return Image.new("RGB", size, (130, 140, 150))

    eng_off._infer_with_auto_offload_retry = fake_inf_off
    job_off = Job(
        workflow="lora_closeup_1024",
        input_path=str(src_path),
        target="face",
        output_dir=str(tmp_path / "off_1024"),
        width=1024, height=1024, side=1024,
        enhance=True, auto_zoom=True,
        custom={
            "num_inference_steps": 8,
            "qwen_detail_profile": "maximum",
            "ultimate_qwen_upscale_enabled": False,
            "ultimate_qwen_scale": 2,
        },
    )
    outs_off = eng_off.process(job_off)
    with Image.open(outs_off[0]) as im_off:
        assert im_off.size == (1024, 1024)
    assert off_calls == [(1024, 1024)]


def test_calibrated_tile_sigma_tail_and_ultimate_grain_removal():
    """Verify:
    1) qwen_turbo_tile_sampling_params strictly interpolates along the official distilled QWEN_TURBO_SIGMAS
       manifold while guaranteeing exact target steps (int(schedule_steps * s) == 8).
    2) Low denoise strengths (0.05, 0.15, 0.35) proportionally blend the refined tile with the crisp
       source tile without running destructive pre/post blur filters on the canvas.
    3) Reference conditioning (skip_ref_images=False) is preserved on every Ultimate tile.
    """
    import numpy as np
    from PIL import Image
    from pai_studio.engine import QWEN_TURBO_SIGMAS, qwen_turbo_tile_sampling_params
    from pai_studio.image_ops import clean_iso_grain_and_chroma_noise, crossfade_tile_seams

    from pai_studio.image_ops import blend_ultimate_tile_by_denoise
    for s in (0.05, 0.10, 0.15, 0.20, 0.25):
        params = qwen_turbo_tile_sampling_params(steps=8, strength=s, compensate_img2img_steps=True)
        sched_steps = params["sample_steps"]
        sigmas = params["custom_sigmas"]
        t_enc = int(sched_steps * s)
        assert t_enc == 8
        assert sigmas[0] == float(QWEN_TURBO_SIGMAS[0])
        # Terminal jump must be locked to 0.414568 -> 0.0 (never stepping inside (0, 0.414568))
        assert abs(sigmas[-2] - float(QWEN_TURBO_SIGMAS[-2])) < 1e-6
        assert sigmas[-1] == 0.0
        assert all(sig >= float(QWEN_TURBO_SIGMAS[-2]) for sig in sigmas[:-1])
        assert all(sigmas[i] >= sigmas[i + 1] for i in range(len(sigmas) - 1))

    # Verify blend_ultimate_tile_by_denoise scales monotonically with denoise_strength
    base_tile = Image.new("RGB", (64, 64), (100, 100, 100))
    ref_tile = Image.new("RGB", (64, 64), (200, 200, 200))
    b05 = blend_ultimate_tile_by_denoise(base_tile, ref_tile, 0.05).getpixel((32, 32))[0]
    b25 = blend_ultimate_tile_by_denoise(base_tile, ref_tile, 0.25).getpixel((32, 32))[0]
    b55 = blend_ultimate_tile_by_denoise(base_tile, ref_tile, 0.55).getpixel((32, 32))[0]
    assert 100 < b05 <= b25 < b55 <= 200

    # Verify grain removal reduces high-frequency variance on noisy flat surface
    rng = np.random.default_rng(42)
    noisy_arr = np.clip(128.0 + rng.normal(0.0, 14.0, (256, 256, 3)), 0.0, 255.0).astype(np.uint8)
    noisy_im = Image.fromarray(noisy_arr, mode="RGB")
    cleaned_im = clean_iso_grain_and_chroma_noise(noisy_im, strength=0.85)
    std_before = float(np.asarray(noisy_im, dtype=np.float32).std())
    std_after = float(np.asarray(cleaned_im, dtype=np.float32).std())
    assert std_after < std_before * 0.50, f"Expected >50% grain reduction, got {std_before:.2f} -> {std_after:.2f}"

    # Verify crossfade_tile_seams smooths a hard 2x2 center boundary at 50% coverage
    quad = np.full((256, 256, 3), 100, dtype=np.uint8)
    quad[:, 128:, :] = 130
    quad_im = Image.fromarray(quad, mode="RGB")
    smoothed = crossfade_tile_seams(quad_im, [0, 128], [0, 128], 128, 128)
    sm_arr = np.asarray(smoothed, dtype=np.float32)
    jump_before = float(abs(int(quad[64, 128, 0]) - int(quad[64, 127, 0])))
    jump_after = float(abs(sm_arr[64, 128, 0] - sm_arr[64, 127, 0]))
    assert jump_after < jump_before * 0.5
