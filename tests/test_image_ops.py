from PIL import Image

from pai_studio.constants import (
    MODEL_CATALOG_BY_ID,
    RESTORE_OLD_PRINT_PROMPT,
    RESTORE_PHOTO_PROMPT,
    realesrgan_native_scale,
)
from pai_studio.image_ops import (
    composite_square_outpaint_result,
    match_tile_color_to_reference,
)


def test_square_outpaint_seam_feather_uses_requested_width_beyond_eight_pixels():
    source = Image.new("RGB", (64, 38), (255, 255, 255))
    generated = Image.new("RGB", (64, 64), (0, 0, 0))

    result = composite_square_outpaint_result(
        source, generated, 64, (0, 13, 64, 51), enhance=False, feather=16
    )

    # At the seam generated pixels dominate; deeper in the source rectangle the original wins.
    assert result.getpixel((32, 13))[0] == 0
    assert 40 <= result.getpixel((32, 17))[0] <= 100
    assert result.getpixel((32, 30))[0] == 255


def test_square_outpaint_enhance_blend_fades_original_near_seam_only():
    source = Image.new("RGB", (64, 38), (255, 255, 255))
    generated = Image.new("RGB", (64, 64), (0, 0, 0))

    result = composite_square_outpaint_result(
        source, generated, 64, (0, 13, 64, 51), enhance=True, feather=16
    )

    assert result.getpixel((32, 13))[0] == 255
    assert 150 <= result.getpixel((32, 17))[0] <= 220
    assert result.getpixel((32, 30))[0] == 0


def test_match_tile_color_to_reference_aligns_channel_means_without_losing_detail():
    reference = Image.new("RGB", (32, 32), (100, 120, 140))
    shifted = Image.new("RGB", (32, 32), (140, 90, 180))
    matched = match_tile_color_to_reference(shifted, reference, strength=1.0)
    assert matched.size == (32, 32)
    r, g, b = matched.getpixel((16, 16))
    assert abs(r - 100) <= 1
    assert abs(g - 120) <= 1
    assert abs(b - 140) <= 1


def test_realesrgan_native_scale_and_texture_fix_vae_catalog_entries():
    assert realesrgan_native_scale("realesrgan-x4plus") == 4
    assert realesrgan_native_scale("realesrnet-x4plus") == 4
    assert realesrgan_native_scale("realesrgan-x4plus-anime") == 4
    assert realesrgan_native_scale("realesr-animevideov3") == 4
    assert realesrgan_native_scale("realesr-animevideov3-x3") == 3
    assert realesrgan_native_scale("realesr-animevideov3-x2") == 2
    assert "madebyollin_texture_fix_vae_bf16" in MODEL_CATALOG_BY_ID
    vae_meta = MODEL_CATALOG_BY_ID["madebyollin_texture_fix_vae_bf16"]
    assert vae_meta["role"] == "vae"
    assert vae_meta["filename"] == "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
    assert "Restore the photo:" in RESTORE_PHOTO_PROMPT
    assert "Restore the old photo:" in RESTORE_OLD_PRINT_PROMPT
