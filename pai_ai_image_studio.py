#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P.AI AI IMAGE STUDIO · Version 1 Beta
Entrypoint launcher for the modularized `pai_studio` package.

Uses the official upstream stable-diffusion.cpp native runtime on loopback only;
no external workflow server service is needed. Qwen Image 2.1 handles generative enhancement/upscaling;
Real-ESRGAN and Real-ESRNet remain separate standalone Vulkan upscalers.

Run:
    python pai_ai_image_studio.py
"""
from __future__ import annotations

from pai_studio import *  # noqa: F401,F403
from pai_studio.constants import (  # noqa: F401
    _download_ssl_context,
    _human_gb,
    _realesrgan_model_assets_present,
    _sha256_file,
)
from pai_studio.grounding import (  # noqa: F401
    _QWEN3VL_GROUNDER,
    _QWEN3VL_GROUNDER_LOCK,
    _get_qwen3vl_grounder,
    _shutdown_qwen3vl_grounder,
)
from pai_studio.ui.main_window import App, main  # noqa: F401
from pai_studio.workspace import (  # noqa: F401
    _LOCAL_BBOX_CACHE,
    _LOCAL_BBOX_CACHE_LIMIT,
    _extract_llama_grounding_archive_safe,
    _model_role_mismatch,
    _relative_if_inside,
    _text_encoder_embeds_vision,
)

if __name__ == "__main__":
    main()
