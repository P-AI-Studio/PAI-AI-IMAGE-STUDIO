# -*- coding: utf-8 -*-
"""
Prompt construction, identity lock, and Qwen output dimension calculations.
"""
from __future__ import annotations

from typing import Optional

from .constants import (
    CLOSEUP_MANUAL_CROP_PROMPT,
    CLOSEUP_SEMANTIC_AUTO_PROMPT,
    FAITHFUL_ENHANCE_PROMPT,
    FIXED_SQUARE_OUTPUT_SIZES,
    MAX_OUTPUT_SIDE,
    SQUARE_ENHANCE_OUTPAINT_PROMPT,
    SQUARE_OUTPAINT_ONLY_PROMPT,
)

def build_edit_prompt(user_instruction: str, lock_identity: bool = True,
                      preserve_composition: bool = True) -> str:
    instruction = str(user_instruction or "").strip()
    if not instruction:
        return ""
    parts = [instruction.rstrip(" .") + "."]
    if lock_identity:
        parts.append("Preserve the main subject’s identity, defining features, proportions, and expression.")
    if preserve_composition:
        parts.append("Change only what is requested. Keep all unmentioned content, framing, perspective, lighting, and background unchanged.")
    return " ".join(p for p in parts if p.strip())

def build_closeup_prompt(target: str, custom_prompt: Optional[str] = None,
                         is_semantic_auto: bool = True, trigger_word: str = "") -> str:
    """Build one consistent target-aware prompt; optionally activate a selected LoRA."""
    clean_target = str(target or "the main subject detail").strip() or "the main subject detail"
    template = CLOSEUP_SEMANTIC_AUTO_PROMPT if is_semantic_auto else CLOSEUP_MANUAL_CROP_PROMPT
    prompt = template.format(target=clean_target)
    if is_semantic_auto:
        prompt += (
            " The geometric crop has already been made before this inference. Keep this crop and target placement; "
            "do not zoom back out, reframe, add margins, or recreate the full scene. Make the named target visually "
            "clear within this crop and enhance only source-supported edge clarity and fine texture. Preserve identity, "
            "geometry, anatomy, proportions, colors, lighting, and all visible content; do not invent missing details."
        )
    if trigger_word and trigger_word.strip():
        trigger = trigger_word.strip()
        if trigger.casefold() not in prompt.casefold():
            prompt = f"{trigger}, {prompt}"
    if custom_prompt and custom_prompt.strip():
        prompt += " Follow this additional user instruction where it does not conflict with the target-preservation requirements: " + custom_prompt.strip().rstrip(" .") + "."
    return prompt

def build_enhance_prompt(user_instruction: str = "") -> str:
    instruction = str(user_instruction or "").strip()
    if not instruction or instruction == FAITHFUL_ENHANCE_PROMPT:
        return FAITHFUL_ENHANCE_PROMPT
    edit_instruction = build_edit_prompt(instruction, lock_identity=True, preserve_composition=True)
    return (
        edit_instruction + "\n\nENHANCE RECONSTRUCTION: re-render the photo with source-conditioned Qwen image-to-image; "
        "do not simply upscale or copy the input. Reconstruct clean natural skin pores, delicate body vellus hair (peach fuzz), "
        "individual hair strands, tactile fabric weave/fuzz and environmental microtexture, crisp 3D object shape, "
        "suppress noise and compression artifacts first, and preserve identity, scene geometry, colors, "
        "lighting and framing. Only add detail supported by the source; do not invent facial features or objects."
    )


def build_square_core_enhance_prompt(user_instruction: str = "") -> str:
    """Build a source-core enhancement prompt without leaking square outpaint directions."""
    instruction = str(user_instruction or "").strip()
    if instruction in {SQUARE_OUTPAINT_ONLY_PROMPT, SQUARE_ENHANCE_OUTPAINT_PROMPT}:
        instruction = ""
    return build_enhance_prompt(instruction)

WATERMARK_REMOVAL_INSTRUCTION = (
    "WATERMARK REMOVAL TASK: Remove only visible watermark overlays, stock-photo stamps, "
    "semi-transparent watermark text, or author signatures that are placed on top of the image. "
    "Reconstruct the obscured pixels by continuing the immediately surrounding real texture, "
    "lighting, color, and perspective naturally. Keep the same image, subject identity, geometry, "
    "composition, and all non-watermark text, labels, logos on physical objects, and scene content. "
    "Do not crop, blur, smudge, or replace the whole image to hide a watermark."
)


def apply_watermark_removal_prompt(prompt: str, enabled: bool = False) -> str:
    """Add an explicit, narrowly scoped Qwen instruction only when watermark removal is enabled."""
    text = str(prompt or "").strip()
    if not enabled:
        return text
    if "WATERMARK REMOVAL TASK:" in text:
        return text
    return WATERMARK_REMOVAL_INSTRUCTION + ("\n\n" + text if text else "")


IDENTITY_LOCK_PROMPT = (
    "Strict identity/content lock: keep the same person or subject unmistakably unchanged. "
    "Preserve facial identity and geometry, expression, pose, anatomy, distinctive marks, object shape, "
    "readable text, colors, lighting, framing, and every unmentioned area. Do not replace, redesign, "
    "or invent subject details; make only the requested local improvement."
)


def apply_identity_lock(prompt: str, enabled: bool = True) -> str:
    """Append a stronger prompt constraint when the Workspace identity lock is enabled."""
    text = str(prompt or "").strip()
    if not enabled or IDENTITY_LOCK_PROMPT in text:
        return text
    return f"{text}\n\n{IDENTITY_LOCK_PROMPT}" if text else IDENTITY_LOCK_PROMPT


def build_square_prompt(user_instruction: str = "", enhance: bool = False) -> str:
    instruction = str(user_instruction or "").strip()
    if not instruction or instruction in (SQUARE_OUTPAINT_ONLY_PROMPT, SQUARE_ENHANCE_OUTPAINT_PROMPT):
        return SQUARE_ENHANCE_OUTPAINT_PROMPT if enhance else SQUARE_OUTPAINT_ONLY_PROMPT
    instruction = instruction.rstrip(" .") + "."
    if enhance:
        return (instruction + " Enhance the image conservatively, preserving subject identity, object geometry, readable text, colors, lighting, and composition. "
                "Extend the original image to a square canvas by generating content beyond its original edges, matching nearby perspective, lighting, and texture; do not create solid-color bars or frames. "
                "Do not introduce unrelated objects or restyle the central image.")
    return (instruction + " Extend the original image to a seamless square canvas by generating content beyond its original edges. Preserve existing content and readable text; never create solid-color bars or frames. "
            "Continue the immediate surroundings with consistent perspective, lighting, color, and texture. Do not add unrelated subjects, objects, or text.")

def snap_qwen_dimension_value(value: int, minimum: int = 64, maximum: int = MAX_OUTPUT_SIDE) -> int:
    """Normalize custom Qwen dimensions to a multiple of 32 within the supported UI range."""
    return max(minimum, min(maximum, int(round(int(value) / 32.0)) * 32))

def normalize_fixed_square_side(value: int) -> int:
    """Return the nearest supported fixed canvas size for Close-up and Square workflows."""
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 1024
    return min(FIXED_SQUARE_OUTPUT_SIZES, key=lambda side: abs(side - requested))

def calculate_qwen_output_size(source_size: tuple[int, int], workflow: str, mode: str,
                               scale: int, custom_width: int, custom_height: int) -> tuple[int, int]:
    """Resolve Qwen output dimensions; Real-ESRGAN has a separate output-size control.

    Enhance may use Scale or Custom dimensions. Close-up and Square accept only one
    of the fixed square sides; their live UI passes the selected side in custom_width.
    Each native inference pass is capped at MAX_INFERENCE_SIDE; larger Enhance outputs use
    Qwen-only overlapping detail-refinement tiles after the global composition pass.
    """
    workflow = str(workflow or "enhance")
    if workflow in ("lora_closeup_1024", "square_1_1"):
        side = normalize_fixed_square_side(custom_width)
        return side, side
    if str(mode) == "Custom dimensions":
        width = snap_qwen_dimension_value(custom_width)
        height = snap_qwen_dimension_value(custom_height)
        return width, height
    multiplier = max(1, min(4, int(scale)))
    source_w = max(1, int(source_size[0]))
    source_h = max(1, int(source_size[1]))
    if multiplier == 1:
        # Preserve the exact input pixel dimensions in the output at 1×.
        return source_w, source_h
    factor = min(float(multiplier), MAX_OUTPUT_SIDE / max(source_w, source_h))
    # A scale selector must mean an actual pixel multiplier (2× is exactly double,
    # rather than a nearby multiple of 32). Native Qwen inference dimensions are
    # normalized separately in LocalEngine._inference_size(); the final canvas and
    # all stitched tiles can retain the exact requested pixel dimensions.
    width = max(1, min(MAX_OUTPUT_SIDE, int(round(source_w * factor))))
    height = max(1, min(MAX_OUTPUT_SIDE, int(round(source_h * factor))))
    return width, height

