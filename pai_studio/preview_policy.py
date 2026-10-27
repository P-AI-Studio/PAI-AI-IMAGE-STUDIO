"""Pure workflow-to-preview presentation policy (kept independent of Qt)."""
from __future__ import annotations


_SOURCE_WORKFLOWS = frozenset({
    "image_edit",
    "enhance",
    "upscale_only",
    "lora_closeup_1024",
    "square_1_1",
})


def preview_presentation_for_workflow(workflow: str) -> str:
    """Return selectable source comparison views, or live-only for generation.

    Image Generate deliberately never displays a Before pane. Source workflows
    preserve both source/result images but let the user choose their presentation.
    """
    key = str(workflow or "").strip().lower()
    return "selectable" if key in _SOURCE_WORKFLOWS else "live_only"


def live_preview_images_for_workflow(workflow: str, source_image, live_image):
    """Return canvas images for a live frame without losing source-based comparisons."""
    if preview_presentation_for_workflow(workflow) == "selectable":
        return source_image, live_image
    return None, live_image


def preview_mode_for_workflow(workflow: str) -> str:
    """Return a default mode, not a forced mode, for the selected workflow."""
    return "Side by side" if preview_presentation_for_workflow(workflow) == "selectable" else "Enhanced"
