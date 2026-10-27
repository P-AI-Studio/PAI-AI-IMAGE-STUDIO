"""Pure preview-layout policy and persistence migration tests."""
from __future__ import annotations

from pai_studio.constants import DEFAULTS, SETTINGS_VERSION
from pai_studio.preview_policy import preview_presentation_for_workflow
from pai_studio.validation import validate_and_migrate_settings


def test_source_workflows_allow_user_selected_view_but_generate_is_live_only():
    assert preview_presentation_for_workflow("image_generate") == "live_only"
    for workflow in ("image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"):
        assert preview_presentation_for_workflow(workflow) == "selectable"
    assert preview_presentation_for_workflow("unknown") == "live_only"


def test_live_preview_defaults_to_tae_for_new_settings():
    assert DEFAULTS["live_preview_method"] == "tae"
    assert SETTINGS_VERSION == 44


def test_previous_settings_migrate_proj_to_tae_once():
    migrated = validate_and_migrate_settings({"settings_version": 40, "live_preview_method": "proj"})
    assert migrated["live_preview_method"] == "tae"
    assert migrated["settings_version"] == 44



def test_v43_settings_migrate_to_selectable_preview_default_without_changing_global_tile_floor():
    migrated = validate_and_migrate_settings({
        "settings_version": 43,
        "live_preview_method": "tae",
        "ultimate_qwen_denoise_strength": 0.20,
    })
    assert migrated["settings_version"] == 44
    assert migrated["preview_mode"] == "Side by side"
    assert migrated["ultimate_qwen_denoise_strength"] == 0.20

def test_current_explicit_preview_method_is_preserved():
    for method in ("proj", "tae", "vae", "none"):
        migrated = validate_and_migrate_settings({"settings_version": 44, "live_preview_method": method})
        assert migrated["live_preview_method"] == method


def test_live_frame_updates_keep_source_for_comparison_workflows():
    from pai_studio.preview_policy import live_preview_images_for_workflow

    source, live = object(), object()
    assert live_preview_images_for_workflow("image_edit", source, live) == (source, live)
    assert live_preview_images_for_workflow("enhance", source, live) == (source, live)
    assert live_preview_images_for_workflow("image_generate", source, live) == (None, live)


def test_source_preview_policy_is_independent_of_live_frame_availability():
    from pai_studio.preview_policy import preview_presentation_for_workflow

    # Source/result images remain available in every selectable mode, even before
    # the first Live frame arrives.
    assert preview_presentation_for_workflow("enhance") == "selectable"
    assert preview_presentation_for_workflow("lora_closeup_1024") == "selectable"


def test_ui_wires_workflow_policy_and_keeps_original_during_live_refresh():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "pai_studio" / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'policy = preview_presentation_for_workflow(workflow)' in source
    assert 'self.preview.set_presentation_policy(policy)' in source
    assert "live_preview_images_for_workflow(active_workflow, self.current_original, preview)" in source
    assert "self.preview_mode_label.setVisible(not live_only)" in source


def test_canvas_has_explicit_live_only_placeholder_and_selectable_source_policy():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "pai_studio" / "ui" / "canvas.py").read_text(encoding="utf-8")
    assert 'def set_presentation_policy(self, policy: str):' in source
    assert 'if self.presentation_policy == "before_live":' in source
    assert 'if self.presentation_policy == "live_only":' in source
    assert '"LIVE · PREVIEW / RESULT"' in source


def test_local_diagnostics_button_is_in_settings_not_closeup_basic():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "pai_studio" / "ui" / "main_window.py").read_text(encoding="utf-8")
    settings_start = source.index('self.settings_dialog.setWindowTitle(')
    sidebar_start = source.index('def _build_sidebar(self):')
    settings_source = source[settings_start:sidebar_start]
    closeup_start = source.index('self.lora_group = QGroupBox("Close-up')
    closeup_end = source.index('self.btn_cl_auto_zoom = QPushButton', closeup_start)
    closeup_source = source[closeup_start:closeup_end]
    assert 'QGroupBox("Local Diagnostics")' in settings_source
    assert 'self.settings_diagnostics_button = QPushButton("🩺 Run local Diagnostics…")' in settings_source
    assert 'self.settings_diagnostics_button.clicked.connect(self.open_diagnostics)' in settings_source
    assert 'btn_open_diagnostics' not in closeup_source


def test_live_frame_refresh_preserves_user_zoom_and_pan():
    from pathlib import Path
    canvas_source = (Path(__file__).resolve().parents[1] / "pai_studio" / "ui" / "canvas.py").read_text(encoding="utf-8")
    window_source = (Path(__file__).resolve().parents[1] / "pai_studio" / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "def set_live_image(" in canvas_source
    assert "self.preview.set_live_image(live_image, preserve_view=True)" in window_source
