from __future__ import annotations

import pytest

import importlib


def _get_profile(name):
    try:
        module = importlib.import_module("pai_studio.performance")
    except ImportError:
        pytest.fail("performance policy module is not implemented")
    getter = getattr(module, "get_grounding_profile", None)
    assert callable(getter), "get_grounding_profile is not implemented"
    return getter(name)


def test_profiles_are_immutable():
    profile = _get_profile("balanced")
    with pytest.raises((AttributeError, TypeError)):
        profile.max_attempts = 0


def test_fast_profile_disables_multiscale():
    profile = _get_profile("fast")
    assert profile.max_attempts == 1
    assert profile.max_tiles == 0
    assert not profile.multiscale_enabled
    assert profile.per_request_timeout_seconds > 0
    assert profile.total_budget_seconds >= profile.per_request_timeout_seconds


def test_balanced_profile_uses_small_finite_budget():
    profile = _get_profile("balanced")
    assert profile.max_attempts >= 2
    assert 1 <= profile.max_tiles <= 4
    assert profile.total_budget_seconds < 1200
    assert profile.multiscale_enabled


def test_accuracy_profile_has_larger_finite_budget():
    balanced = _get_profile("balanced")
    accuracy = _get_profile("accuracy")
    assert accuracy.max_attempts >= balanced.max_attempts
    assert accuracy.max_tiles > balanced.max_tiles
    assert accuracy.total_budget_seconds > balanced.total_budget_seconds
    assert accuracy.review_ambiguous
    with pytest.raises(ValueError):
        _get_profile("unlimited")


def test_profile_settings_migrate_without_losing_existing_settings():
    from pai_studio.constants import SETTINGS_VERSION
    from pai_studio.validation import validate_and_migrate_settings

    migrated = validate_and_migrate_settings({
        "settings_version": SETTINGS_VERSION - 1,
        "workflow_mode": "image_edit",
        "gen_prompt": "preserve this setting",
        "grounding_profile": "accuracy",
    })
    assert migrated["settings_version"] == SETTINGS_VERSION
    assert migrated["workflow_mode"] == "image_edit"
    assert migrated["gen_prompt"] == "preserve this setting"
    assert migrated["grounding_profile"] == "accuracy"


def test_invalid_saved_profile_is_safely_defaulted():
    from pai_studio.validation import validate_and_migrate_settings

    loaded = validate_and_migrate_settings({"settings_version": 39, "grounding_profile": "unlimited"})
    assert loaded["grounding_profile"] == "balanced"


def test_ui_profile_selection_updates_strategy_policy():
    from pathlib import Path
    from pai_studio.performance import normalize_grounding_profile
    from pai_studio.validation import validate_job_custom

    selected = normalize_grounding_profile("accuracy")
    job_custom = validate_job_custom({"grounding_profile": selected})
    assert _get_profile(job_custom["grounding_profile"]).name == "accuracy"
    source = Path("pai_studio/ui/main_window.py").read_text(encoding="utf-8")
    assert 'self.grounding_profile_combo.currentData()' in source
    assert '"grounding_profile": normalize_grounding_profile' in source


def test_profile_help_text_describes_tradeoffs_without_accuracy_guarantees():
    from pai_studio.performance import grounding_profile_description

    text = grounding_profile_description("accuracy").lower()
    assert "cannot guarantee" in text
    assert "memory" in text or "latency" in text


def test_profile_budget_is_enforced_during_cancellation():
    from PIL import Image
    from pai_studio.grounding_strategy import run_grounding_strategy

    calls = []
    result = run_grounding_strategy(
        Image.new("RGB", (64, 64)), "target",
        locator=lambda *args, **kwargs: calls.append(1),
        profile="accuracy", should_cancel=lambda: True,
    )
    assert result.status == "cancelled"
    assert calls == []


def test_enhance_ultimate_off_selects_single_full_frame_pass_without_tiles():
    from pai_studio.performance import enhance_detail_mode_for_ultimate

    assert enhance_detail_mode_for_ultimate(False) == "fast"
    assert enhance_detail_mode_for_ultimate(True) == "ultimate"
