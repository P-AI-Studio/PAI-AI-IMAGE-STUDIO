from __future__ import annotations

import importlib
import io

import pytest
from PIL import Image

from pai_studio.performance import get_grounding_profile
from pai_studio.grounding_strategy import run_grounding_strategy


def _geometry(name, *args, **kwargs):
    module = importlib.import_module("pai_studio.grounding_strategy")
    function = getattr(module, name, None)
    assert callable(function), f"{name} is not implemented"
    return function(*args, **kwargs)


def test_tile_plan_is_bounded_and_overlapping():
    tiles = _geometry("plan_grounding_tiles", (1200, 800), get_grounding_profile("balanced"))
    assert 1 <= len(tiles) <= get_grounding_profile("balanced").max_tiles
    assert all(0 <= l < r <= 1200 and 0 <= t < b <= 800 for l, t, r, b in tiles)
    assert any(max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
               for i, a in enumerate(tiles) for b in tiles[i + 1:])


def test_tile_bbox_maps_to_original_normalized_coordinates():
    bbox = _geometry("map_tile_bbox_to_image", (100, 50, 300, 250), (400, 300), (0.25, 0.25, 0.75, 0.75))
    assert bbox == pytest.approx((0.375, 1/3, 0.625, 2/3))


def test_tile_mapping_handles_image_edges_and_aspect_ratio():
    bbox = _geometry("map_tile_bbox_to_image", (300, 0, 400, 100), (400, 100), (0.8, 0.1, 1.0, 0.9))
    assert bbox == pytest.approx((0.95, 0.1, 1.0, 0.9))


def test_multiscale_deduplicates_overlapping_boxes():
    boxes = ((0.10, 0.10, 0.40, 0.40), (0.11, 0.11, 0.41, 0.41))
    result = _geometry("deduplicate_grounding_candidates", boxes, 0.5)
    assert len(result) == 1


def test_multiscale_returns_ambiguous_when_distinct_candidates_remain():
    calls = []
    def locator(image, target, **kwargs):
        calls.append(image.size)
        if len(calls) <= 3:
            return None
        return (0.1, 0.1, 0.3, 0.3)
    result = run_grounding_strategy(Image.new("RGB", (300, 200)), "target", locator=locator, profile="accuracy")
    assert result.status == "ambiguous"
    assert result.bbox_xyxy is None
    assert len(result.candidates) >= 2


def test_cancel_stops_before_next_tile():
    calls = []
    def locator(image, target, **kwargs):
        calls.append(image.size)
        return None
    result = run_grounding_strategy(
        Image.new("RGB", (300, 200)), "target", locator=locator, profile="balanced",
        should_cancel=lambda: len(calls) >= 2,
    )
    assert result.status == "cancelled"
    assert len(calls) == 2


def test_total_attempt_budget_is_never_exceeded():
    calls = []
    def locator(image, target, **kwargs):
        calls.append(image.size)
        return None
    profile = get_grounding_profile("accuracy")
    result = run_grounding_strategy(Image.new("RGB", (300, 200)), "missing", locator=locator, profile=profile)
    assert len(calls) <= profile.max_attempts + profile.max_tiles
    assert result.attempts == len(calls)


def test_exif_oriented_image_uses_oriented_coordinate_frame():
    raw = Image.new("RGB", (32, 16), "white")
    exif = raw.getexif()
    exif[274] = 6
    buff = io.BytesIO()
    raw.save(buff, format="JPEG", exif=exif)
    buff.seek(0)
    loaded = Image.open(buff)
    observed = []
    def locator(image, target, **kwargs):
        observed.append(image.size)
        return (0.1, 0.1, 0.7, 0.8)
    result = run_grounding_strategy(loaded, "target", locator=locator, profile="fast")
    assert result.status == "found"
    assert observed[0] == (16, 32)


def test_normalized_bbox_validation_rejects_non_finite_and_bad_geometry():
    validation = importlib.import_module("pai_studio.validation")
    validator = getattr(validation, "validate_normalized_bbox", None)
    assert callable(validator), "normalized bbox validator is not implemented"
    assert validator((0.1, 0.2, 0.8, 0.9)) == (0.1, 0.2, 0.8, 0.9)
    with pytest.raises(validation.ValidationError):
        validator((float("nan"), 0.2, 0.8, 0.9))
    with pytest.raises(validation.ValidationError):
        validator((0.8, 0.2, 0.2, 0.9))


def test_engine_closeup_calls_bounded_grounding_strategy(monkeypatch):
    import pai_studio.engine as engine_module
    engine = engine_module.LocalEngine(log_callback=lambda message: None)
    captured = []
    def fake_locator(image, target, log_callback=None, model_path=None, mmproj_path=None, timeout_seconds=240, detailed=False, shutdown_after_request=True):
        captured.append((image.size, target, model_path, mmproj_path, timeout_seconds, shutdown_after_request))
        return (0.2, 0.25, 0.7, 0.8)
    monkeypatch.setattr(engine_module, "detect_target_bbox_local", fake_locator)
    resolver = getattr(engine, "_locate_closeup_target", None)
    assert callable(resolver), "engine Close-up strategy integration is not implemented"
    result = resolver(Image.new("RGB", (100, 100)), "test target", "model.gguf", "mmproj.gguf", profile="fast")
    assert result.status == "found"
    assert result.bbox_xyxy == (0.2, 0.25, 0.7, 0.8)
    assert captured and captured[0][2:4] == ("model.gguf", "mmproj.gguf")


def test_accuracy_retries_use_distinct_prompt_variants():
    observed_targets = []

    def locator(image, target, **kwargs):
        observed_targets.append(target)
        return None

    run_grounding_strategy(Image.new("RGB", (64, 64)), "silver ring", locator=locator, profile="accuracy")
    assert len(observed_targets) >= 3
    assert len(set(observed_targets[:3])) == 3


def test_request_timeout_never_exceeds_remaining_total_budget(monkeypatch):
    import pai_studio.grounding_strategy as strategy
    from pai_studio.performance import GroundingProfile

    times = iter((0.0, 0.25, 0.25, 0.5))
    monkeypatch.setattr(strategy.time, "monotonic", lambda: next(times))
    observed_timeouts = []

    def locator(image, target, **kwargs):
        observed_timeouts.append(kwargs["timeout_seconds"])
        return (0.1, 0.1, 0.4, 0.4)

    profile = GroundingProfile("tiny-test", 1, 0, 2.0, 0.5, False, False)
    result = run_grounding_strategy(Image.new("RGB", (64, 64)), "target", locator=locator, profile=profile)
    assert result.status == "found"
    assert observed_timeouts == [pytest.approx(0.25)]


def test_locator_result_after_total_deadline_is_discarded(monkeypatch):
    import pai_studio.grounding_strategy as strategy
    from pai_studio.performance import GroundingProfile

    now = {"value": 0.0}
    def monotonic():
        return now["value"]
    monkeypatch.setattr(strategy.time, "monotonic", monotonic)
    observed_timeouts = []

    def slow_locator(image, target, **kwargs):
        observed_timeouts.append(kwargs["timeout_seconds"])
        now["value"] = 0.6
        return (0.1, 0.1, 0.4, 0.4)

    profile = GroundingProfile("tiny-test", 1, 0, 2.0, 0.5, False, False)
    result = run_grounding_strategy(Image.new("RGB", (64, 64)), "target", locator=slow_locator, profile=profile)
    assert result.status == "timeout"
    assert result.reason_code == "total_budget_exhausted"
    assert observed_timeouts == [pytest.approx(0.5)]


def test_explicit_whole_image_not_found_continues_to_tiles_and_maps_box():
    from pai_studio.grounding import GroundingParseResult

    seen = []
    def locator(image, target, **kwargs):
        seen.append((image.size, target))
        if len(seen) == 1:
            return GroundingParseResult("not_found", None, "target_not_found")
        if len(seen) == 2:
            return GroundingParseResult("found", (0.2, 0.2, 0.6, 0.6), "ok")
        return GroundingParseResult("not_found", None, "target_not_found")

    source = Image.new("RGB", (300, 200))
    result = run_grounding_strategy(source, "small object", locator=locator, profile="balanced")
    assert result.status == "found"
    assert result.bbox_xyxy is not None
    assert result.bbox_xyxy != (0.2, 0.2, 0.6, 0.6)  # mapped from tile to source coordinates
    assert len(seen) > 1
    assert seen[0][0] == source.size
    assert all(size != source.size for size, _target in seen[1:])


def test_engine_reuses_selected_grounder_for_expansion_and_shuts_down_after_strategy(monkeypatch):
    import pai_studio.engine as engine_module
    from pai_studio.grounding import GroundingParseResult

    calls = []
    shutdowns = []
    engine = engine_module.LocalEngine(log_callback=lambda message: None)

    def fake_locator(image, target, log_callback=None, model_path=None, mmproj_path=None,
                     timeout_seconds=240, detailed=False, shutdown_after_request=True):
        calls.append(("locate", target, model_path, mmproj_path, shutdown_after_request))
        if target == "face":
            return GroundingParseResult("not_found", None, "target_not_found")
        return GroundingParseResult("found", (0.2, 0.2, 0.6, 0.6), "ok")

    def fake_expand(target, *, model_path=None, mmproj_path=None, timeout_seconds=60,
                    max_candidates=3, shutdown_after_request=True):
        calls.append(("expand", target, model_path, mmproj_path, shutdown_after_request))
        return ["visible human face"]

    monkeypatch.setattr(engine_module, "detect_target_bbox_local", fake_locator)
    monkeypatch.setattr(engine_module, "expand_qwen3vl_target_descriptions", fake_expand)
    monkeypatch.setattr(engine_module, "_shutdown_qwen3vl_grounder", lambda: shutdowns.append(True))

    result = engine._locate_closeup_target(
        Image.new("RGB", (320, 240)), "face", "selected.gguf", "selected-mmproj.gguf", profile="balanced"
    )
    assert result.status == "found"
    assert result.matched_description == "visible human face"
    assert [item[0] for item in calls].count("expand") == 1
    assert all(item[2:4] == ("selected.gguf", "selected-mmproj.gguf") for item in calls)
    assert all(item[4] is False for item in calls)
    assert shutdowns == [True]


def test_ambiguous_semantic_candidates_keep_model_generated_labels_aligned():
    from pai_studio.grounding import GroundingParseResult

    def locator(image, target, **kwargs):
        if target == "face":
            return GroundingParseResult("not_found", None, "target_not_found")
        if target == "visible eyes":
            return GroundingParseResult("found", (0.05, 0.1, 0.3, 0.4), "ok")
        if target == "visible nose":
            return GroundingParseResult("found", (0.55, 0.1, 0.8, 0.4), "ok")
        return GroundingParseResult("not_found", None, "target_not_found")

    result = run_grounding_strategy(
        Image.new("RGB", (320, 240)), "face", locator=locator, profile="accuracy",
        expand_target=lambda target, timeout: ["visible eyes", "visible nose"],
    )
    assert result.status == "ambiguous"
    assert len(result.candidates) == 2
    assert result.candidate_descriptions == ("visible eyes", "visible nose")
