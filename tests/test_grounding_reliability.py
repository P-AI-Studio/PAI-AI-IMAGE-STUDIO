from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from PIL import Image

import importlib
import json

from pai_studio.grounding import parse_qwen3vl_grounding_response


def _detailed_parser(text):
    module = importlib.import_module("pai_studio.grounding")
    parser = getattr(module, "parse_qwen3vl_grounding_response_detailed", None)
    assert callable(parser), "detailed grounding parser is not implemented"
    return parser(text)


def _run_strategy(*args, **kwargs):
    try:
        module = importlib.import_module("pai_studio.grounding_strategy")
    except ImportError:
        pytest.fail("grounding strategy module is not implemented")
    runner = getattr(module, "run_grounding_strategy", None)
    assert callable(runner), "run_grounding_strategy is not implemented"
    return runner(*args, **kwargs)


def test_detailed_parser_distinguishes_explicit_not_found():
    result = _detailed_parser('{"found": false}')
    assert result.status == "not_found"
    assert result.bbox_xyxy is None
    assert result.reason_code == "target_not_found"


def test_detailed_parser_reports_malformed_response():
    result = _detailed_parser("I cannot return coordinates")
    assert result.status == "invalid_response"
    assert result.bbox_xyxy is None
    assert result.reason_code == "bbox_missing"


@pytest.mark.parametrize("bbox", [
    [float("nan"), 0.1, 0.8, 0.9],
    [0.1, float("inf"), 0.8, 0.9],
    [0.8, 0.1, 0.2, 0.9],
    [0.2, 0.4, 0.2, 0.9],
])
def test_parser_rejects_non_finite_inverted_and_degenerate_boxes(bbox):
    result = _detailed_parser(
        json.dumps({"found": True, "bbox_2d": bbox}, allow_nan=True)
    )
    assert result.status == "invalid_response"
    assert result.bbox_xyxy is None
    assert parse_qwen3vl_grounding_response(
        json.dumps({"found": True, "bbox_2d": bbox}, allow_nan=True)
    ) is None


def test_parser_rejects_near_full_frame_box():
    result = _detailed_parser(
        '{"found": true, "bbox_2d": [0, 0, 1000, 990]}'
    )
    assert result.status == "invalid_response"
    assert result.reason_code == "near_full_frame"


def test_strategy_does_not_retry_explicit_not_found_identically():
    calls = []
    def locator(image, target, **kwargs):
        calls.append(target)
        return _detailed_parser('{"found": false}')
    result = _run_strategy(
        Image.new("RGB", (64, 64)), "target", locator=locator, profile="fast"
    )
    assert result.status == "not_found"
    assert len(calls) == 1


def test_strategy_retries_only_within_profile_budget():
    calls = []
    def locator(image, target, **kwargs):
        calls.append(target)
        return None if len(calls) < 2 else (0.2, 0.2, 0.6, 0.6)
    result = _run_strategy(
        Image.new("RGB", (64, 64)), "target", locator=locator, profile="balanced"
    )
    assert result.status == "found"
    assert result.bbox_xyxy == (0.2, 0.2, 0.6, 0.6)
    assert len(calls) == 2
    assert result.attempts == 2


def test_strategy_reports_timeout_separately():
    def locator(*args, **kwargs):
        raise TimeoutError("private prompt detail")
    result = _run_strategy(
        Image.new("RGB", (64, 64)), "target", locator=locator, profile="fast"
    )
    assert result.status == "timeout"
    assert result.reason_code == "inference_timeout"


def test_strategy_reports_runtime_error_separately():
    def locator(*args, **kwargs):
        raise RuntimeError("runtime unavailable")
    result = _run_strategy(
        Image.new("RGB", (64, 64)), "target", locator=locator, profile="fast"
    )
    assert result.status == "runtime_error"
    assert result.reason_code == "runtime_error"


def test_inference_shutdowns_server_after_request_failure():
    # Regression contract is exercised through the cleanup helper, independent of a real runtime.
    from pai_studio.grounding import Qwen3VLGrounder
    grounder = Qwen3VLGrounder("model.gguf", "mmproj.gguf")
    shutdown_calls = []
    grounder._start_server_locked = lambda **kwargs: "http://127.0.0.1:12345"
    grounder.shutdown = lambda: shutdown_calls.append(True)
    def fail_request(*args, **kwargs):
        raise OSError("simulated request failure")
    from unittest.mock import patch
    with patch("pai_studio.grounding.urllib.request.urlopen", side_effect=fail_request):
        with pytest.raises(RuntimeError):
            grounder.locate(Image.new("RGB", (32, 32)), "target")
    assert shutdown_calls == [True]


def test_logs_redact_image_payload_and_full_prompt():
    messages = []
    def locator(image, target, **kwargs):
        return None
    _run_strategy(
        Image.new("RGB", (64, 64)), "private user prompt content", locator=locator,
        profile="fast", log_callback=messages.append,
    )
    assert all("private user prompt content" not in message for message in messages)
    assert all("base64" not in message.lower() for message in messages)


def test_local_adapter_preserves_explicit_not_found_in_detailed_mode(monkeypatch):
    import pai_studio.grounding as grounding
    explicit = grounding.GroundingParseResult("not_found", None, "target_not_found")
    class FakeGrounder:
        def locate_detailed(self, *args, **kwargs):
            return explicit
    monkeypatch.setattr(grounding, "_get_qwen3vl_grounder", lambda *args: FakeGrounder())
    grounding._LOCAL_BBOX_CACHE.clear()
    result = grounding.detect_target_bbox_local(
        Image.new("RGB", (40, 40), "red"), "target", model_path="model.gguf",
        mmproj_path="projector.gguf", detailed=True,
    )
    assert isinstance(result, grounding.GroundingParseResult)
    assert result.status == "not_found"
    assert result.reason_code == "target_not_found"


def test_target_expansion_parser_strictly_filters_and_caps_candidates():
    from pai_studio.grounding import parse_target_description_candidates

    raw = json.dumps([
        "  visible human face  ",
        "FACE",
        "eyes, nose, mouth, and ears as visible facial features",
        "",
        7,
        "contains\x00control",
        "x" * 201,
        "another valid option",
    ])
    assert parse_target_description_candidates(raw, "face", max_candidates=2) == [
        "visible human face",
        "eyes, nose, mouth, and ears as visible facial features",
    ]


def test_target_expansion_parser_rejects_non_json_or_wrong_shape():
    from pai_studio.grounding import parse_target_description_candidates

    assert parse_target_description_candidates("Here are some ideas: face", "face") == []
    assert parse_target_description_candidates('{"candidates":"face"}', "face") == []


def test_local_expansion_adapter_reuses_selected_grounder_and_cleans_up_by_default(monkeypatch):
    import pai_studio.grounding as grounding

    observed = {}
    class FakeGrounder:
        def expand_target_descriptions(self, target, timeout_seconds=60.0, max_candidates=3, shutdown_after_request=True):
            observed.update(target=target, timeout=timeout_seconds, cap=max_candidates, shutdown=shutdown_after_request)
            return ["visible human face", "eyes and nose"]

    monkeypatch.setattr(grounding, "_get_qwen3vl_grounder", lambda *args: FakeGrounder())
    result = grounding.expand_qwen3vl_target_descriptions(
        "face", model_path="model.gguf", mmproj_path="projector.gguf", timeout_seconds=7,
        max_candidates=1, shutdown_after_request=False,
    )
    assert result == ["visible human face"]
    assert observed == {"target": "face", "timeout": 7, "cap": 1, "shutdown": False}


def test_grounding_strategy_tries_generated_description_only_after_direct_not_found():
    from pai_studio.grounding import GroundingParseResult
    from pai_studio.grounding_strategy import run_grounding_strategy

    calls = []
    expansions = []
    def locator(image, target, **kwargs):
        calls.append((image.size, target))
        if target == "face":
            return GroundingParseResult("not_found", None, "target_not_found")
        return GroundingParseResult("found", (0.2, 0.2, 0.6, 0.6), "ok")
    def expand(target, timeout):
        expansions.append((target, timeout))
        return ["visible human face"]

    result = run_grounding_strategy(Image.new("RGB", (320, 240)), "face", locator=locator,
                                    profile="balanced", expand_target=expand)
    assert result.status == "found"
    assert result.matched_description == "visible human face"
    assert expansions and expansions[0][0] == "face"
    assert any(target == "visible human face" for _size, target in calls)


def test_grounding_expansion_is_not_requested_when_deadline_is_exhausted(monkeypatch):
    import pai_studio.grounding_strategy as strategy
    from pai_studio.performance import GroundingProfile

    tick = {"now": 0.0}
    monkeypatch.setattr(strategy.time, "monotonic", lambda: tick["now"])
    def locator(*args, **kwargs):
        tick["now"] = 2.0
        return None
    expansion_calls = []
    profile = GroundingProfile("tiny-expansion", 1, 1, 1.0, 1.0, True, False)
    result = strategy.run_grounding_strategy(
        Image.new("RGB", (20, 20)), "target", locator=locator, profile=profile,
        expand_target=lambda *args: expansion_calls.append(args) or ["alternative"],
    )
    assert result.status == "timeout"
    assert expansion_calls == []


def test_semantic_expansion_http_timeout_never_rounds_up_to_one_second(monkeypatch):
    import pai_studio.grounding as grounding
    grounder = grounding.Qwen3VLGrounder("model.gguf", "projector.gguf")
    monkeypatch.setattr(grounder, "_start_server_locked", lambda **kwargs: "http://127.0.0.1:12345")
    monkeypatch.setattr(grounding.time, "monotonic", lambda: 10.0)
    observed = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": '["visible human face"]'}}]}).encode()

    def fake_urlopen(request, timeout):
        observed["timeout"] = timeout
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(grounding.urllib.request, "urlopen", fake_urlopen)
    result = grounder.expand_target_descriptions("face", timeout_seconds=0.2, max_candidates=1, shutdown_after_request=False)
    assert result == ["visible human face"]
    assert observed["timeout"] == pytest.approx(0.2)
    prompt = observed["payload"]["messages"][0]["content"].casefold()
    assert "same language as the user's target" in prompt


def test_english_prompt_translation_parser_returns_clean_translation_and_rejects_empty_output():
    from pai_studio.grounding import parse_english_prompt_translation

    assert parse_english_prompt_translation('```text\nA portrait of a red-haired woman in soft daylight.\n```', "زن مو قرمز در نور نرم") == "A portrait of a red-haired woman in soft daylight."
    assert parse_english_prompt_translation('"A quiet street after rain."', "بعد از باران") == "A quiet street after rain."
    assert parse_english_prompt_translation("", "یک پرتره") == ""


def test_english_prompt_translation_parser_rejects_output_that_stays_entirely_non_latin():
    from pai_studio.grounding import parse_english_prompt_translation

    assert parse_english_prompt_translation("یک پرتره زیبا با نور طبیعی", "یک پرتره زیبا") == ""
