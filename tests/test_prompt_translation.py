from pathlib import Path

from pai_studio.engine import LocalEngine


def test_generation_prompt_translation_uses_selected_local_encoder_and_caches(monkeypatch, tmp_path):
    encoder = tmp_path / "selected-qwen3vl.gguf"
    projector = tmp_path / "selected-mmproj.gguf"
    engine = object.__new__(LocalEngine)
    engine.model_path = str(tmp_path)
    engine._prompt_translation_cache = {}
    logs = []
    statuses = []
    engine.log = logs.append
    engine.status_callback = statuses.append
    engine._find_model_parts = lambda _folder: {"llm": encoder, "llm_vision": projector}

    calls = []

    def translate(text, **kwargs):
        calls.append((text, kwargs))
        return "A detailed cinematic portrait in warm natural light."

    monkeypatch.setattr("pai_studio.engine.translate_prompt_via_local_encoder", translate)
    original = "یک پرتره سینمایی با نور طبیعی گرم"

    assert engine._translate_image_prompt_to_english(original, "image_generate") == (
        "A detailed cinematic portrait in warm natural light."
    )
    assert engine._translate_image_prompt_to_english(original, "image_edit") == (
        "A detailed cinematic portrait in warm natural light."
    )
    assert len(calls) == 1
    assert calls[0][0] == original
    assert calls[0][1]["model_path"] == encoder
    assert calls[0][1]["mmproj_path"] == projector
    assert calls[0][1]["gpu_layers"] == 0
    assert calls[0][1]["shutdown_after_request"] is True


def test_prompt_translation_falls_back_to_original_if_model_fails(monkeypatch, tmp_path):
    encoder = tmp_path / "selected-qwen3vl.gguf"
    projector = tmp_path / "selected-mmproj.gguf"
    engine = object.__new__(LocalEngine)
    engine.model_path = str(tmp_path)
    engine._prompt_translation_cache = {}
    logs = []
    statuses = []
    engine.log = logs.append
    engine.status_callback = statuses.append
    engine._find_model_parts = lambda _folder: {"llm": encoder, "llm_vision": projector}

    def fail(_text, **_kwargs):
        raise TimeoutError("test timeout")

    monkeypatch.setattr("pai_studio.engine.translate_prompt_via_local_encoder", fail)
    original = "گربه در یک باغ"
    assert engine._translate_image_prompt_to_english(original, "image_edit") == original
    assert any("preserving the original prompt" in line for line in logs)
    assert statuses[-1] == "Prompt translation unavailable · using the original prompt"
