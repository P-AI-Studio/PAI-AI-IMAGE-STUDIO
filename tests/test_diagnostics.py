from __future__ import annotations

import json
import time
from pathlib import Path

from pai_studio.diagnostics import (
    run_diagnostics,
    save_diagnostics_report,
    serialize_diagnostics_report,
)


def _by_id(items):
    return {item.check_id: item for item in items}


def test_diagnostics_report_expected_runtime_missing(tmp_path):
    items = _by_id(run_diagnostics({
        "runtime_dir": tmp_path / "missing-runtime",
        "encoder_path": tmp_path / "missing-encoder.gguf",
        "projector_path": tmp_path / "missing-mmproj.gguf",
        "workspace_dir": tmp_path,
    }))
    assert items["grounding_runtime"].status == "fail"
    assert items["grounding_encoder"].status == "fail"
    assert items["grounding_projector"].status == "fail"
    assert items["grounding_runtime"].remediation


def test_diagnostics_marks_unsupported_gpu_telemetry_not_applicable(tmp_path):
    items = _by_id(run_diagnostics({"runtime_dir": tmp_path, "gpu_telemetry_supported": False}))
    assert items["gpu_telemetry"].status == "not_applicable"


def test_diagnostics_times_out_hung_health_check(tmp_path):
    def slow_health_check():
        time.sleep(0.15)
        return True

    items = _by_id(run_diagnostics({"runtime_dir": tmp_path, "health_check": slow_health_check}, timeout_seconds=0.02))
    assert items["local_api"].status == "warning"
    assert "timed out" in items["local_api"].summary.lower()


def test_diagnostics_report_redacts_tokens_and_prompt_data(tmp_path):
    items = run_diagnostics({
        "runtime_dir": tmp_path,
        "technical_details": "Authorization: Bearer abc123 token=secret-value prompt=private image data:image/png;base64,QUJDRA==",
    })
    report = serialize_diagnostics_report(items)
    assert "abc123" not in report
    assert "secret-value" not in report
    assert "private" not in report
    assert "QUJDRA==" not in report
    json.loads(report)


def test_diagnostics_is_read_only(tmp_path):
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    run_diagnostics({"runtime_dir": tmp_path, "workspace_dir": tmp_path})
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert after == before


def test_diagnostics_export_is_explicit_and_atomic(tmp_path):
    items = run_diagnostics({"runtime_dir": tmp_path / "missing"})
    destination = tmp_path / "report.json"
    assert not destination.exists()
    save_diagnostics_report(items, destination)
    assert destination.is_file()
    json.loads(destination.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp"))
