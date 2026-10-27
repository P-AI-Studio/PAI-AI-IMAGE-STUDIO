"""Read-only local diagnostics with bounded checks and redacted report export."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Mapping
import urllib.request


@dataclass(frozen=True)
class DiagnosticItem:
    check_id: str
    status: str
    summary: str
    remediation: str = ""
    technical_details: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warning", "fail", "not_applicable"}:
            raise ValueError("Unsupported diagnostics status.")


_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret)\s*[:=]\s*[^\s,;]+"), "[SECRET]=[REDACTED]"),
    (re.compile(r"(?i)data:image/[^\s,;]+;base64,[A-Za-z0-9+/=_-]+"), "[IMAGE_DATA_REDACTED]"),
    (re.compile(r"(?i)\bprompt\s*[:=]\s*[^\n\r,;]+"), "prompt=[REDACTED]"),
    (re.compile(r"(?i)\b(?:hf|gh[pousr]?)_[A-Za-z0-9]{12,}\b"), "[TOKEN_REDACTED]"),
)


def redact_diagnostic_text(value: Any) -> str:
    text = str(value or "")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    try:
        home = str(Path.home())
        if home and home not in {"/", "\\"}:
            text = text.replace(home, "~")
    except (OSError, RuntimeError):
        pass
    # Keep reports compact and prevent arbitrary multi-megabyte output from plugins/tools.
    text = text[:4000]
    return text


def _ctx(context: Any, key: str, default=None):
    if isinstance(context, Mapping):
        return context.get(key, default)
    return getattr(context, key, default)


def _item(check_id: str, status: str, summary: str, remediation: str = "", technical_details: str = "") -> DiagnosticItem:
    return DiagnosticItem(check_id, status, redact_diagnostic_text(summary), redact_diagnostic_text(remediation), redact_diagnostic_text(technical_details))


def _timed_health_check(context: Any, timeout_seconds: float) -> DiagnosticItem:
    callback = _ctx(context, "health_check")
    server_url = str(_ctx(context, "server_url", "") or "").strip().rstrip("/")
    if callback is None and not server_url:
        return _item("local_api", "warning", "No active local API health endpoint is configured for this diagnostics run.",
                     "Run the Close-up test operation or configure a local endpoint when you need to verify live API readiness.")

    def check():
        if callback is not None:
            value = callback()
            if isinstance(value, Mapping):
                return bool(value.get("ok")), str(value.get("details", ""))
            return bool(value), "Injected local health check"
        url = server_url if server_url.endswith("/health") else server_url + "/health"
        with urllib.request.urlopen(url, timeout=max(0.1, timeout_seconds)) as response:
            return 200 <= int(getattr(response, "status", response.getcode())) < 300, f"HTTP {getattr(response, 'status', response.getcode())}"

    result: dict[str, Any] = {}
    done = threading.Event()

    def run():
        try:
            result["value"] = check()
        except Exception as exc:  # Surface only exception type; avoid leaking URL/token details.
            result["error_type"] = type(exc).__name__
        finally:
            done.set()

    thread = threading.Thread(target=run, name="pai-diagnostics-health", daemon=True)
    thread.start()
    cancel_event = _ctx(context, "cancel_event")
    deadline = time.monotonic() + max(0.01, timeout_seconds)
    while not done.is_set() and time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            return _item("local_api", "warning", "Diagnostics were cancelled while waiting for local API health.")
        done.wait(min(0.05, max(0.0, deadline - time.monotonic())))
    if not done.is_set():
        return _item("local_api", "warning", "Local API health check timed out.", "Check whether llama-server is running and responsive, then run Diagnostics again.")
    if "error_type" in result:
        return _item("local_api", "warning", "Local API health check could not complete.", "Check the local runtime status and Activity Log.", result["error_type"])
    ok, details = result.get("value", (False, ""))
    return _item("local_api", "pass" if ok else "warning",
                 "Local API health check passed." if ok else "Local API health check did not report ready.",
                 "" if ok else "Start or restart the local grounding runtime and retry.", details)


def run_diagnostics(context: Any = None, *, timeout_seconds: float = 3.0) -> list[DiagnosticItem]:
    """Run bounded, read-only checks; do not start a server, download, or run inference."""
    context = {} if context is None else context
    timeout = max(0.01, min(float(timeout_seconds), 30.0))
    items: list[DiagnosticItem] = []
    cancel_event = _ctx(context, "cancel_event")

    def cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def append_cancelled_if_needed() -> bool:
        if cancelled():
            items.append(_item("diagnostics_cancelled", "warning", "Diagnostics were cancelled; remaining checks were skipped."))
            return True
        return False

    runtime_dir = _ctx(context, "runtime_dir")
    if runtime_dir is None:
        try:
            from .workspace import LLAMA_GROUNDING_RUNTIME_DIR
            runtime_dir = LLAMA_GROUNDING_RUNTIME_DIR
        except Exception:
            runtime_dir = None
    if runtime_dir is None:
        items.append(_item("grounding_runtime", "warning", "Grounding runtime path is not configured.", "Open Model Catalog or Runtime settings to confirm the local llama.cpp runtime directory."))
    else:
        runtime_path = Path(runtime_dir).expanduser()
        expected = _ctx(context, "runtime_executable")
        if expected:
            candidates = [Path(expected).expanduser()]
        elif runtime_path.is_file():
            candidates = [runtime_path]
        else:
            candidates = list(runtime_path.glob("llama-server.exe")) + list(runtime_path.glob("llama-server")) if runtime_path.is_dir() else []
            if runtime_path.is_dir() and not candidates:
                try:
                    candidates = list(runtime_path.rglob("llama-server.exe"))[:1] + list(runtime_path.rglob("llama-server"))[:1]
                except OSError:
                    candidates = []
        found = next((p for p in candidates if p.is_file()), None)
        items.append(_item(
            "grounding_runtime", "pass" if found else "fail",
            "Local llama.cpp server executable was found." if found else "Local llama.cpp server executable was not found.",
            "Install or re-prepare the compatible local grounding runtime from Model Catalog." if not found else "",
            str(found) if found else f"runtime_dir={runtime_path}",
        ))

    if append_cancelled_if_needed():
        return items

    for check_id, key, display in (
        ("grounding_encoder", "encoder_path", "selected text encoder"),
        ("grounding_projector", "projector_path", "selected vision projector"),
    ):
        path_value = _ctx(context, key)
        if not path_value:
            items.append(_item(check_id, "warning", f"The {display} path is not configured for this diagnostics run.", f"Select a compatible {display} in Model Catalog to use semantic Close-up."))
        else:
            path = Path(path_value).expanduser()
            exists = path.is_file()
            items.append(_item(
                check_id, "pass" if exists else "fail",
                f"The {display} file exists." if exists else f"The {display} file is missing.",
                "Restore or reselect the matching model component." if not exists else "",
                str(path),
            ))

    if append_cancelled_if_needed():
        return items
    items.append(_timed_health_check(context, min(timeout, 10.0)))
    if append_cancelled_if_needed():
        return items

    telemetry_supported = _ctx(context, "gpu_telemetry_supported")
    if telemetry_supported is False:
        items.append(_item("gpu_telemetry", "not_applicable", "GPU telemetry is unavailable in this environment; this is not treated as a grounding-runtime failure.", "Use the system's GPU tools if you need hardware telemetry."))
    else:
        nvidia_smi = _ctx(context, "nvidia_smi") or shutil.which("nvidia-smi")
        if nvidia_smi and Path(str(nvidia_smi)).is_file():
            items.append(_item("gpu_telemetry", "pass", "NVIDIA telemetry utility is available.", technical_details=str(nvidia_smi)))
        else:
            items.append(_item("gpu_telemetry", "not_applicable", "NVIDIA telemetry utility was not found; diagnostics did not probe hardware with a blocking command.", "Install/update the NVIDIA driver only if GPU telemetry or CUDA inference itself is unavailable."))

    if append_cancelled_if_needed():
        return items

    workspace = _ctx(context, "workspace_dir")
    if workspace:
        path = Path(workspace).expanduser()
        if not path.exists():
            items.append(_item("workspace", "warning", "Workspace path does not exist.", "Confirm the configured portable data/workspace location.", str(path)))
        else:
            writable = os.access(path, os.W_OK)
            try:
                free_bytes = shutil.disk_usage(path).free
            except OSError:
                free_bytes = None
            status = "pass" if writable else "fail"
            summary = "Workspace is accessible and marked writable." if writable else "Workspace is not writable for the current process."
            details = f"free_bytes={free_bytes}" if free_bytes is not None else "free_space=unavailable"
            items.append(_item("workspace", status, summary, "Choose a writable folder and ensure sufficient free disk space." if not writable else "", details))

    package_versions = {}
    for package in ("Pillow", "PySide6", "pytest"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not installed"
    items.append(_item("python_packages", "pass", "Python package metadata was collected.", technical_details=json.dumps(package_versions, sort_keys=True)))

    supplied_details = _ctx(context, "technical_details")
    if supplied_details:
        items.append(_item("diagnostic_context", "warning", "Additional context was redacted before display/export.", technical_details=str(supplied_details)))
    return items


def serialize_diagnostics_report(items: list[DiagnosticItem]) -> str:
    """Serialize a local report with redaction applied to every string field."""
    rows = []
    for item in items:
        row = asdict(item)
        rows.append({key: redact_diagnostic_text(value) if isinstance(value, str) else value for key, value in row.items()})
    return json.dumps({"schema_version": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": rows}, indent=2, ensure_ascii=False) + "\n"


def save_diagnostics_report(items: list[DiagnosticItem], destination: str | Path) -> Path:
    """Write a report only when explicitly invoked by the UI after a file picker."""
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(serialize_diagnostics_report(items), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return path
