# -*- coding: utf-8 -*-
"""
Background QThread workers for inference jobs, model loading, catalog downloads,
Real-ESRGAN setup, LoRA CivitAI hash lookup, and hardware telemetry.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:  # Headless test fallback when PySide6 is not installed
    class QThread:  # type: ignore[no-redef]
        def __init__(self, parent=None):
            self.parent = parent

    def Signal(*args, **kwargs):  # type: ignore[no-redef]
        return None

from .constants import (
    APP_VERSION,
    _download_ssl_context,
    _human_gb,
    _sha256_file,
    model_download_sources,
)
from .engine import Aborted, Job, LocalEngine, _find_nvidia_smi

class Worker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    aborted = Signal()
    grounding_review_requested = Signal(object)

    def __init__(self, engine: LocalEngine, job: Job):
        super().__init__()
        self.engine = engine
        self.job = job

    def request_grounding_review(self, payload: dict[str, Any], timeout_seconds: float = 900.0):
        """Ask the GUI thread to review a box while the worker waits cancellably."""
        from .ui.grounding_review_dialog import GroundingReviewDecision
        timeout = max(1.0, min(float(timeout_seconds), 900.0))
        event = threading.Event()
        request = dict(payload)
        request["event"] = event
        request["decision"] = GroundingReviewDecision("cancel")
        self.grounding_review_requested.emit(request)
        deadline = time.monotonic() + timeout
        while not event.wait(0.1):
            if self.engine.force_stop.is_set():
                return GroundingReviewDecision("cancel")
            if time.monotonic() >= deadline:
                return GroundingReviewDecision("cancel")
        if self.engine.force_stop.is_set():
            return GroundingReviewDecision("cancel")
        decision = request.get("decision")
        return decision if isinstance(decision, GroundingReviewDecision) else GroundingReviewDecision("cancel")

    def run(self):
        try:
            self.progress.emit(f"Running {self.job.workflow}…")
            paths = self.engine.process(self.job)
            self.completed.emit(paths)
        except Aborted:
            self.aborted.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

class ModelLoadWorker(QThread):
    loaded = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, engine: LocalEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.engine.ensure_loaded()
            self.loaded.emit("AtomicChat Qwen-Image 2.1 Turbo engine ready · CUDA local inference · no hosted API.")
        except Aborted:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))



class ModelBundleDownloadWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, target_dir: Path, assets: list[dict[str, Any]], parent=None):
        super().__init__(parent)
        self.target_dir = target_dir
        self.assets = list(assets)
        self.cancel_event = threading.Event()
        self.manifest_path = target_dir / "model-manifest.json"
        self.manifest: dict[str, Any] = {"bundle": "qwen-image-2.1-turbo-atomicchat-selectable-components-v3", "files": {}}
        try:
            saved = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("files"), dict):
                self.manifest.update(saved)
        except (OSError, ValueError, TypeError):
            pass
        self.completed_weight = 0
        self.total_weight = max(1, sum(int(item["size_hint"]) for item in self.assets))

    def request_cancel(self):
        self.cancel_event.set()

    def _write_manifest(self):
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def _emit_progress(self, current_bytes: int, current_hint: int, label: str):
        value = int(100 * (self.completed_weight + min(current_bytes, current_hint)) / self.total_weight)
        self.progress.emit(max(0, min(99, value)), label)

    def _verify_hash(self, path: Path, expected_hash: str, meta: Optional[dict[str, Any]] = None) -> str:
        hint = int(meta["size_hint"]) if meta else max(1, path.stat().st_size)
        title = meta["title"] if meta else path.name
        def report(byte_count: int):
            self._emit_progress(byte_count, hint, f"Verifying {title} · {byte_count / 1e9:.2f} / ~{hint / 1e9:.2f} GB")
        return _sha256_file(path, self.cancel_event, report)

    def _download_one(self, meta: dict[str, Any], dest: Path) -> bool:
        part = dest.with_name(dest.name + ".part")
        from .validation import validate_https_url, validate_sha256_hex

        expected_hash = validate_sha256_hex(str(meta.get("sha256", "")), allow_empty=True)
        expected_hint = int(meta["size_hint"])
        key = str(meta["filename"])
        if expected_hint <= 0:
            raise RuntimeError(f"Invalid expected artifact size for {key}.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        stored = self.manifest.get("files", {}).get(key, {})
        minimum_plausible = max(10_000_000, int(expected_hint * 0.70))
        if dest.is_file() and dest.stat().st_size < minimum_plausible:
            # Do not move/delete the active file before a replacement validates. An old
            # installed artifact remains usable until the staged download is accepted.
            self.progress.emit(0, f"Existing file is implausibly small; it will be preserved until a validated replacement is ready: {key}")
        if dest.is_file():
            stat = dest.stat()
            stored_hash = str(stored.get("sha256", "")).lower()
            if ((expected_hash and stored_hash == expected_hash) or (not expected_hash and len(stored_hash) == 64)) and stored.get("size") == stat.st_size and stored.get("mtime_ns") == stat.st_mtime_ns:
                self.completed_weight += expected_hint
                cache_label = (
                    f"Cached (pinned upstream SHA-256 recorded): {key}"
                    if expected_hash else
                    f"Cached using local hash metadata; upstream authenticity remains unverified: {key}"
                )
                self.progress.emit(min(99, int(100 * self.completed_weight / self.total_weight)), cache_label)
                return True
            self.progress.emit(min(98, int(100 * self.completed_weight / self.total_weight)), f"Verifying existing file: {key}")
            actual = self._verify_hash(dest, expected_hash, meta)
            if not expected_hash or actual == expected_hash:
                stat = dest.stat()
                self.manifest.setdefault("files", {})[key] = {"sha256": actual, "sha256_source": "upstream-pinned" if expected_hash else "computed-locally-no-upstream-pin", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                self._write_manifest()
                self.completed_weight += expected_hint
                verified_label = (
                    f"Existing file checked against pinned upstream SHA-256: {key}"
                    if expected_hash else
                    f"Existing file checked using local SHA-256 only; upstream authenticity remains unverified: {key}"
                )
                self.progress.emit(min(99, int(100 * self.completed_weight / self.total_weight)), verified_label)
                return True
            self.progress.emit(0, f"Existing file hash differs; the old file will be preserved until the replacement validates: {key}")

        last_error = None
        source_urls = tuple(validate_https_url(url) for url in model_download_sources(meta))
        if not source_urls:
            raise RuntimeError(f"No valid HTTPS source is configured for {key}.")
        for source_index, source_url in enumerate(source_urls):
            # Never resume a partial response from one host against another mirror.
            if source_index:
                part.unlink(missing_ok=True)
            for attempt in range(1, 3):
                if self.cancel_event.is_set():
                    return False
                offset = part.stat().st_size if part.exists() else 0
                try:
                    request = urllib.request.Request(
                        source_url,
                        headers={
                            "User-Agent": f"P.AI-Image-Studio/{APP_VERSION} (model-downloader)",
                            "Accept-Encoding": "identity",
                            **({"Range": f"bytes={offset}-"} if offset else {}),
                        },
                    )
                    expected_response_total = None
                    with urllib.request.urlopen(request, timeout=120, context=_download_ssl_context()) as response:
                        status = getattr(response, "status", response.getcode())
                        if offset and status != 206:
                            offset = 0
                        mode = "ab" if offset and status == 206 else "wb"
                        content_range = str(response.headers.get("Content-Range", ""))
                        if content_range and "/" in content_range:
                            try:
                                total_text = content_range.rsplit("/", 1)[1]
                                if total_text != "*":
                                    expected_response_total = int(total_text)
                            except (ValueError, TypeError):
                                expected_response_total = None
                        if expected_response_total is None:
                            content_length = response.headers.get("Content-Length")
                            if content_length is not None:
                                try:
                                    expected_response_total = offset + int(content_length) if status == 206 else int(content_length)
                                except (ValueError, TypeError):
                                    expected_response_total = None
                        current = offset
                        with part.open(mode) as handle:
                            while True:
                                if self.cancel_event.is_set():
                                    handle.flush()
                                    os.fsync(handle.fileno())
                                    raise Aborted()
                                block = response.read(4 * 1024 * 1024)
                                if not block:
                                    break
                                handle.write(block)
                                current += len(block)
                                self._emit_progress(current, expected_hint, f"Downloading {meta['title']} · {current / 1e9:.2f} / ~{expected_hint / 1e9:.2f} GB")
                            handle.flush()
                            os.fsync(handle.fileno())
                    actual_size = part.stat().st_size
                    if expected_response_total is not None and actual_size != expected_response_total:
                        raise RuntimeError(
                            f"Incomplete response for {key}: received {actual_size} bytes, server reported {expected_response_total}. Retry will resume or use the mirror."
                        )
                    if actual_size < minimum_plausible:
                        part.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Downloaded file {key} is too small ({actual_size} bytes; expected roughly {expected_hint} bytes). "
                            "The server may have returned an HTML/error page or an incomplete file."
                        )
                    # Reject HTML/XML error pages even when a CDN returns a plausible-sized response.
                    with part.open("rb") as validation_handle:
                        header = validation_handle.read(512).lstrip().lower()
                    if header.startswith((b"<!doctype html", b"<html", b"<?xml", b'{"error"')):
                        part.unlink(missing_ok=True)
                        raise RuntimeError(f"The server returned an error document instead of model data for {key}.")
                    actual = self._verify_hash(part, expected_hash, meta)
                    if expected_hash and actual != expected_hash:
                        part.unlink(missing_ok=True)
                        # Do not retry a different host for integrity failure: a mirror mismatch
                        # must not be silently accepted for a pinned checkpoint.
                        raise RuntimeError(
                            f"SHA-256 verification failed for {key}. The downloaded file was not installed. Check the connection/proxy and retry."
                        )
                    # Atomic replacement occurs only after size/content and any pinned
                    # digest have passed validation. Until this point dest remains intact.
                    os.replace(part, dest)
                    stat = dest.stat()
                    self.manifest.setdefault("files", {})[key] = {"sha256": actual, "sha256_source": "upstream-pinned" if expected_hash else "computed-locally-no-upstream-pin", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                    self._write_manifest()
                    self.completed_weight += expected_hint
                    saved_label = (
                        f"Downloaded and verified against pinned upstream SHA-256: {key}"
                        if expected_hash else
                        f"Downloaded and size/header checked; upstream authenticity remains unverified (local SHA-256 only): {key}"
                    )
                    self.progress.emit(min(99, int(100 * self.completed_weight / self.total_weight)), saved_label)
                    return True
                except Aborted:
                    part.unlink(missing_ok=True)
                    return False
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code == 416 and part.exists():
                        part.unlink(missing_ok=True)
                    elif exc.code in (401, 403, 404):
                        part.unlink(missing_ok=True)
                        # Try next configured source (e.g., HF mirror) before reporting a hard failure.
                        if source_index == len(source_urls) - 1:
                            raise RuntimeError(f"Every model source returned HTTP {exc.code} for {key}; verify the URL and access permissions. Last source: {source_url}") from exc
                except RuntimeError as exc:
                    last_error = exc
                    if "SHA-256 verification failed" in str(exc):
                        # A corrupt primary response is rejected. Try the next source, but
                        # install only if that source independently matches the same pin.
                        part.unlink(missing_ok=True)
                        if source_index < len(source_urls) - 1:
                            break
                        raise
                    # An incomplete response is safe to retry on this host; stale partials
                    # are deleted if they cannot be resumed cleanly.
                    if attempt == 2:
                        part.unlink(missing_ok=True)
                except (urllib.error.URLError, TimeoutError, OSError, ssl.SSLError) as exc:
                    last_error = exc
                if attempt < 2:
                    time.sleep(float(attempt))
        raise RuntimeError(f"Download failed for {key} after all available sources and retries: {last_error}")

    def run(self):
        try:
            if not self.assets:
                raise RuntimeError("No downloadable catalog components were selected. Choose a catalog model or browse a local file.")
            self.target_dir.mkdir(parents=True, exist_ok=True)
            stored_files = self.manifest.get("files", {})
            new_bytes = 0
            for item in self.assets:
                path = self.target_dir / item["filename"]
                needs_extra_space = not path.is_file()
                if path.is_file():
                    stat = path.stat()
                    stored = stored_files.get(item["filename"], {}) if isinstance(stored_files, dict) else {}
                    item_hash = str(item.get("sha256", "")).lower()
                    stored_hash = str(stored.get("sha256", "")).lower()
                    verified_cache = (
                        ((item_hash and stored_hash == item_hash) or (not item_hash and len(stored_hash) == 64))
                        and stored.get("size") == stat.st_size
                        and stored.get("mtime_ns") == stat.st_mtime_ns
                    )
                    # If it is not cached as verified, the old file is kept until the replacement validates.
                    needs_extra_space = not verified_cache
                if needs_extra_space:
                    new_bytes += int(item["size_hint"])
            required_free = max(750_000_000, new_bytes + 750_000_000)
            free_bytes = shutil.disk_usage(self.target_dir).free
            if new_bytes and free_bytes < required_free:
                raise RuntimeError(
                    f"This drive has {free_bytes / 1e9:.1f} GB free; the selected download requires about {_human_gb(new_bytes)}, and at least {_human_gb(required_free)} free is recommended."
                )
            for meta in self.assets:
                if self.cancel_event.is_set():
                    self.cancelled.emit()
                    return
                if not self._download_one(meta, self.target_dir / meta["filename"]):
                    self.cancelled.emit()
                    return
            unpinned = [str(item.get("filename", "artifact")) for item in self.assets if not str(item.get("sha256", "")).strip()]
            if unpinned:
                self.progress.emit(
                    100,
                    f"All selected components passed available integrity checks. Upstream authenticity remains unverified for {len(unpinned)} item(s) without a pinned SHA-256; local hashes are for corruption checks only.",
                )
            else:
                self.progress.emit(100, "All selected components downloaded and checked against pinned upstream SHA-256 values.")
            self.completed.emit(str(self.target_dir.resolve()))
        except Exception as exc:
            self.failed.emit(f"{exc}\n\nPartial .part files are retained for retry when possible.")


class RealESRGANRuntimeWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, engine: LocalEngine, parent=None):
        super().__init__(parent)
        self.engine = engine

    def run(self):
        try:
            path = self.engine._ensure_realesrgan_runtime()
            self.completed.emit(str(path))
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")



class LoRAInfoLookupWorker(QThread):
    """Optional CivitAI LoRA info lookup using the selected local file's SHA-256."""
    progress = Signal(str)
    found = Signal(dict)
    failed = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = str(path)

    def run(self):
        file_path = Path(self.path).expanduser()
        if not file_path.is_file():
            self.failed.emit("Select an existing LoRA file first.")
            return
        try:
            size = file_path.stat().st_size
            digest = hashlib.sha256()
            processed = 0
            with file_path.open("rb") as handle:
                while True:
                    block = handle.read(8 * 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    processed += len(block)
                    self.progress.emit(f"Hashing local LoRA · {processed / max(1, size) * 100:.0f}%")
            sha256 = digest.hexdigest()
            url = "https://civitai.com/api/v1/model-versions/by-hash/" + urllib.parse.quote(sha256, safe="")
            self.progress.emit("Checking CivitAI metadata by SHA-256…")
            request = urllib.request.Request(url, headers={
                "User-Agent": f"P.AI-Image-Studio/{APP_VERSION}",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(request, timeout=18, context=_download_ssl_context()) as response:
                payload = json.loads(response.read(3 * 1024 * 1024 + 1).decode("utf-8", errors="replace"))
            if not isinstance(payload, dict):
                raise RuntimeError("CivitAI returned an unexpected response.")
            version = payload.get("modelVersion") if isinstance(payload.get("modelVersion"), dict) else payload
            model = version.get("model") if isinstance(version.get("model"), dict) else payload.get("model", {})
            if not isinstance(model, dict):
                model = {}
            def short(value: Any, limit: int = 500) -> str:
                if isinstance(value, list):
                    value = ", ".join(str(x) for x in value if x not in (None, ""))
                text_value = re.sub(r"<[^>]+>", " ", str(value or ""))
                text_value = re.sub(r"\s+", " ", text_value).strip()
                return text_value[:limit] + ("…" if len(text_value) > limit else "")
            trained = version.get("trainedWords") or version.get("trained_words") or []
            if isinstance(trained, str):
                trained = re.split(r"[,\n;|]", trained)
            trigger_words = [short(item, 100) for item in trained if short(item, 100)] if isinstance(trained, list) else []
            examples_value = version.get("examplePrompts") or version.get("example_prompts") or []
            if isinstance(examples_value, str):
                examples_value = re.split(r"\n+|\s*\|\|\s*", examples_value)
            example_prompts = [short(item, 360) for item in examples_value if short(item, 360)] if isinstance(examples_value, list) else []
            images = version.get("images") or payload.get("images") or model.get("images") or []
            if isinstance(images, list):
                for sample in images:
                    if not isinstance(sample, dict):
                        continue
                    meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
                    prompt = meta.get("prompt") or sample.get("prompt") or sample.get("examplePrompt")
                    if isinstance(prompt, list):
                        example_prompts.extend(short(item, 360) for item in prompt if short(item, 360))
                    elif prompt:
                        example_prompts.append(short(prompt, 360))
                    if len(example_prompts) >= 4:
                        break
            example_prompts = list(dict.fromkeys(item for item in example_prompts if item))[:4]
            model_id = model.get("id") or payload.get("modelId")
            version_id = version.get("id") or payload.get("id")
            page_url = (f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
                        if model_id and version_id else f"https://civitai.com/models/{model_id}" if model_id else "")
            self.found.emit({
                "sha256": sha256,
                "name": short(model.get("name") or version.get("name") or file_path.stem, 180),
                "base_model": short(version.get("baseModel") or version.get("base_model") or "Not specified", 140),
                "creator": short((model.get("creator") or {}).get("username", "") if isinstance(model.get("creator"), dict) else model.get("creator", "Not specified"), 120),
                "description": short(version.get("description") or model.get("description") or "No public description found.", 600),
                "trigger_words": list(dict.fromkeys(trigger_words))[:24],
                "example_prompts": example_prompts,
                "url": page_url,
                "source": "CivitAI (SHA-256 matched)",
            })
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self.failed.emit("No CivitAI record matched this file hash. Local metadata is still available.")
            elif exc.code == 429:
                self.failed.emit("CivitAI rate-limited this lookup. Try again later.")
            else:
                self.failed.emit(f"CivitAI lookup failed (HTTP {exc.code}). Local metadata is still available.")
        except Exception as exc:
            self.failed.emit(f"CivitAI lookup failed: {exc}. Local metadata is still available.")



class HardwareMetricsWorker(QThread):
    """Low-overhead CPU/RAM and NVIDIA telemetry reader isolated from the UI thread."""
    metrics_ready = Signal(dict)

    def __init__(self, gpu_index: int = 0, parent=None):
        super().__init__(parent)
        self.gpu_index = max(0, int(gpu_index))

    def run(self):
        metrics = {
            "cpu": {"text": "CPU N/A", "tooltip": "CPU usage unavailable; install psutil."},
            "ram": {"text": "RAM N/A", "tooltip": "RAM usage unavailable; install psutil."},
            "gpu": {"text": "GPU N/A", "tooltip": "NVIDIA GPU metrics unavailable; check the NVIDIA driver and nvidia-smi."},
            "vram": {"text": "VRAM N/A", "tooltip": "Dedicated VRAM usage unavailable."},
            "temp": {"text": "TEMP N/A", "tooltip": "GPU temperature unavailable."},
        }
        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=0.12)
                metrics["cpu"] = {"text": f"CPU {cpu:.0f}%", "tooltip": f"System CPU utilization: {cpu:.1f}%"}
                memory = psutil.virtual_memory()
                metrics["ram"] = {
                    "text": f"RAM {memory.percent:.0f}%",
                    "tooltip": f"System RAM: {memory.used / (1024**3):.2f} / {memory.total / (1024**3):.2f} GiB",
                }
            except Exception:
                pass
        smi = _find_nvidia_smi()
        if smi:
            try:
                result = subprocess.run(
                    [str(smi), f"--id={self.gpu_index}", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
                    if len(parts) >= 5:
                        gpu_util, used, total, temp, name = parts[:5]
                        used_mb, total_mb = float(used), float(total)
                        metrics["gpu"] = {"text": f"GPU {float(gpu_util):.0f}%", "tooltip": f"{name} · utilization {gpu_util}%"}
                        metrics["vram"] = {"text": f"VRAM {used_mb / 1024:.1f}/{total_mb / 1024:.0f}G", "tooltip": f"Dedicated VRAM: {used_mb / 1024:.2f} / {total_mb / 1024:.2f} GiB"}
                        metrics["temp"] = {"text": f"TEMP {float(temp):.0f}°C", "tooltip": f"GPU core temperature: {temp} °C"}
            except Exception:
                pass
        self.metrics_ready.emit(metrics)



class DiagnosticsWorker(QThread):
    """Run read-only health checks away from the GUI thread."""
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, context: dict[str, Any], parent=None):
        super().__init__(parent)
        self.context = dict(context)
        self.cancel_event = threading.Event()
        self.context["cancel_event"] = self.cancel_event

    def request_cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            from .diagnostics import run_diagnostics
            results = run_diagnostics(self.context)
            self.completed.emit(results)
        except Exception as exc:
            self.failed.emit(f"Diagnostics failed ({type(exc).__name__}).")
