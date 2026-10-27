from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from pai_studio.validation import ValidationError, validate_archive_member_path
from pai_studio.workers import ModelBundleDownloadWorker
from pai_studio.workspace import _extract_llama_grounding_archive_safe, install_directory_transactionally


PAYLOAD = b"P" * 10_000_128
GOOD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _SignalSink:
    def __init__(self):
        self.calls = []

    def emit(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FakeResponse:
    def __init__(self, data: bytes, *, declared_length: int | None = None, on_first_read=None):
        self.data = data
        self.position = 0
        self.status = 200
        self.headers = {"Content-Length": str(declared_length if declared_length is not None else len(data))}
        self.on_first_read = on_first_read
        self._called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getcode(self):
        return self.status

    def read(self, size=-1):
        if not self._called:
            self._called = True
            if self.on_first_read:
                self.on_first_read()
        if self.position >= len(self.data):
            return b""
        size = len(self.data) - self.position if size < 0 else size
        chunk = self.data[self.position:self.position + size]
        self.position += len(chunk)
        return chunk


class _WorkerHarness:
    @staticmethod
    def build(tmp_path: Path):
        worker = ModelBundleDownloadWorker(tmp_path / "bundle", [], None)
        worker.progress = _SignalSink()
        worker.failed = _SignalSink()
        worker.completed = _SignalSink()
        worker.cancelled = _SignalSink()
        worker._emit_progress = lambda *args, **kwargs: None
        return worker


def _meta(digest: str = GOOD_SHA, **overrides):
    meta = {
        "filename": "asset.gguf",
        "title": "Unit-test artifact",
        "url": "https://example.com/asset.gguf",
        "sha256": digest,
        "size_hint": 10_000_000,
    }
    meta.update(overrides)
    return meta


def test_wrong_pinned_digest_rejects_download_and_preserves_installed_file(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = b"O" * 10_000_100
    dest.write_bytes(old)
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD))
    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        worker._download_one(_meta("0" * 64), dest)
    assert dest.read_bytes() == old
    assert not dest.with_name(dest.name + ".part").exists()


def test_missing_upstream_digest_is_never_called_authenticated(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD))
    assert worker._download_one(_meta(""), dest) is True
    record = worker.manifest["files"]["asset.gguf"]
    assert record["sha256_source"] == "computed-locally-no-upstream-pin"
    assert record["sha256"] == GOOD_SHA
    assert record["sha256_source"] != "upstream-pinned"


def test_missing_upstream_digest_is_explicit_in_download_progress(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD))

    assert worker._download_one(_meta(""), dest) is True
    labels = [str(args[1]) for args, _kwargs in worker.progress.calls if len(args) > 1]
    assert any("upstream" in label.lower() and ("unverified" in label.lower() or "not authenticated" in label.lower()) for label in labels)


def test_partial_download_does_not_replace_installed_asset(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = b"O" * 10_000_100
    dest.write_bytes(old)
    monkeypatch.setattr("pai_studio.workers.time.sleep", lambda *_: None)
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD, declared_length=len(PAYLOAD) + 100))
    with pytest.raises(RuntimeError, match="Download failed"):
        worker._download_one(_meta(GOOD_SHA), dest)
    assert dest.read_bytes() == old
    assert not dest.with_name(dest.name + ".part").exists()


def test_failed_update_preserves_previous_runtime(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = b"V" * 10_000_100
    dest.write_bytes(old)
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD))
    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        worker._download_one(_meta("f" * 64), dest)
    assert dest.read_bytes() == old


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    target = tmp_path / "target"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "bad")
    with pytest.raises((RuntimeError, ValidationError)):
        _extract_llama_grounding_archive_safe(archive, target)
    assert not (tmp_path / "outside.txt").exists()
    with pytest.raises(ValidationError):
        validate_archive_member_path(target, "../../outside.txt")


def test_cancel_cleans_staging_file_and_preserves_previous_asset(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = b"V" * 10_000_100
    dest.write_bytes(old)
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD, on_first_read=worker.cancel_event.set))
    assert worker._download_one(_meta(GOOD_SHA), dest) is False
    assert dest.read_bytes() == old
    assert not dest.with_name(dest.name + ".part").exists()


def test_manifest_records_hash_source_and_file_metadata(tmp_path, monkeypatch):
    worker = _WorkerHarness.build(tmp_path)
    dest = worker.target_dir / "asset.gguf"
    monkeypatch.setattr("pai_studio.workers.urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(PAYLOAD))
    assert worker._download_one(_meta(""), dest) is True
    manifest_path = worker.target_dir / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["files"]["asset.gguf"]
    assert record["size"] == len(PAYLOAD)
    assert isinstance(record["mtime_ns"], int)
    assert record["sha256"] == GOOD_SHA
    assert record["sha256_source"] == "computed-locally-no-upstream-pin"


def test_invalid_runtime_archive_preserves_previous_installation(tmp_path):
    archive = tmp_path / "replacement.zip"
    target = tmp_path / "runtime"
    target.mkdir()
    old_server = target / "llama-server.exe"
    old_server.write_bytes(b"working-old-runtime")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("readme.txt", "not a server")
    with pytest.raises(RuntimeError, match="llama-server.exe"):
        _extract_llama_grounding_archive_safe(archive, target)
    assert old_server.read_bytes() == b"working-old-runtime"
    assert not (target / "readme.txt").exists()


def test_failed_post_install_validation_restores_previous_runtime(tmp_path):
    target = tmp_path / "runtime"
    staging = tmp_path / "runtime.staging"
    target.mkdir()
    staging.mkdir()
    (target / "sd-server.exe").write_bytes(b"previous-good-runtime")
    (staging / "sd-server.exe").write_bytes(b"new-runtime")

    def reject_install(installed_path):
        assert (installed_path / "sd-server.exe").read_bytes() == b"new-runtime"
        raise RuntimeError("missing required CUDA dependency")

    with pytest.raises(RuntimeError, match="missing required CUDA dependency"):
        install_directory_transactionally(staging, target, validator=reject_install)
    assert (target / "sd-server.exe").read_bytes() == b"previous-good-runtime"
    assert not staging.exists()
