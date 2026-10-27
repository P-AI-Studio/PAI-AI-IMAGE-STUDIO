# -*- coding: utf-8 -*-
"""
Portable workspace directory layout, path encoding, role validation, and LoRA metadata inspection.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

def app_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "pai_studio" else here

APP_DIR = app_directory()
# Fully portable layout: all settings, models, runtime downloads, temporary files,
# caches and default image inputs/outputs remain under ./data beside the source/EXE.
# The app deliberately does not use AppData, Roaming, home directories or the registry.
USER_DATA_ROOT = APP_DIR / "data"

def configure_workspace() -> tuple[Path, dict[str, Path]]:
    # The fallback remains under APP_DIR; the application never silently writes user data
    # to AppData, Roaming, ~/.local/share or another external profile location.
    candidates = [USER_DATA_ROOT, APP_DIR]
    chosen = None
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".pai_write_test"
            probe.write_text("write-test", encoding="utf-8")
            probe.unlink()
            chosen = root
            break
        except OSError:
            continue
    if chosen is None:
        raise OSError("Could not create a writable P.AI workspace in the install or user-data directory.")
    images = chosen / "images"
    paths = {
        "generated": images / "generated",
        "edited": images / "edited",
        "input": images / "input",
        "output": images / "output",
        "closeups": images / "closeups",
        "outpaint": images / "outpaint",
        "cache": chosen / "cache",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return chosen, paths

DATA_ROOT, WORKSPACE_DIRS = configure_workspace()
GENERATED_DIR = WORKSPACE_DIRS["generated"]
EDITED_DIR = WORKSPACE_DIRS["edited"]
INPUT_DIR = WORKSPACE_DIRS["input"]
OUTPUT_DIR = WORKSPACE_DIRS["output"]
CLOSEUPS_DIR = WORKSPACE_DIRS["closeups"]
OUTPAINT_DIR = WORKSPACE_DIRS["outpaint"]
CACHE_DIR = WORKSPACE_DIRS["cache"]
SETTINGS_FILE = DATA_ROOT / "settings.json"
MODEL_DIR = DATA_ROOT / "models" / "Qwen-Image-2.1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR = DATA_ROOT / "loras"
LORA_DIR.mkdir(parents=True, exist_ok=True)

# Close-up semantic grounding reuses the text encoder + vision projector selected in Model Catalog.
# Only the small llama.cpp server runtime is installed separately; no second VLM weights are downloaded.
LLAMA_GROUNDING_RUNTIME_DIR = DATA_ROOT / "runtime" / "llama-cpp-grounding"
LLAMA_GROUNDING_ARCHIVE_FILENAME = "llama-b11744-bin-win-cuda-12.4-x64.zip"
GROUNDING_RUNTIME_ASSETS: list[dict[str, Any]] = [
    {
        "id": "llama_cpp_cuda_runtime", "role": "grounding_runtime",
        "title": "llama.cpp Windows x64 CUDA 12 GPU runtime · b11744 (~45 MB)",
        "filename": LLAMA_GROUNDING_ARCHIVE_FILENAME,
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11744/llama-b11744-bin-win-cuda-12.4-x64.zip",
        "sha256": "", "size_hint": 45_000_000, "license": "MIT",
        "license_url": "https://github.com/ggml-org/llama.cpp",
        "source": "ggml-org/llama.cpp release b11744 (CUDA 12.4)", "default": True,
        "description": "Local NVIDIA CUDA GPU runtime for llama.cpp. Close-up runs the selected Model Catalog text encoder and vision projector on the GPU.",
    },
]

def llama_grounding_runtime_present() -> bool:
    if not LLAMA_GROUNDING_RUNTIME_DIR.is_dir():
        return False
    return any(p.is_file() for p in LLAMA_GROUNDING_RUNTIME_DIR.rglob("llama-server.exe"))


def _extract_llama_grounding_archive_safe(archive: Path, root: Path) -> None:
    """Validate a llama.cpp ZIP in staging, then swap it into place transactionally."""
    try:
        from .grounding import _shutdown_qwen3vl_grounder
        _shutdown_qwen3vl_grounder()
    except Exception:
        pass
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, timeout=5, check=False, creationflags=creationflags,
            )
            time.sleep(0.25)
        except Exception:
            pass

    from .validation import validate_archive_member_path

    archive = Path(archive).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.extracting-{uuid.uuid4().hex}"
    backup = root.parent / f".{root.name}.backup-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    moved_old = False
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"The downloaded llama.cpp runtime archive failed CRC verification at {bad}")
            entries = zf.infolist()
            if not entries:
                raise RuntimeError("The llama.cpp runtime archive is empty.")
            for info in entries:
                destination = validate_archive_member_path(staging, info.filename)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src_h, destination.open("wb") as dst_h:
                    shutil.copyfileobj(src_h, dst_h, length=4 * 1024 * 1024)
        if not any(path.is_file() for path in staging.rglob("llama-server.exe")):
            raise RuntimeError("The validated runtime archive does not contain llama-server.exe.")

        if root.exists():
            try:
                os.replace(root, backup)
                moved_old = True
            except PermissionError:
                # If Windows still holds a lock on llama-server-impl.dll in an already-valid
                # runtime directory, keep the working installation instead of failing with Errno 13.
                if any(path.is_file() for path in root.rglob("llama-server.exe")):
                    return
                raise
        try:
            os.replace(staging, root)
        except Exception:
            if moved_old and backup.exists() and not root.exists():
                os.replace(backup, root)
                moved_old = False
            raise
        if not any(path.is_file() for path in root.rglob("llama-server.exe")):
            # Defensive post-swap check: restore the prior install if the staged payload
            # differs from what was validated before the directory rename.
            shutil.rmtree(root, ignore_errors=True)
            if moved_old and backup.exists():
                os.replace(backup, root)
                moved_old = False
            raise RuntimeError("llama-server.exe was not present after installing the validated runtime.")
        if moved_old and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
            moved_old = False
    except Exception:
        if moved_old and backup.exists() and not root.exists():
            try:
                os.replace(backup, root)
                moved_old = False
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        # Never delete a backup if restore failed; it may be the only working copy.

def install_directory_transactionally(staging: Path, root: Path, validator=None) -> Path:
    """Atomically install a staged directory and restore the prior directory on validation failure."""
    staging = Path(staging).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    if not staging.is_dir():
        raise RuntimeError("Runtime staging directory is missing.")
    if staging.parent != root.parent:
        raise RuntimeError("Runtime staging directory must be a sibling of the install directory for atomic replacement.")
    backup = root.parent / f".{root.name}.backup-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if root.exists():
            os.replace(root, backup)
            moved_old = True
        os.replace(staging, root)
        if validator is not None:
            validator(root)
        if moved_old and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
            moved_old = False
        return root
    except Exception:
        if root.exists():
            if root.is_dir():
                shutil.rmtree(root, ignore_errors=True)
            else:
                try:
                    root.unlink(missing_ok=True)
                except OSError:
                    pass
        if moved_old and backup.exists():
            try:
                os.replace(backup, root)
                moved_old = False
            except OSError:
                # Preserve the backup rather than deleting the only known-good runtime.
                pass
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# Backward-compatible helper name. The 'bundle' now means runtime-only; model weights are from Model Catalog.
def qwen_grounding_bundle_present() -> bool:
    return llama_grounding_runtime_present()

def resource_path(relative: str) -> Path:
    """Resolve a bundled resource in source runs and PyInstaller builds."""
    root = Path(getattr(sys, "_MEIPASS", str(APP_DIR)))
    return root / relative

def _relative_if_inside(path: Path, root: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None

def _model_role_mismatch(role: str, filename: str) -> str:
    """Return a clear diagnostic when a local checkpoint is obviously assigned to the wrong role.

    A filename cannot prove that arbitrary weights are compatible, but it can catch common
    mistakes such as using the Qwen3-VL text encoder as the diffusion denoiser.
    Recognizes `convrot` tags (including filenames ending with `convrot` / `int8-convrot` / `int8convrot`).
    """
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1].lower().replace("_", "-")
    stem = name.rsplit(".", 1)[0] if "." in name else name
    has_convrot_tag = stem.endswith("convrot") or "convrot" in name
    is_qwen3vl = "qwen3vl" in name or "qwen3-vl" in name
    is_mmproj = "mmproj" in name or "projector" in name
    is_vae = "vae" in name or ("texture-fix" in name and "vae" in name)
    is_explicit_text_encoder = (
        is_qwen3vl
        or "text-encoder" in name
        or "textencoder" in name
        or "-te-" in name
        or name.startswith("te-")
        or ("qwen-image-2.1-turbo-uncensored" in name and not is_mmproj)
        or ("qwen-image-2.1-turbo-abliterated-uncensored" in name and not is_mmproj)
        or ("heretic" in name and not is_mmproj and not is_vae)
    )
    is_turbo_denoiser = (
        ("qwen-image-2.1-turbo-ad-" in name and "uncensored" not in name)
        or (("qwen-image-2.1-turbo" in name or "rayqwen" in name or "sakura-image-2.1-turbo" in name)
            and not is_explicit_text_encoder and not is_mmproj and not is_vae
            and (has_convrot_tag or "turbo" in name))
    )
    is_uncensored_encoder = is_explicit_text_encoder and not is_qwen3vl and "qwen-image-2.1-turbo-uncensored" not in name
    # A user-selected GGUF mmproj is allowed in the Vision Projector slot.
    if role == "vision_projector" and is_mmproj and name.endswith(".gguf"):
        return ""

    if role == "diffusion":
        if is_qwen3vl:
            return (
                f"File '{filename}' is a Qwen3-VL/text encoder, not the main image diffusion model. "
                "Choose a Qwen-Image 2.1 Turbo denoiser for the Image model, for example "
                "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors or Qwen-Image-2.1-Turbo-AD-Q4_K.gguf."
            )
        if is_mmproj or is_vae or is_uncensored_encoder or "text-encoder" in name or "textencoder" in name or "llm" in name:
            return (
                f"File '{filename}' does not match the {role} role; its name suggests a projector, VAE, or text encoder. "
                "Choose a main Qwen-Image 2.1 model in Models / Download."
            )
    elif role == "text_encoder":
        if is_mmproj or is_vae or is_turbo_denoiser:
            return (
                f"File '{filename}' is not a text encoder; it looks like a projector, VAE, or diffusion denoiser. "
                "Use a text encoder compatible with Qwen-Image 2.1 Turbo (including INT8-ConvRot Safetensors or GGUF)."
            )
    elif role == "vision_projector":
        if is_qwen3vl or has_convrot_tag:
            return (
                f"File '{filename}' belongs to the Qwen3-VL-8B-Instruct / ConvRot text-encoder family and is not a Vision Projector. "
                "Safetensors/ConvRot text encoders already include vision support and do not need mmproj. "
                "Only select mmproj when using a GGUF text encoder."
            )
        if is_vae or is_turbo_denoiser or is_uncensored_encoder:
            return (
                f"File '{filename}' is a text encoder or main model, not a Vision Projector. "
                "Vision Projector (mmproj) is optional and only used with GGUF text encoders."
            )
    elif role == "vae":
        if is_qwen3vl or is_mmproj or is_turbo_denoiser or is_uncensored_encoder:
            return (
                f"File '{filename}' is not a VAE. Use a VAE compatible with Qwen-Image 2.1."
            )
    return ""


def _text_encoder_embeds_vision(path: Path | str | None) -> bool:
    """Qwen3-VL / ConvRot safetensors layouts include vision weights; only GGUF encoders use a separate mmproj."""
    if not path:
        return False
    p = Path(str(path))
    name = p.name.lower().replace("_", "-")
    stem = p.stem.lower().replace("_", "-")
    if p.suffix.lower() not in {".safetensors", ".sft"}:
        return False
    if "mmproj" in name or "vae" in name:
        return False
    return (
        stem.endswith("convrot")
        or "convrot" in name
        or "qwen3vl" in name
        or "qwen3-vl" in name
        or "text-encoder" in name
        or "textencoder" in name
        or "qwen-image-2.1-turbo-uncensored" in name
        or "heretic" in name
        or "abliterated" in name
    )


def encode_saved_path(value: str) -> str:
    if not value:
        return value
    try:
        p = Path(value).expanduser().resolve()
        rel = _relative_if_inside(p, APP_DIR)
        if rel is not None:
            return "${INSTALL}" + ("/" + rel if rel else "")
        rel = _relative_if_inside(p, USER_DATA_ROOT)
        if rel is not None:
            return "${USERDATA}" + ("/" + rel if rel else "")
    except (OSError, RuntimeError):
        pass
    return value

def decode_saved_path(value: str) -> str:
    if value.startswith("${INSTALL}"):
        suffix = value[len("${INSTALL}"):].lstrip("/\\")
        return str((APP_DIR / suffix).resolve())
    if value.startswith("${USERDATA}"):
        suffix = value[len("${USERDATA}"):].lstrip("/\\")
        return str((USER_DATA_ROOT / suffix).resolve())
    return value

_LOCAL_BBOX_CACHE: "OrderedDict[tuple[Any, ...], tuple[float, float, float, float]]" = OrderedDict()
_LOCAL_BBOX_CACHE_LIMIT = 64


def prepare_native_prompt(prompt: str, log_callback=None) -> str:
    """Preserve prompt text exactly; the diffusion text-conditioning encoder is not a translator."""
    text = str(prompt or "")
    if text.strip() and re.search(r"[\u0600-\u06ff\u0750-\u077f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        if log_callback:
            log_callback(
                "Prompt is passed to the native image model unchanged. Its conditioning encoder does not provide "
                "standalone translation; no translator service or extra language model is used."
            )
    return text



def inspect_lora_file(path: str | Path) -> dict[str, Any]:
    """Read lightweight LoRA details from safe JSON sidecars / safetensors headers only.

    Tensor payloads are never deserialized. Embedded metadata is a best-effort source; not
    every LoRA publisher includes a base model, trigger words or a description.
    """
    file_path = Path(str(path or "")).expanduser()
    result: dict[str, Any] = {
        "path": str(file_path), "exists": False, "file_name": file_path.name,
        "size_bytes": 0, "size_label": "", "base_model": "Not specified",
        "creator": "Not specified", "description": "No embedded description found.",
        "trigger_words": [], "example_prompts": [], "metadata_found": False, "source": "Local file",
    }
    if not file_path.is_file():
        return result
    try:
        stat = file_path.stat()
        result["exists"] = True
        result["size_bytes"] = stat.st_size
        if stat.st_size >= 1024 ** 3:
            result["size_label"] = f"{stat.st_size / (1024 ** 3):.2f} GiB"
        elif stat.st_size >= 1024 ** 2:
            result["size_label"] = f"{stat.st_size / (1024 ** 2):.1f} MiB"
        else:
            result["size_label"] = f"{stat.st_size / 1024:.0f} KiB"
    except OSError:
        pass

    records: list[dict[str, Any]] = []
    # Sidecar conventions used by local LoRA managers, plus the common CivitAI info filename.
    sidecars = (
        file_path.with_suffix(file_path.suffix + ".json"),
        file_path.with_suffix(".json"),
        file_path.with_name(file_path.name + ".civitai.info"),
        file_path.with_name(file_path.stem + ".metadata.json"),
    )
    for sidecar in sidecars:
        if not sidecar.is_file():
            continue
        try:
            if sidecar.stat().st_size > 4 * 1024 * 1024:
                continue
            obj = json.loads(sidecar.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                records.append(obj)
        except (OSError, ValueError, TypeError):
            continue

    if file_path.suffix.lower() in {".safetensors", ".sft"}:
        try:
            with file_path.open("rb") as handle:
                raw_len = handle.read(8)
                if len(raw_len) == 8:
                    header_len = int.from_bytes(raw_len, "little", signed=False)
                    if 0 < header_len <= 16 * 1024 * 1024:
                        header = json.loads(handle.read(header_len).decode("utf-8", errors="strict"))
                        metadata = header.get("__metadata__", {}) if isinstance(header, dict) else {}
                        if isinstance(metadata, dict) and metadata:
                            records.append(metadata)
        except (OSError, ValueError, UnicodeError, OverflowError):
            pass

    def decode_jsonish(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if len(stripped) <= 2 * 1024 * 1024 and stripped[:1] in "{[\"":
                try:
                    return json.loads(stripped)
                except (ValueError, TypeError):
                    return value
        return value

    def walk(obj: Any, parents: tuple[str, ...] = ()):
        obj = decode_jsonish(obj)
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_str = str(key)
                yield key_str, value, parents
                yield from walk(value, parents + (key_str,))
        elif isinstance(obj, list):
            for value in obj[:500]:
                yield from walk(value, parents)

    pairs: list[tuple[str, Any, tuple[str, ...]]] = []
    for record in records:
        pairs.extend(list(walk(record)))

    def clean_text(value: Any, limit: int = 500) -> str:
        value = decode_jsonish(value)
        if isinstance(value, dict):
            for key in ("name", "title", "value", "description", "username"):
                if key in value and value[key]:
                    value = value[key]
                    break
        if isinstance(value, list):
            value = ", ".join(str(x) for x in value if x not in (None, ""))
        text_value = re.sub(r"<[^>]+>", " ", str(value or ""))
        text_value = re.sub(r"\s+", " ", text_value).strip()
        return text_value[:limit] + ("…" if len(text_value) > limit else "")

    def first_value(keys: tuple[str, ...]) -> Any:
        wanted = {key.casefold() for key in keys}
        for key, value, _parents in pairs:
            if key.casefold() in wanted and value not in (None, "", [], {}):
                return value
        return None

    base = first_value(("modelspec.architecture", "ss_base_model_version", "base_model", "baseModel", "base_model_name", "ss_sd_model_name", "base model"))
    creator = first_value(("modelspec.author", "creator", "creator_name", "author", "username", "ss_author"))
    description = first_value(("modelspec.description", "description", "ss_training_comment", "notes", "comment", "trained_description"))
    triggers = first_value(("modelspec.trigger_words", "trigger_words", "trainedwords", "trained_words", "ss_trigger_words", "activation text", "activation_text", "tags"))
    examples_value = first_value(("modelspec.prompt", "example_prompt", "example_prompts", "example prompts", "sample_prompts", "sample_prompt", "examples"))

    trigger_list: list[str] = []
    triggers = decode_jsonish(triggers)
    if isinstance(triggers, dict):
        # Some safetensors write tag frequency as {dataset: {token: count}}.
        counts: dict[str, float] = {}
        def collect_counts(item: Any):
            item = decode_jsonish(item)
            if isinstance(item, dict):
                for term, count in item.items():
                    if isinstance(count, (int, float)):
                        term_text = clean_text(term, 120)
                        if term_text:
                            counts[term_text] = counts.get(term_text, 0.0) + float(count)
                    else:
                        collect_counts(count)
            elif isinstance(item, list):
                for child in item:
                    collect_counts(child)
        collect_counts(triggers)
        trigger_list = [term for term, _count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:24]]
    elif isinstance(triggers, list):
        trigger_list = [clean_text(item, 120) for item in triggers if clean_text(item, 120)]
    elif isinstance(triggers, str):
        trigger_list = [clean_text(item, 120) for item in re.split(r"[,\n;|]", triggers) if clean_text(item, 120)]

    if not trigger_list:
        # Fall back to tag-frequency metadata that has a different key than common trigger fields.
        for key, value, _parents in pairs:
            if key.casefold() in {"ss_tag_frequency", "tag_frequency", "tagfrequency"}:
                counts: dict[str, float] = {}
                def gather(item: Any):
                    item = decode_jsonish(item)
                    if isinstance(item, dict):
                        for term, count in item.items():
                            if isinstance(count, (int, float)):
                                counts[clean_text(term, 120)] = counts.get(clean_text(term, 120), 0.0) + float(count)
                            else:
                                gather(count)
                    elif isinstance(item, list):
                        for child in item:
                            gather(child)
                gather(value)
                trigger_list = [term for term, _count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:24] if term]
                if trigger_list:
                    break

    def collect_examples(value: Any) -> list[str]:
        value = decode_jsonish(value)
        found: list[str] = []
        if isinstance(value, str):
            found.extend(clean_text(item, 360) for item in re.split(r"\n+|\s*\|\|\s*", value) if clean_text(item, 360))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    prompt = next((item.get(key) for key in ("prompt", "text", "example_prompt", "examplePrompt", "caption") if item.get(key)), None)
                    if prompt:
                        found.extend(collect_examples(prompt))
                else:
                    found.extend(collect_examples(item))
                if len(found) >= 4:
                    break
        elif isinstance(value, dict):
            for key, item in value.items():
                if any(token in str(key).casefold() for token in ("prompt", "example", "sample")):
                    found.extend(collect_examples(item))
                    if len(found) >= 4:
                        break
        return list(dict.fromkeys(item for item in found if item))[:4]

    result.update({
        "base_model": clean_text(base, 140) if base else "Not specified",
        "creator": clean_text(creator, 120) if creator else "Not specified",
        "description": clean_text(description, 600) if description else "No embedded description found.",
        "trigger_words": list(dict.fromkeys(trigger_list))[:24],
        "example_prompts": collect_examples(examples_value),
        "metadata_found": bool(records and any(value not in (None, "", [], {}) for _, value, _ in pairs)),
        "source": "Local sidecar / safetensors metadata" if records else "File only; no metadata found",
    })
    return result
