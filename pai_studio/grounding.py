# -*- coding: utf-8 -*-
"""
Open-vocabulary semantic visual grounding using the selected Model Catalog Qwen3-VL encoder
and llama.cpp loopback server, plus built-in local visual target grounding for Safetensors/ConvRot
and mmproj-free configurations.
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

from .constants import (
    APP_VERSION,
    _download_ssl_context,
)
from .workspace import (
    APP_DIR,
    CACHE_DIR,
    DATA_ROOT,
    GROUNDING_RUNTIME_ASSETS,
    LLAMA_GROUNDING_ARCHIVE_FILENAME,
    LLAMA_GROUNDING_RUNTIME_DIR,
    _LOCAL_BBOX_CACHE,
    _LOCAL_BBOX_CACHE_LIMIT,
    _extract_llama_grounding_archive_safe,
    llama_grounding_runtime_present,
)

def parse_english_prompt_translation(response_text: str, original_prompt: str = "") -> str:
    """Extract a clean translated prompt from a local text-encoder response."""
    text = str(response_text or "").strip()
    text = re.sub(r"^```(?:text|plain|english)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" \t\r\n\"'“”‘’`")
    text = re.sub(r"^(?:english translation|translated prompt|translation)\s*[:：-]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n\"'“”‘’`")
    # Never substitute the original automatically here: an empty result is a
    # failure signal so the caller can explicitly log and fall back safely.
    letters = [ch for ch in text if ch.isalpha()]
    # If the response contains no Latin letters, the model likely echoed a
    # non-English prompt instead of translating it. This still permits quoted
    # in-image text in the source language when the surrounding prompt is English.
    if not letters or not any("a" <= ch.casefold() <= "z" for ch in letters):
        return ""
    return text[:4000].strip()




_FA_TO_EN_PHRASES: tuple[tuple[str, str], ...] = (
    ("لباسش را", "the clothing"),
    ("لباس را", "the clothing"),
    ("پس زمینه را", "the background"),
    ("پس‌زمینه را", "the background"),
    ("نور طبیعی گرم", "warm natural light"),
    ("نور طبیعی", "natural light"),
    ("پرتره سینمایی", "cinematic portrait"),
    ("با کیفیت بالا", "high quality, detailed"),
    ("جزئیات بالا", "high detail"),
    ("چشم چپ", "left eye"),
    ("چشم راست", "right eye"),
    ("هر دو چشم", "both eyes"),
    ("لب بالا", "upper lip"),
    ("لب پایین", "lower lip"),
    ("دست چپ", "left hand"),
    ("دست راست", "right hand"),
    ("پای چپ", "left leg"),
    ("پای راست", "right leg"),
    ("سینه چپ", "left breast"),
    ("سینه راست", "right breast"),
    ("تغییر بده", "change"),
    ("عوض کن", "replace"),
    ("حذف کن", "remove"),
    ("اضافه کن", "add"),
    ("بهتر کن", "enhance"),
)

_FA_TO_EN_WORDS: dict[str, str] = {
    "یک": "a",
    "پرتره": "portrait",
    "سینمایی": "cinematic",
    "با": "with",
    "نور": "light",
    "طبیعی": "natural",
    "گرم": "warm",
    "سرد": "cool",
    "نرم": "soft",
    "زن": "woman",
    "دختر": "young woman",
    "مرد": "man",
    "پسر": "young man",
    "صورت": "face",
    "چهره": "face",
    "چشم": "eye",
    "چشمها": "eyes",
    "چشم‌ها": "eyes",
    "ابرو": "eyebrows",
    "مژه": "eyelashes",
    "بینی": "nose",
    "لب": "lips",
    "لبها": "lips",
    "لب‌ها": "lips",
    "دهان": "mouth",
    "دندان": "teeth",
    "لبخند": "smile",
    "گردن": "neck",
    "گوش": "ear",
    "مو": "hair",
    "موها": "hair",
    "پوست": "skin",
    "بدن": "body",
    "سینه": "chest and breasts",
    "سینهها": "breasts",
    "سینه‌ها": "breasts",
    "پستان": "breasts",
    " نوک": "nipple",
    "شکم": "stomach and abdomen",
    "ناف": "navel",
    "کمر": "waist",
    "باسن": "buttocks and hips",
    "کون": "buttocks",
    "ران": "thighs",
    "پا": "legs and feet",
    "پاها": "legs and feet",
    "دست": "hand",
    "دستها": "hands",
    "دست‌ها": "hands",
    "انگشت": "fingers",
    "ناخن": "nails",
    "کس": "pussy",
    "واژن": "vagina",
    "لباس": "clothing",
    "پیراهن": "dress",
    "مانتو": "coat",
    "شلوار": "pants",
    "کفش": "shoes",
    "کلاه": "hat",
    "عینک": "glasses",
    "گردنبند": "necklace",
    "انگشتر": "ring",
    "گوشواره": "earring",
    "ساعت": "watch",
    "قرمز": "red",
    "آبی": "blue",
    "سبز": "green",
    "زرد": "yellow",
    "سیاه": "black",
    "مشکی": "black",
    "سفید": "white",
    "طلایی": "golden",
    "نقره‌ای": "silver",
    "صورتی": "pink",
    "بنفش": "purple",
    "قهوه‌ای": "brown",
    "در": "in",
    "روی": "on",
    "کنار": "beside",
    "ساحل": "beach",
    "باغ": "garden",
    "اتاق": "room",
    "خیابان": "street",
    "جنگل": "forest",
    "شب": "night",
    "روز": "daylight",
    "خورشید": "sunlight",
    "باران": "rain",
    "برف": "snow",
    "گربه": "cat",
    "سگ": "dog",
    "گل": "flower",
    "ماشین": "car",
    "زیبا": "beautiful",
    "جذاب": "attractive",
    "واقعی": "photorealistic",
    "شفاف": "clear and sharp",
    "کن": "",
    "بده": "",
    "را": "",
    "به": "to",
    "رنگ": "color",
}


def translate_prompt_non_gguf_fallback(prompt_text: str) -> str:
    """Offline fallback translation for non-English prompts/targets when no GGUF encoder is on disk."""
    text = str(prompt_text or "").strip()
    if not text:
        return ""
    if not any(ord(ch) > 127 for ch in text):
        return text
    working = text.replace("\u200c", " ")
    for src_phrase, dst_phrase in _FA_TO_EN_PHRASES:
        working = working.replace(src_phrase, f" {dst_phrase} ")
    tokens = re.split(r"(\s+|[,.;:!؟?،]+)", working)
    out_tokens: list[str] = []
    for tok in tokens:
        clean = tok.strip()
        if not clean:
            out_tokens.append(tok)
            continue
        if clean in {"،", "؛"}:
            out_tokens.append(",")
            continue
        if clean == "؟":
            out_tokens.append("?")
            continue
        mapped = _FA_TO_EN_WORDS.get(clean)
        if mapped is None and clean.endswith(("ها", "های", "ان")):
            stem = re.sub(r"(ها|های|ان)$", "", clean).strip()
            mapped = _FA_TO_EN_WORDS.get(stem)
        if mapped is None and clean.endswith(("ش", "م", "ت")) and len(clean) > 2:
            stem = clean[:-1]
            mapped = _FA_TO_EN_WORDS.get(stem)
        if mapped is not None:
            if mapped:
                out_tokens.append(mapped)
        elif not any(ord(ch) > 127 for ch in clean):
            out_tokens.append(clean)
    result = re.sub(r"\s+", " ", " ".join(t for t in out_tokens if t.strip())).strip(" ,.")
    return parse_english_prompt_translation(result, text) or text


def parse_qwen3vl_grounding_response(response_text: str) -> Optional[tuple[float, float, float, float]]:
    """Parse a model bounding box into finite normalized XYXY coordinates.

    The model normally returns coordinates in [0, 1000]. Fractional [0, 1]
    coordinates and the documented spatial-token form are also accepted. Invalid,
    inverted, degenerate, out-of-range and near-full-frame boxes are rejected.
    This function stays self-contained for release-smoke source-extraction tests.
    """
    text = str(response_text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        pass
    for match in re.finditer(r"\{[^{}]*\}|\[[^\[\]]*\]", text, flags=re.DOTALL):
        try:
            candidates.append(json.loads(match.group(0)))
        except (json.JSONDecodeError, TypeError):
            continue

    boxes: list[list[float]] = []
    def collect(obj):
        if isinstance(obj, dict):
            found = obj.get("found", True)
            raw = obj.get("bbox_2d", obj.get("bbox", obj.get("box")))
            if found is False or str(found).strip().lower() in {"false", "no", "not found"}:
                return
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                try:
                    if any(isinstance(v, bool) for v in raw[:4]):
                        return
                    boxes.append([float(v) for v in raw[:4]])
                except (ValueError, TypeError, OverflowError):
                    pass
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    collect(value)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)
    for candidate in candidates:
        collect(candidate)
        if boxes:
            break

    if not boxes:
        token_match = re.search(
            r"(?:<\|box_start\|>|<box>)\s*\(?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*\)?\s*[,; ]+\(?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*\)?\s*(?:<\|box_end\|>|</box>)?",
            text, flags=re.IGNORECASE,
        )
        if token_match:
            boxes.append([float(token_match.group(i)) for i in range(1, 5)])
    if not boxes:
        return None

    values = boxes[0]
    if len(values) != 4 or any(value != value or value in (float("inf"), float("-inf")) for value in values):
        return None
    if max(abs(value) for value in values) <= 1.0:
        values = [value * 1000.0 for value in values]
    # Do not clamp arbitrary malformed output into an apparently valid box.
    if any(value < 0.0 or value > 1000.0 for value in values):
        return None
    x0, y0, x1, y1 = values
    if not (x0 < x1 and y0 < y1):
        return None
    area = ((x1 - x0) * (y1 - y0)) / 1_000_000.0
    if area > 0.96:
        return None
    return (x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0)


@dataclass(frozen=True)
class GroundingParseResult:
    status: str
    bbox_xyxy: Optional[tuple[float, float, float, float]]
    reason_code: str


def parse_qwen3vl_grounding_response_detailed(response_text: str) -> GroundingParseResult:
    """Return a safe, classifiable parser outcome without retaining raw model text."""
    text = str(response_text or "").strip()
    if not text:
        return GroundingParseResult("invalid_response", None, "empty_response")
    candidates: list[Any] = []
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        candidates.append(json.loads(cleaned))
    except (json.JSONDecodeError, TypeError):
        pass
    for match in re.finditer(r"\{[^{}]*\}|\[[^\[\]]*\]", cleaned, flags=re.DOTALL):
        try:
            candidates.append(json.loads(match.group(0)))
        except (json.JSONDecodeError, TypeError):
            continue

    explicit_not_found = False
    raw_boxes: list[Any] = []
    def inspect(obj: Any) -> None:
        nonlocal explicit_not_found
        if isinstance(obj, dict):
            found = obj.get("found", True)
            if found is False or str(found).strip().lower() in {"false", "no", "not found"}:
                explicit_not_found = True
            raw = obj.get("bbox_2d", obj.get("bbox", obj.get("box")))
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                raw_boxes.append(raw[:4])
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    inspect(value)
        elif isinstance(obj, list):
            for value in obj:
                inspect(value)
    for item in candidates:
        inspect(item)
    if explicit_not_found and not raw_boxes:
        return GroundingParseResult("not_found", None, "target_not_found")

    box = parse_qwen3vl_grounding_response(text)
    if box is not None:
        return GroundingParseResult("found", box, "ok")
    if raw_boxes:
        try:
            vals = [float(v) for v in raw_boxes[0]]
            if len(vals) == 4 and all(v == v and v not in (float("inf"), float("-inf")) for v in vals):
                if max(abs(v) for v in vals) <= 1.0:
                    vals = [v * 1000.0 for v in vals]
                if all(0 <= v <= 1000 for v in vals) and vals[0] < vals[2] and vals[1] < vals[3]:
                    area = ((vals[2] - vals[0]) * (vals[3] - vals[1])) / 1_000_000.0
                    if area > 0.96:
                        return GroundingParseResult("invalid_response", None, "near_full_frame")
        except (ValueError, TypeError, OverflowError):
            pass
        return GroundingParseResult("invalid_response", None, "invalid_box_geometry")
    if explicit_not_found:
        return GroundingParseResult("not_found", None, "target_not_found")
    return GroundingParseResult("invalid_response", None, "bbox_missing")


def _sanitize_target_description_candidates(candidates, target: str, max_candidates: int = 3) -> list[str]:
    """Keep a tiny list of safe, distinct model-generated descriptions."""
    try:
        cap = max(0, min(3, int(max_candidates)))
    except (TypeError, ValueError, OverflowError):
        return []
    if cap <= 0 or isinstance(candidates, (str, bytes)) or not isinstance(candidates, (list, tuple)):
        return []
    original = str(target or "").strip().casefold()
    seen = {original} if original else set()
    result: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if not text or len(text) > 180 or any(ord(char) < 32 or ord(char) == 127 for char in text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= cap:
            break
    return result


def parse_target_description_candidates(response_text: str, target: str, max_candidates: int = 3) -> list[str]:
    """Parse a strict JSON array of concise semantic descriptions; fail closed."""
    text = str(response_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    return _sanitize_target_description_candidates(payload, target, max_candidates)


def resolve_closeup_grounding_gguf_pair(
    model_path: Path | str | None = None,
    mmproj_path: Path | str | None = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """Find an installed GGUF Qwen3-VL/Uncensored encoder + GGUF mmproj on disk for Close-up grounding.

    Even when the user selects an INT8-ConvRot .safetensors text encoder (and sets Vision Projector
    to No projector) for image diffusion in sd-server.exe, Close-up can reuse any compatible GGUF
    text encoder and GGUF mmproj already present in the model folder or a sibling P.AI installation.
    """
    cand_model = Path(str(model_path)).expanduser() if model_path and str(model_path).strip() else None
    cand_mmproj = Path(str(mmproj_path)).expanduser() if mmproj_path and str(mmproj_path).strip().lower() not in {"", "none"} else None

    if cand_model is not None and cand_model.suffix.lower() == ".gguf" and cand_model.is_file():
        gguf_model: Optional[Path] = cand_model.resolve()
    else:
        gguf_model = None

    if cand_mmproj is not None and cand_mmproj.suffix.lower() == ".gguf" and cand_mmproj.is_file():
        gguf_mmproj: Optional[Path] = cand_mmproj.resolve()
    else:
        gguf_mmproj = None

    if gguf_model is not None and gguf_mmproj is not None:
        return gguf_model, gguf_mmproj

    search_dirs: list[Path] = []
    if cand_model is not None:
        try:
            parent_dir = cand_model.resolve().parent if cand_model.exists() else cand_model.parent
            if parent_dir.is_dir():
                search_dirs.append(parent_dir)
        except OSError:
            pass
    default_model_dir = DATA_ROOT / "models" / "Qwen-Image-2.1"
    if default_model_dir.is_dir() and default_model_dir not in search_dirs:
        search_dirs.append(default_model_dir)
    for root in (APP_DIR.parent, APP_DIR.parent.parent, Path.home() / "Downloads"):
        try:
            for proj in root.glob("PAI_AI_IMAGE_STUDIO*"):
                if proj.is_dir():
                    m_dir = proj / "data" / "models" / "Qwen-Image-2.1"
                    if m_dir.is_dir() and m_dir not in search_dirs:
                        search_dirs.append(m_dir)
                    for sub_m in proj.glob("**/data/models/Qwen-Image-2.1"):
                        if sub_m.is_dir() and sub_m not in search_dirs:
                            search_dirs.append(sub_m)
        except OSError:
            continue

    gguf_files: list[Path] = []
    seen: set[str] = set()
    for sdir in search_dirs:
        try:
            for p in sdir.rglob("*.gguf"):
                if p.is_file():
                    key = str(p.resolve()).casefold()
                    if key not in seen:
                        seen.add(key)
                        gguf_files.append(p.resolve())
        except OSError:
            continue

    if gguf_model is None:
        from .workspace import _model_role_mismatch
        # Prefer Qwen3-VL / Abliterated / Heretic / Sakura / Turbo-Uncensored GGUF encoders and exclude diffusion denoisers
        for matcher in (
            lambda p, n: ("qwen3-vl" in n or "qwen3vl" in n) and ("abliterated" in n or "heretic" in n) and "mmproj" not in n and "projector" not in n and _model_role_mismatch("text_encoder", p.name) == "",
            lambda p, n: "sakura-textencoder" in n and "mmproj" not in n and _model_role_mismatch("text_encoder", p.name) == "",
            lambda p, n: ("qwen3vl-8b-instruct" in n or "qwen3-vl-8b-instruct" in n) and "mmproj" not in n and "projector" not in n and _model_role_mismatch("text_encoder", p.name) == "",
            lambda p, n: "qwen-image-2.1-turbo-uncensored" in n and "mmproj" not in n and "projector" not in n and _model_role_mismatch("text_encoder", p.name) == "",
            lambda p, n: ("qwen3vl" in n or "qwen3-vl" in n or "text-encoder" in n or "textencoder" in n) and "mmproj" not in n and "projector" not in n and _model_role_mismatch("text_encoder", p.name) == "",
        ):
            found = next((p for p in gguf_files if matcher(p, p.name.lower().replace("_", "-"))), None)
            if found is not None:
                gguf_model = found
                break

    if gguf_mmproj is None:
        llm_name = gguf_model.name.lower().replace("_", "-") if gguf_model is not None else ""
        if "qwen-image-2.1-turbo-uncensored" in llm_name:
            gguf_mmproj = next((p for p in gguf_files if "mmproj-qwen-image-2.1-turbo-uncensored" in p.name.lower().replace("_", "-")), None)
        elif "qwen3vl-8b-instruct" in llm_name:
            gguf_mmproj = next((p for p in gguf_files if "mmproj-qwen3vl-8b-instruct" in p.name.lower().replace("_", "-")), None)
        if gguf_mmproj is None:
            gguf_mmproj = next(
                (p for p in gguf_files if "mmproj" in p.name.lower() and any(k in p.name.lower().replace("_", "-") for k in ("qwen3vl", "owen3vl", "qwen-image", "qwen3-vl"))),
                None,
            )

    return gguf_model, gguf_mmproj


def locate_target_bbox_non_gguf(
    image: Image.Image,
    target: str,
    log_callback=None,
) -> GroundingParseResult:
    """Locate a visual target bounding box locally when no GGUF/mmproj pair is installed on disk.

    Uses strict YCbCr + RGB skin-tone segmentation, central subject contour detection,
    optional OpenCV face cascade anchoring, and tight anatomical/detail boxes so
    requested Close-up zoom factors are honored accurately.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    thumb = rgb.copy()
    thumb.thumbnail((320, 320), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 4 or w < 4:
        return GroundingParseResult("found", (0.40, 0.40, 0.60, 0.60), "ok")

    # 1. Foreground vs. border background contrast
    bw = max(1, min(w // 12, 16))
    bh = max(1, min(h // 12, 16))
    border_pixels = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(border_pixels, axis=0)
    fg_dist = np.linalg.norm(arr - bg_color[None, None, :], axis=2)
    fg_max = float(np.percentile(fg_dist, 98))
    fg_norm = np.clip(fg_dist / max(12.0, fg_max), 0.0, 1.0)

    # 2. Local gradient / detail energy
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    gx[:, 1:-1] = np.linalg.norm(arr[:, 2:, :] - arr[:, :-2, :], axis=2) * 0.5
    gy[1:-1, :] = np.linalg.norm(arr[2:, :, :] - arr[:-2, :, :], axis=2) * 0.5
    grad = np.hypot(gx, gy)
    grad_max = float(np.percentile(grad, 98))
    grad_norm = np.clip(grad / max(8.0, grad_max), 0.0, 1.0)

    # 3. Strict YCbCr + RGB skin-tone likelihood
    r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    y_luma = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    cb = 128.0 - 0.168736 * r_ch - 0.331264 * g_ch + 0.5 * b_ch
    cr = 128.0 + 0.5 * r_ch - 0.418688 * g_ch - 0.081312 * b_ch
    rgb_skin_gate = ((r_ch > g_ch) & (r_ch > b_ch) & ((r_ch - np.minimum(g_ch, b_ch)) > 10.0) & (y_luma > 30.0)).astype(np.float32)
    skin_likelihood = (
        np.exp(-0.5 * (((cb - 105.0) / 16.0) ** 2 + ((cr - 152.0) / 16.0) ** 2))
        * (0.25 + 0.75 * rgb_skin_gate)
    ).astype(np.float32)

    # 4. Estimate human body / foreground subject bounds accurately
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = yy / max(1.0, float(h - 1))
    xn = xx / max(1.0, float(w - 1))
    center_prior = np.exp(-0.5 * (((xn - 0.5) / 0.35) ** 2 + ((yn - 0.52) / 0.44) ** 2))

    skin_mask = (skin_likelihood * (0.4 + 0.6 * center_prior)) >= 0.28
    if int(np.count_nonzero(skin_mask)) >= max(12, int(h * w * 0.015)):
        ys, xs = np.where(skin_mask)
        sx0 = float(np.percentile(xs, 3)) / float(w)
        sy0 = float(np.percentile(ys, 2)) / float(h)
        sx1 = float(np.percentile(xs, 97)) / float(w)
        sy1 = float(np.percentile(ys, 98)) / float(h)
    else:
        subject_map = (0.45 * fg_norm + 0.35 * grad_norm + 0.20 * skin_likelihood) * (0.40 + 0.60 * center_prior)
        subj_thresh = float(np.percentile(subject_map, 75))
        subj_mask = subject_map >= max(0.08, subj_thresh)
        if np.any(subj_mask):
            ys, xs = np.where(subj_mask)
            sx0 = float(np.percentile(xs, 5)) / float(w)
            sy0 = float(np.percentile(ys, 4)) / float(h)
            sx1 = float(np.percentile(xs, 95)) / float(w)
            sy1 = float(np.percentile(ys, 96)) / float(h)
        else:
            sx0, sy0, sx1, sy1 = 0.20, 0.08, 0.80, 0.92

    sx0 = max(0.02, min(0.80, sx0))
    sy0 = max(0.02, min(0.80, sy0))
    sx1 = max(sx0 + 0.12, min(0.98, sx1))
    sy1 = max(sy0 + 0.12, min(0.98, sy1))
    subj_w = sx1 - sx0
    subj_h = sy1 - sy0
    subj_cx = 0.5 * (sx0 + sx1)

    # 5. Apply semantic spatial zone & feature weights within the detected subject
    raw_target = str(target or "").strip()
    translated_target = translate_prompt_non_gguf_fallback(raw_target) if any(ord(ch) > 127 for ch in raw_target) else raw_target
    t_lower = f"{raw_target} {translated_target}".strip().casefold()
    upper_terms = ("face", "head", "eye", "eyes", "lip", "lips", "mouth", "nose", "hair", "ear", "portrait", "expression", "صورت", "چشم", "لب", "مو", "سر", "چهره")
    upper_mid_terms = ("chest", "breast", "breasts", "boobs", "tits", "nipple", "nipples", "cleavage", "torso", "neck", "shoulder", "collarbone", "سینه", "گردن", "شانه", "بالاتنه")
    groin_terms = ("pussy", "vagina", "vulva", "crotch", "groin", "pelvis", "pubic", "panties", "bikini")
    hip_waist_terms = ("hip", "hips", "waist", "belly", "navel", "ass", "butt", "buttocks", "glutes", "thigh", "thighs", "کمر", "باسن", "ران", "شکم")
    bottom_terms = ("foot", "feet", "toe", "toes", "ankle", "shoe", "shoes", "heel", "heels", "calf", "leg", "legs", "پا", "کفش", "ساق")
    hand_terms = ("hand", "hands", "finger", "fingers", "ring", "watch", "wrist", "دست", "انگشت", "ساعت", "انگشتر")

    if any(term in t_lower for term in upper_terms):
        target_y = sy0 + 0.16 * subj_h
        sigma_y = max(0.07, 0.13 * subj_h)
        sigma_x = max(0.09, 0.20 * subj_w)
        feature_mix = 0.50 * grad_norm + 0.40 * skin_likelihood + 0.10 * fg_norm
        rel_box_w, rel_box_h = 0.12, 0.12
    elif any(term in t_lower for term in upper_mid_terms):
        target_y = sy0 + 0.36 * subj_h
        sigma_y = max(0.08, 0.14 * subj_h)
        sigma_x = max(0.10, 0.22 * subj_w)
        feature_mix = 0.55 * skin_likelihood + 0.35 * grad_norm + 0.10 * fg_norm
        rel_box_w, rel_box_h = 0.14, 0.14
    elif any(term in t_lower for term in groin_terms):
        # Groin/pelvis sits at ~73% down the visible body span, along the central body axis
        # where skin meets high local crease/shadow gradient.
        target_y = sy0 + 0.73 * subj_h
        sigma_y = max(0.06, 0.11 * subj_h)
        sigma_x = max(0.06, 0.15 * subj_w)
        # Compute local horizontal center of skin in the lower-mid band [0.60..0.85] of the body
        band_y0 = max(0, min(h - 2, int(round((sy0 + 0.60 * subj_h) * h))))
        band_y1 = max(band_y0 + 1, min(h, int(round((sy0 + 0.85 * subj_h) * h))))
        band_skin = skin_likelihood[band_y0:band_y1, :]
        if float(band_skin.sum()) > 1.0:
            col_weights = band_skin.sum(axis=0)
            subj_cx = float((col_weights * np.arange(w, dtype=np.float32)).sum() / (col_weights.sum() * max(1.0, float(w - 1))))
        crease_boost = np.clip((220.0 - y_luma) / 160.0, 0.2, 1.0) * grad_norm
        feature_mix = 0.50 * skin_likelihood + 0.40 * crease_boost + 0.10 * fg_norm
        rel_box_w, rel_box_h = 0.08, 0.08
    elif any(term in t_lower for term in hip_waist_terms):
        target_y = sy0 + 0.62 * subj_h
        sigma_y = max(0.08, 0.14 * subj_h)
        sigma_x = max(0.10, 0.22 * subj_w)
        feature_mix = 0.55 * skin_likelihood + 0.32 * grad_norm + 0.13 * fg_norm
        rel_box_w, rel_box_h = 0.14, 0.14
    elif any(term in t_lower for term in bottom_terms):
        target_y = sy0 + 0.88 * subj_h
        sigma_y = max(0.08, 0.14 * subj_h)
        sigma_x = max(0.11, 0.24 * subj_w)
        feature_mix = 0.48 * grad_norm + 0.32 * skin_likelihood + 0.20 * fg_norm
        rel_box_w, rel_box_h = 0.12, 0.12
    elif any(term in t_lower for term in hand_terms):
        target_y = sy0 + 0.50 * subj_h
        sigma_y = max(0.12, 0.22 * subj_h)
        sigma_x = max(0.14, 0.30 * subj_w)
        feature_mix = 0.48 * grad_norm + 0.40 * skin_likelihood + 0.12 * fg_norm
        rel_box_w, rel_box_h = 0.10, 0.10
    else:
        target_y = sy0 + 0.45 * subj_h
        sigma_y = max(0.12, 0.24 * subj_h)
        sigma_x = max(0.12, 0.26 * subj_w)
        feature_mix = 0.48 * grad_norm + 0.32 * fg_norm + 0.20 * skin_likelihood
        rel_box_w, rel_box_h = 0.14, 0.14

    spatial_prior = np.exp(
        -0.5 * (((xn - subj_cx) / sigma_x) ** 2 + ((yn - target_y) / sigma_y) ** 2)
    ).astype(np.float32)
    score_map = (0.60 * feature_mix + 0.40 * spatial_prior) * spatial_prior

    # Zero-padded integral image (cumsum) so box-window smoothing has exact alignment
    win_y = max(3, int(round(h * 0.08)))
    win_x = max(3, int(round(w * 0.08)))
    pad_top = win_y // 2
    pad_bottom = win_y - pad_top
    pad_left = win_x // 2
    pad_right = win_x - pad_left
    padded = np.pad(score_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float32)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    smoothed = (
        integral[win_y:win_y + h, win_x:win_x + w]
        - integral[:h, win_x:win_x + w]
        - integral[win_y:win_y + h, :w]
        + integral[:h, :w]
    ) / float(win_x * win_y)
    peak_idx = int(np.argmax(smoothed))
    py, px = divmod(peak_idx, w)
    cx = (float(px) + 0.5) / float(w)
    cy = (float(py) + 0.5) / float(h)

    box_w = max(0.06, min(0.28, rel_box_w))
    box_h = max(0.06, min(0.28, rel_box_h))
    x0 = max(0.01, min(0.99 - box_w, cx - 0.5 * box_w))
    y0 = max(0.01, min(0.99 - box_h, cy - 0.5 * box_h))
    x1 = min(0.99, x0 + box_w)
    y1 = min(0.99, y0 + box_h)
    bbox = (round(float(x0), 4), round(float(y0), 4), round(float(x1), 4), round(float(y1), 4))
    if log_callback:
        log_callback(
            f"Close-up local visual target grounding (ConvRot / non-GGUF · no mmproj required) · normalized bbox_xyxy={bbox}."
        )
    return GroundingParseResult("found", bbox, "ok")


class Qwen3VLGrounder:
    """Run the already-selected Model Catalog Qwen3-VL-family encoder locally on the GPU via llama.cpp.

    The model and matching multimodal projector are resolved from the active catalog profile;
    no separate grounding-model weights are downloaded. The child process binds to loopback only
    and offloads all layers to the NVIDIA CUDA GPU (--n-gpu-layers 99), releasing VRAM immediately after locating the target.
    """
    def __init__(self, model_path: Path | str, mmproj_path: Path | str | None = None):
        self.model_path = Path(model_path).expanduser().resolve()
        self.mmproj_path = (
            Path(mmproj_path).expanduser().resolve()
            if mmproj_path and str(mmproj_path).strip().lower() not in {"", "none"}
            else None
        )
        self._require_mmproj = True
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._gpu_layers: Optional[int] = None
        self._log_handle = None

    def _ensure_cuda_runtime_installed(self) -> None:
        if llama_grounding_runtime_present():
            return
        archive_dir = DATA_ROOT / "runtime-downloads"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / LLAMA_GROUNDING_ARCHIVE_FILENAME
        if not archive.is_file():
            url = GROUNDING_RUNTIME_ASSETS[0]["url"]
            part = archive.with_suffix(".zip.part")
            part.unlink(missing_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": f"P.AI-Image-Studio/{APP_VERSION}", "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(req, timeout=180, context=_download_ssl_context()) as resp, part.open("wb") as out:
                    shutil.copyfileobj(resp, out, length=4 * 1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
                minimum_size = int(GROUNDING_RUNTIME_ASSETS[0].get("size_hint", 0) * 0.70)
                if part.stat().st_size < minimum_size:
                    raise RuntimeError("The downloaded llama.cpp runtime archive is unexpectedly small; it was not installed.")
                from .constants import _sha256_file
                local_digest = _sha256_file(part)
                os.replace(part, archive)
                manifest_path = archive_dir / "llama-runtime-manifest.json"
                manifest_tmp = manifest_path.with_suffix(".json.tmp")
                stat = archive.stat()
                manifest_tmp.write_text(json.dumps({
                    "filename": archive.name, "sha256": local_digest,
                    "sha256_source": "computed-locally-no-upstream-pin",
                    "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                }, indent=2), encoding="utf-8")
                os.replace(manifest_tmp, manifest_path)
            except Exception:
                part.unlink(missing_ok=True)
                raise
        else:
            minimum_size = int(GROUNDING_RUNTIME_ASSETS[0].get("size_hint", 0) * 0.70)
            if archive.stat().st_size < minimum_size:
                raise RuntimeError("Cached llama.cpp runtime archive is unexpectedly small; download it again.")
        _extract_llama_grounding_archive_safe(archive, LLAMA_GROUNDING_RUNTIME_DIR)

    def _paths(self) -> tuple[Path, Path, Path]:
        # Deliberately use the exact text encoder and projector selected in Model Catalog.
        # llama.cpp's chat endpoint requires a GGUF language model plus the matching GGUF mmproj;
        # the native stable-diffusion.cpp encoder API does not expose token generation / bbox output.
        model = self.model_path
        mmproj = self.mmproj_path
        if not llama_grounding_runtime_present():
            try:
                self._ensure_cuda_runtime_installed()
            except Exception:
                pass
        server_candidates = sorted(LLAMA_GROUNDING_RUNTIME_DIR.rglob("llama-server.exe")) if LLAMA_GROUNDING_RUNTIME_DIR.exists() else []
        if not server_candidates:
            for search_root in (APP_DIR.parent, APP_DIR.parent.parent, Path.home() / "Downloads"):
                try:
                    for proj in search_root.glob("PAI_AI_IMAGE_STUDIO*"):
                        if proj.is_dir():
                            found = sorted(proj.glob("**/runtime/llama-cpp-grounding/**/llama-server.exe"))
                            if found:
                                server_candidates = found
                                break
                except OSError:
                    continue
                if server_candidates:
                    break
        server = server_candidates[0] if server_candidates else Path()
        if model.suffix.lower() != ".gguf":
            raise RuntimeError(
                "Close-up must reuse the selected Model Catalog text encoder, but the selected encoder is Safetensors/ConvRot. "
                "The local llama.cpp chat runtime cannot execute that file as a chat model. No second text encoder will be downloaded. "
                "Open Model Catalog and select the GGUF Qwen3-VL/Qwen Image 2.1 Turbo-Uncensored text encoder together with its matching vision projector."
            )
        model_name = model.name.lower().replace("_", "-")
        if not (
            "qwen3vl" in model_name
            or "qwen3-vl" in model_name
            or "qwen-image-2.1-turbo-uncensored" in model_name
            or "qwen-image-2.1-turbo-abliterated-uncensored" in model_name
            or "sakura-textencoder" in model_name
            or "text-encoder" in model_name
            or "textencoder" in model_name
            or "heretic" in model_name
            or "abliterated" in model_name
        ):
            raise RuntimeError(
                f"The selected Model Catalog text encoder does not look like a Qwen3-VL image encoder: {model.name}. "
                "Select the matching Qwen3-VL/Qwen Image 2.1 Turbo-Uncensored GGUF encoder to use it for Close-up target grounding."
            )
        missing = []
        if not model.is_file():
            missing.append(f"selected catalog text encoder: {model}")
        if getattr(self, "_require_mmproj", True) and (mmproj is None or not mmproj.is_file()):
            missing.append(f"matching selected catalog vision projector (mmproj): {mmproj}")
        if not server.is_file():
            missing.append("llama-server.exe runtime (runtime-only download; no model weights)")
        if missing:
            raise RuntimeError(
                "Close-up semantic grounding cannot start with the currently selected Model Catalog components. Missing: "
                + "; ".join(missing)
                + ". Choose a compatible GGUF encoder/projector in Model Catalog and use 'Prepare Locator Runtime' for the small runtime only."
            )
        return model, mmproj, server

    @staticmethod
    def _reserve_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_server_locked(self, timeout_seconds: float = 300.0, gpu_layers: int = 99) -> str:
        requested_gpu_layers = max(0, min(99, int(gpu_layers)))
        if self._process is not None and self._process.poll() is None and self._port is not None:
            if self._gpu_layers == requested_gpu_layers:
                return f"http://127.0.0.1:{self._port}"
            # Translation runs on CPU so it can coexist with diffusion VRAM use.
            # Restart if this shared server was previously started for GPU grounding.
            self.shutdown()
        model, mmproj, server = self._paths()
        self._port = self._reserve_loopback_port()
        log_path = CACHE_DIR / "qwen3vl-grounding-llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
        self._log_handle = log_path.open("ab")
        threads = max(2, min(8, (os.cpu_count() or 4) - 2))
        command = [str(server), "--model", str(model)]
        if mmproj is not None and mmproj.is_file():
            command.extend(["--mmproj", str(mmproj)])
        command.extend([
            "--host", "127.0.0.1", "--port", str(self._port),
            "--ctx-size", "4096", "--n-gpu-layers", str(requested_gpu_layers), "--threads", str(threads), "--alias", "pai-selected-qwen-image-encoder",
        ])
        env = os.environ.copy()
        dll_dirs = {str(server.parent)}
        for runtime_root in (server.parent, server.parent.parent, APP_DIR / "runtime", DATA_ROOT / "runtime", LLAMA_GROUNDING_RUNTIME_DIR):
            if runtime_root.is_dir():
                try:
                    dll_dirs.update(str(d.parent) for d in runtime_root.rglob("*.dll"))
                except OSError:
                    pass
        from .engine import LocalEngine
        cuda_search_dirs: list[Path] = []
        for candidate_root in (DATA_ROOT / "runtime", APP_DIR / "runtime", server.parent, server.parent.parent):
            if candidate_root.is_dir():
                cuda_search_dirs.append(candidate_root)
        for dl_root in (APP_DIR.parent, APP_DIR.parent.parent, Path.home() / "Downloads"):
            try:
                for proj in dl_root.glob("PAI_AI_IMAGE_STUDIO*"):
                    rt = proj / "data" / "runtime"
                    if rt.is_dir():
                        cuda_search_dirs.append(rt)
                    for nested_rt in proj.glob("**/data/runtime"):
                        if nested_rt.is_dir():
                            cuda_search_dirs.append(nested_rt)
            except OSError:
                pass
        cuda_search_dirs.extend(LocalEngine._windows_cuda_dependency_dirs())
        for dll_name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
            target_dll = server.parent / dll_name
            if not target_dll.is_file():
                for sroot in cuda_search_dirs:
                    try:
                        found_dll = next((p for p in sroot.rglob(dll_name) if p.is_file()), None) if sroot.is_dir() else None
                    except OSError:
                        found_dll = None
                    if found_dll is not None:
                        try:
                            shutil.copy2(found_dll, target_dll)
                            break
                        except OSError:
                            pass
        dll_dirs.update(str(path) for path in LocalEngine._windows_cuda_dependency_dirs())
        env["PATH"] = os.pathsep.join(sorted(dll_dirs)) + os.pathsep + env.get("PATH", "")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                command, cwd=str(server.parent), env=env, stdin=subprocess.DEVNULL,
                stdout=self._log_handle, stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self._gpu_layers = requested_gpu_layers
        except OSError as exc:
            raise RuntimeError(f"Could not start the local Qwen3-VL grounding runtime: {exc}") from exc
        base = f"http://127.0.0.1:{self._port}"
        deadline = time.monotonic() + max(1.0, min(300.0, float(timeout_seconds)))
        last_error = "server is loading its local model"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-3500:]
                except OSError:
                    tail = ""
                raise RuntimeError(
                    "The local chat runtime stopped while loading the Model Catalog text encoder/projector. Check system RAM and selected GGUF compatibility. "
                    + tail[-1800:]
                )
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        return base
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
            time.sleep(1.0)
        self._process.terminate()
        self._process = None
        self._port = None
        self._gpu_layers = None
        raise TimeoutError(
            "Timed out waiting for the selected Qwen image text encoder to load for Close-up grounding. "
            "This model needs several GB of system RAM. Check cache/qwen3vl-grounding-llama-server.log. " + last_error
        )

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        prepared.thumbnail((640, 640), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        prepared.save(buffer, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def expand_target_descriptions(
        self, target: str, timeout_seconds: float = 60.0, max_candidates: int = 3,
        shutdown_after_request: bool = True,
    ) -> list[str]:
        """Ask the selected local Qwen3-VL language model for bounded visual alternatives."""
        cap = max(0, min(3, int(max_candidates)))
        if cap <= 0:
            return []
        try:
            budget = min(300.0, float(timeout_seconds))
        except (TypeError, ValueError, OverflowError):
            return []
        if not (budget > 0.0 and budget < float("inf")):
            raise TimeoutError("Local target-description expansion has no time budget remaining.")
        request_deadline = time.monotonic() + budget
        target_text = str(target or "").strip()
        if not target_text:
            return []
        prompt = (
            "You help a local image-grounding system interpret a user's visual target. "
            "Return a ranked JSON array containing at most " + str(cap) + " short visual descriptions that are plausible "
            "natural-language alternatives or visually identifiable related parts of this concept. "
            "Use your language understanding; do not use a fixed synonym list. Preserve the same language as the user's target "
            "where practical. For a broad concept, parts may be useful only when they are genuinely related. "
            "Do not claim any candidate is present in an image because "
            "no image is provided to this request. Return only a JSON array of strings, no Markdown or explanation. "
            "Keep each string concise and under 180 characters.\n\nUser target: " + target_text
        )
        with self._lock:
            base = self._start_server_locked(timeout_seconds=max(1.0, request_deadline - time.monotonic()))
            remaining = request_deadline - time.monotonic()
            if remaining <= 0:
                if shutdown_after_request:
                    self.shutdown()
                raise TimeoutError("The grounding runtime used the request budget while starting target expansion.")
            payload = {
                "model": "pai-selected-qwen-image-encoder",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "top_p": 0.2,
                "max_tokens": min(192, 48 + 48 * cap),
                "stream": False,
            }
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=max(0.01, remaining)) as response:
                    raw = json.loads(response.read().decode("utf-8", errors="replace"))
            except (TimeoutError, socket.timeout) as exc:
                raise TimeoutError("Local Qwen3-VL target-description expansion timed out.") from exc
            except Exception as exc:
                raise RuntimeError(f"Local Qwen3-VL target-description expansion failed: {type(exc).__name__}") from exc
            finally:
                if shutdown_after_request:
                    self.shutdown()
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return []
        return parse_target_description_candidates(content, target_text, cap)

    def translate_prompt_to_english(
        self, prompt_text: str, timeout_seconds: float = 90.0,
        shutdown_after_request: bool = True, gpu_layers: int = 0,
    ) -> str:
        """Translate a generation/edit prompt with the already-selected local Qwen encoder."""
        original = str(prompt_text or "").strip()
        if not original:
            return ""
        try:
            budget = max(1.0, min(180.0, float(timeout_seconds)))
        except (TypeError, ValueError, OverflowError):
            budget = 90.0
        deadline = time.monotonic() + budget
        prompt = (
            "Translate the user's complete image-generation or image-editing prompt into clear, natural English "
            "for the Qwen Image 2.1 text-to-image model. Preserve every intended subject, action, attribute, scene, "
            "composition, lighting, style, mood, camera instruction, and any quoted or requested in-image text. "
            "Do not add new concepts, omit constraints, censor, summarize, answer the prompt, or explain your work. "
            "Keep proper names and technical terms intact. Output only the translated English prompt, without labels, "
            "quotation marks, Markdown, or commentary.\n\nUser prompt:\n" + original
        )
        raw: Any = None
        try:
            with self._lock:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("No time budget remains for local prompt translation.")
                    self._require_mmproj = False
                    try:
                        base = self._start_server_locked(timeout_seconds=remaining, gpu_layers=gpu_layers)
                    finally:
                        self._require_mmproj = True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("The selected local text encoder used the translation time budget while loading.")
                    payload = {
                        "model": "pai-selected-qwen-image-encoder",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "top_p": 0.2,
                        "max_tokens": 768,
                        "stream": False,
                    }
                    request = urllib.request.Request(
                        base + "/v1/chat/completions",
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=max(0.01, remaining)) as response:
                        raw = json.loads(response.read().decode("utf-8", errors="replace"))
                finally:
                    if shutdown_after_request:
                        self.shutdown()
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError("The selected local text encoder timed out while translating the prompt.") from exc
        except Exception as exc:
            raise RuntimeError(f"Local prompt translation failed: {type(exc).__name__}: {exc}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The selected local text encoder returned an unexpected translation response.") from exc
        translated = parse_english_prompt_translation(content, original)
        if not translated:
            raise RuntimeError("The selected local text encoder returned an empty prompt translation.")
        return translated

    def locate_detailed(self, image: Image.Image, target: str, log_callback=None, timeout_seconds: float = 240.0,
                        shutdown_after_request: bool = True) -> GroundingParseResult:
        request_deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        target_text = str(target or "main subject").strip() or "main subject"
        prompt = (
            "Locate the single visual target described below in the image. This is an open-vocabulary visual-grounding task: "
            "interpret the complete natural-language description semantically, including attributes, parts, context, and spatial relations; "
            "do not substitute a different object just because it is more salient. The target can be any visible object, person, animal, "
            "material, body part, or image region. Return the tightest useful bounding box around the full requested target, with a small "
            "margin and without cutting off important parts. If the requested target is not visibly present or cannot be distinguished, "
            "return {\"found\": false}. Coordinates must be normalized integers from 0 to 1000 in [x_min, y_min, x_max, y_max] order, "
            "with (0,0) at the top left. Output one JSON object only, with no Markdown. Schema: "
            "{\"found\": true, \"label\": \"short target label\", \"bbox_2d\": [x_min, y_min, x_max, y_max]}.\n\n"
            f"Target description, to interpret literally: {target_text}"
        )
        with self._lock:
            base = self._start_server_locked(timeout_seconds=max(1.0, request_deadline - time.monotonic()))
            remaining = request_deadline - time.monotonic()
            if remaining <= 0:
                if shutdown_after_request:
                    self.shutdown()
                raise TimeoutError("The grounding runtime used the request budget while starting.")
            payload = {
                "model": "pai-selected-qwen-image-encoder",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "temperature": 0.0,
                "top_p": 0.1,
                "max_tokens": 256,
                "stream": False,
            }
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=max(1.0, remaining)) as response:
                    raw = json.loads(response.read().decode("utf-8", errors="replace"))
            except (TimeoutError, socket.timeout) as exc:
                raise TimeoutError("Local Qwen3-VL grounding request timed out.") from exc
            except Exception as exc:
                raise RuntimeError(f"Local Qwen3-VL grounding inference failed: {type(exc).__name__}") from exc
            finally:
                if shutdown_after_request:
                    self.shutdown()
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The local Qwen3-VL server returned an unexpected chat-completion response.") from exc
        result = parse_qwen3vl_grounding_response_detailed(content)
        if log_callback:
            if result.status != "found":
                log_callback(f"Qwen3-VL returned no usable bounding box · reason={result.reason_code}; no keyword/center-crop fallback was applied.")
            else:
                log_callback(f"Model-Catalog Qwen3-VL semantic grounding succeeded · normalized bbox_xyxy={tuple(round(v, 4) for v in result.bbox_xyxy or ())}.")
        return result

    def locate(self, image: Image.Image, target: str, log_callback=None, timeout_seconds: float = 240.0,
               shutdown_after_request: bool = True) -> Optional[tuple[float, float, float, float]]:
        """Backward-compatible Optional-box interface for existing callers."""
        return self.locate_detailed(image, target, log_callback, timeout_seconds,
                                    shutdown_after_request=shutdown_after_request).bbox_xyxy

    def shutdown(self):
        with self._lock:
            process, self._process = self._process, None
            self._port = None
            self._gpu_layers = None
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except OSError:
                    pass
                self._log_handle = None


_QWEN3VL_GROUNDER: Optional[Qwen3VLGrounder] = None
_QWEN3VL_GROUNDER_LOCK = threading.Lock()

def _get_qwen3vl_grounder(model_path: Path | str, mmproj_path: Path | str | None = None) -> Qwen3VLGrounder:
    global _QWEN3VL_GROUNDER
    resolved_model = Path(model_path).expanduser().resolve()
    resolved_mmproj = (
        Path(mmproj_path).expanduser().resolve()
        if mmproj_path and str(mmproj_path).strip().lower() not in {"", "none"}
        else None
    )
    with _QWEN3VL_GROUNDER_LOCK:
        if (_QWEN3VL_GROUNDER is None
                or _QWEN3VL_GROUNDER.model_path != resolved_model
                or _QWEN3VL_GROUNDER.mmproj_path != resolved_mmproj):
            if _QWEN3VL_GROUNDER is not None:
                _QWEN3VL_GROUNDER.shutdown()
            _QWEN3VL_GROUNDER = Qwen3VLGrounder(resolved_model, resolved_mmproj)
        return _QWEN3VL_GROUNDER

def _shutdown_qwen3vl_grounder():
    if _QWEN3VL_GROUNDER is not None:
        _QWEN3VL_GROUNDER.shutdown()

atexit.register(_shutdown_qwen3vl_grounder)


def detect_target_bbox_local(image: Image.Image, target: str, log_callback=None,
                             model_path: Path | str | None = None,
                             mmproj_path: Path | str | None = None, timeout_seconds: float = 240.0,
                             detailed: bool = False, shutdown_after_request: bool = True) -> Optional[tuple[float, float, float, float]] | GroundingParseResult:
    """Ground an open-vocabulary target using either the selected GGUF/mmproj pair or built-in ConvRot/non-GGUF visual grounding.

    When a non-GGUF text encoder (such as INT8-ConvRot .safetensors) is selected or when mmproj
    is not enabled/downloaded, this automatically uses built-in local visual target grounding
    so Close-up works without requiring mmproj or llama.cpp.
    """
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    target_text = str(target or "main subject").strip() or "main subject"
    preview = normalized.copy()
    preview.thumbnail((768, 768), Image.Resampling.BICUBIC)
    model_str = str(Path(model_path).expanduser().resolve()) if model_path else "builtin-convrot-locator"
    mmproj_str = str(Path(mmproj_path).expanduser().resolve()) if mmproj_path else "no-mmproj"
    cache_key = (hashlib.sha256(preview.tobytes()).hexdigest(), target_text.casefold(), normalized.size,
                 model_str, mmproj_str)
    if cache_key in _LOCAL_BBOX_CACHE:
        _LOCAL_BBOX_CACHE.move_to_end(cache_key)
        if log_callback:
            log_callback("Close-up Qwen3-VL grounding: reused cached semantic box for the same image and target.")
        cached = _LOCAL_BBOX_CACHE[cache_key]
        return GroundingParseResult("found", cached, "ok") if detailed else cached

    is_gguf = bool(model_path) and Path(str(model_path)).suffix.lower() == ".gguf"
    has_mmproj = bool(mmproj_path) and str(mmproj_path).strip().lower() not in {"", "none"}

    effective_model = model_path
    effective_mmproj = mmproj_path
    auto_discovered_gguf = False
    if not is_gguf or not has_mmproj:
        resolved_m, resolved_v = resolve_closeup_grounding_gguf_pair(model_path, mmproj_path)
        if resolved_m is not None and resolved_v is not None:
            effective_model = resolved_m
            effective_mmproj = resolved_v
            is_gguf = True
            has_mmproj = True
            auto_discovered_gguf = True

    if not is_gguf or not has_mmproj:
        result = locate_target_bbox_non_gguf(normalized, target_text, log_callback=log_callback)
        box = result.bbox_xyxy
    else:
        grounder = _get_qwen3vl_grounder(effective_model, effective_mmproj)
        try:
            if detailed:
                result = grounder.locate_detailed(normalized, target_text, log_callback, timeout_seconds=timeout_seconds,
                                                  shutdown_after_request=shutdown_after_request)
                box = result.bbox_xyxy
            else:
                box = grounder.locate(normalized, target_text, log_callback, timeout_seconds=timeout_seconds,
                                      shutdown_after_request=shutdown_after_request)
                result = None
        except (RuntimeError, OSError, TimeoutError):
            if auto_discovered_gguf or not Path(str(effective_model)).is_file() or not Path(str(effective_mmproj)).is_file():
                result = locate_target_bbox_non_gguf(normalized, target_text, log_callback=log_callback)
                box = result.bbox_xyxy
            else:
                raise
        if box is None and auto_discovered_gguf:
            result = locate_target_bbox_non_gguf(normalized, target_text, log_callback=log_callback)
            box = result.bbox_xyxy
    if box is None:
        return result if detailed and result is not None else None
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        if log_callback:
            log_callback("Close-up Qwen3-VL returned invalid coordinates; no fallback crop was substituted.")
        invalid = GroundingParseResult("invalid_response", None, "invalid_box_geometry")
        return invalid if detailed else None
    _LOCAL_BBOX_CACHE[cache_key] = box
    while len(_LOCAL_BBOX_CACHE) > _LOCAL_BBOX_CACHE_LIMIT:
        _LOCAL_BBOX_CACHE.popitem(last=False)
    return GroundingParseResult("found", box, "ok") if detailed else box

def expand_qwen3vl_target_descriptions(
    target: str, *, model_path: Path | str | None = None, mmproj_path: Path | str | None = None,
    timeout_seconds: float = 60.0, max_candidates: int = 3, shutdown_after_request: bool = True,
) -> list[str]:
    """Expand a target via the selected local Model Catalog GGUF/mmproj pair when available."""
    if not model_path or not mmproj_path or Path(str(model_path)).suffix.lower() != ".gguf":
        return []
    target_text = str(target or "").strip()
    if not target_text:
        return []
    grounder = _get_qwen3vl_grounder(model_path, mmproj_path)
    try:
        raw_candidates = grounder.expand_target_descriptions(
            target_text, timeout_seconds=timeout_seconds, max_candidates=max_candidates,
            shutdown_after_request=shutdown_after_request,
        )
    except (RuntimeError, OSError, TimeoutError):
        return []
    return _sanitize_target_description_candidates(raw_candidates, target_text, max_candidates)


def translate_prompt_to_english(
    prompt_text: str, *, model_path: Path | str | None = None,
    mmproj_path: Path | str | None = None, timeout_seconds: float = 90.0,
    shutdown_after_request: bool = True, gpu_layers: int = 0,
) -> str:
    """Translate an image prompt with the selected Model Catalog GGUF encoder (mmproj optional for text-only translation)."""
    if not model_path:
        raise RuntimeError("Prompt translation requires the selected Model Catalog GGUF text encoder.")
    text = str(prompt_text or "").strip()
    if not text:
        return ""
    grounder = _get_qwen3vl_grounder(model_path, mmproj_path)
    return grounder.translate_prompt_to_english(
        text, timeout_seconds=timeout_seconds,
        shutdown_after_request=shutdown_after_request, gpu_layers=gpu_layers,
    )


def crop_closeup_from_bbox(original: Image.Image, side: int, bbox: tuple[float, float, float, float],
                           padding: float = 1.55, zoom_factor: Optional[float] = None):
    """Crop around a detected object box, balancing requested zoom and full-target coverage."""
    image = ImageOps.exif_transpose(original).convert("RGB")
    w, h = image.size
    x1, y1, x2, y2 = bbox
    cx, cy = ((x1 + x2) * 0.5 * w, (y1 + y2) * 0.5 * h)
    target_w, target_h = max(1.0, (x2 - x1) * w), max(1.0, (y2 - y1) * h)
    target_extent = max(target_w, target_h)
    # The zoom control must have a visible, predictable effect. Previously max(context_crop,
    # zoom_crop) let the context padding silently override the requested zoom for many targets.
    # Keep a small safety margin around the detected box, then use the requested zoom unless
    # that would cut the target off. Without an explicit zoom, use the wider context padding.
    desired_zoom = max(1.05, min(6.0, float(zoom_factor))) if zoom_factor is not None else None
    minimum_target_crop = target_extent * 1.12
    if desired_zoom is not None:
        zoom_crop = min(w, h) / desired_zoom
        crop_side = int(round(max(minimum_target_crop, zoom_crop)))
    else:
        crop_side = int(round(max(minimum_target_crop, target_extent * max(1.05, min(3.0, float(padding))))))
    crop_side = max(1, min(min(w, h), crop_side))
    left = int(round(cx - crop_side / 2))
    top = int(round(cy - crop_side / 2))
    left = max(0, min(w - crop_side, left))
    top = max(0, min(h - crop_side, top))
    crop = image.crop((left, top, left + crop_side, top + crop_side))
    return crop.resize((side, side), Image.Resampling.LANCZOS), (left, top, left + crop_side, top + crop_side), min(w, h) / max(1, crop_side)
