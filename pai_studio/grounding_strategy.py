"""Bounded orchestration around the existing single-image grounding adapter."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, NamedTuple, Optional

from PIL import Image, ImageOps

from .grounding import GroundingParseResult, _sanitize_target_description_candidates
from .performance import GroundingProfile, get_grounding_profile

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundingResult:
    status: str
    bbox_xyxy: Optional[BBox]
    target_text: str
    attempts: int
    reason_code: str
    diagnostic_summary: str
    elapsed_ms: int
    candidates: tuple[BBox, ...] = ()
    matched_description: str = ""
    candidate_descriptions: tuple[str, ...] = ()


def _valid_bbox(value) -> Optional[BBox]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) for v in value):
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(v == v and v not in (float("inf"), float("-inf")) for v in (x0, y0, x1, y1)):
        return None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    if (x1 - x0) * (y1 - y0) > 0.96:
        return None
    return (x0, y0, x1, y1)


class TileRect(NamedTuple):
    """Half-open integer pixel rectangle in the EXIF-corrected source frame."""
    left: int
    top: int
    right: int
    bottom: int


def plan_grounding_tiles(image_size: tuple[int, int], profile: str | GroundingProfile) -> tuple[TileRect, ...]:
    """Plan a deterministic, overlapping, bounded tile set for fallback grounding."""
    selected = get_grounding_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected, GroundingProfile):
        raise TypeError("profile must be a profile name or GroundingProfile")
    if not selected.multiscale_enabled or selected.max_tiles <= 0:
        return ()
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) pair")
    width, height = image_size
    if isinstance(width, bool) or isinstance(height, bool) or int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    width, height = int(width), int(height)
    tile_budget = selected.max_tiles
    # Square grids make tile behavior easy to reason about and cap growth. If a
    # profile later allows a non-square budget, only full rows are planned.
    columns = max(1, int(tile_budget ** 0.5))
    rows = max(1, tile_budget // columns)
    # Current profiles are 0, 4 and 9; this keeps the contract safe for others.
    while columns * rows > tile_budget:
        rows -= 1
    overlap_scale = 1.25
    tile_width = min(width, max(1, int(round(width / columns * overlap_scale))))
    tile_height = min(height, max(1, int(round(height / rows * overlap_scale))))

    def starts(extent: int, tile_extent: int, count: int) -> list[int]:
        if count <= 1 or tile_extent >= extent:
            return [0]
        max_start = extent - tile_extent
        return [int(round(index * max_start / (count - 1))) for index in range(count)]

    xs = starts(width, tile_width, columns)
    ys = starts(height, tile_height, rows)
    tiles: list[TileRect] = []
    for top in ys:
        for left in xs:
            rect = TileRect(left, top, min(width, left + tile_width), min(height, top + tile_height))
            if rect.right > rect.left and rect.bottom > rect.top and rect not in tiles:
                tiles.append(rect)
            if len(tiles) >= tile_budget:
                return tuple(tiles)
    return tuple(tiles)


def map_tile_bbox_to_image(
    tile_rect: tuple[int, int, int, int],
    original_image_size: tuple[int, int],
    bbox_xyxy: tuple[float, float, float, float],
) -> BBox:
    """Project a tile-normalized box to normalized XYXY coordinates in the source frame."""
    if len(tile_rect) != 4 or len(original_image_size) != 2:
        raise ValueError("tile rectangle and image size have invalid shapes")
    left, top, right, bottom = tile_rect
    width, height = original_image_size
    if any(isinstance(v, bool) or int(v) != v for v in (left, top, right, bottom, width, height)):
        raise ValueError("tile geometry must use integer pixel coordinates")
    if width <= 0 or height <= 0 or left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
        raise ValueError("tile rectangle must be a non-empty region inside the image")
    box = _valid_bbox(bbox_xyxy)
    if box is None:
        raise ValueError("tile bbox must be finite, ordered, and normalized")
    x0, y0, x1, y1 = box
    mapped = ((left + x0 * (right-left)) / width,
              (top + y0 * (bottom-top)) / height,
              (left + x1 * (right-left)) / width,
              (top + y1 * (bottom-top)) / height)
    result = _valid_bbox(mapped)
    if result is None:
        raise ValueError("mapped bbox is invalid")
    return result


def _sanitize_expanded_descriptions(values, target: str, cap: int = 3) -> tuple[str, ...]:
    """Delegate candidate sanitization to grounding._sanitize_target_description_candidates."""
    return tuple(_sanitize_target_description_candidates(values, target, max_candidates=cap))


def _bbox_iou(left: BBox, right: BBox) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1-x0) * max(0.0, y1-y0)
    area_left = (left[2]-left[0]) * (left[3]-left[1])
    area_right = (right[2]-right[0]) * (right[3]-right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_labeled_candidates(items, iou_threshold: float = 0.55) -> tuple[tuple[BBox, ...], tuple[str, ...]]:
    """Keep bounding boxes and their generated descriptions aligned after IoU de-duplication."""
    boxes: list[BBox] = []
    descriptions: list[str] = []
    for raw_box, raw_description in items:
        box = _valid_bbox(raw_box)
        if box is None:
            continue
        if any(_bbox_iou(existing, box) >= iou_threshold for existing in boxes):
            continue
        boxes.append(box)
        descriptions.append(str(raw_description or "").strip())
    return tuple(boxes), tuple(descriptions)


def deduplicate_grounding_candidates(
    candidates, iou_threshold: float = 0.55
) -> tuple[BBox, ...]:
    """Remove overlapping duplicate detections using IoU; this is not a confidence score."""
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)) or not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be between zero and one")
    unique: list[BBox] = []
    for candidate in candidates:
        box = _valid_bbox(candidate)
        if box is None:
            continue
        if any(_bbox_iou(existing, box) >= float(iou_threshold) for existing in unique):
            continue
        unique.append(box)
    return tuple(unique)


def run_grounding_strategy(
    image: Image.Image,
    target: str,
    *,
    locator: Callable,
    profile: str | GroundingProfile = "balanced",
    should_cancel: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    expand_target: Optional[Callable[[str, float], list[str]]] = None,
    max_expanded_candidates: int = 3,
) -> GroundingResult:
    """Run bounded grounding with optional tile fallback in original-image coordinates."""
    selected = get_grounding_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected, GroundingProfile):
        raise TypeError("profile must be a profile name or GroundingProfile")
    target_text = str(target or "main subject").strip() or "main subject"
    source = ImageOps.exif_transpose(image).copy()
    started = time.monotonic()
    deadline = started + selected.total_budget_seconds
    attempts = 0
    retry_target = target_text
    last_reason = "bbox_missing"
    last_status = "invalid_response"
    tile_candidates: list[tuple[BBox, str]] = []
    tile_not_found = 0
    tiles_attempted = 0
    expansion_enabled = bool(expand_target and selected.multiscale_enabled)
    try:
        expansion_cap = max(0, min(3, int(max_expanded_candidates))) if expansion_enabled else 0
    except (TypeError, ValueError, OverflowError):
        expansion_cap = 0
        expansion_enabled = False
    max_total_calls = selected.max_attempts + selected.max_tiles + (expansion_cap + 1 if expansion_enabled and expansion_cap else 0)

    def elapsed() -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    def emit(message: str) -> None:
        if log_callback:
            # Do not include user target text, full prompts, media payloads or exception text.
            log_callback(message)

    def call_locator(region: Image.Image, description: str):
        nonlocal attempts
        if should_cancel and should_cancel():
            return "cancelled", None, "cancelled"
        if attempts >= max_total_calls:
            return "timeout", None, "model_call_budget_exhausted"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout", None, "total_budget_exhausted"
        attempts += 1
        try:
            raw = locator(region, description, log_callback=None,
                          timeout_seconds=min(selected.per_request_timeout_seconds, max(0.001, remaining)))
            if time.monotonic() >= deadline:
                return "timeout", None, "total_budget_exhausted"
        except TimeoutError:
            return "timeout", None, "inference_timeout"
        except (OSError, RuntimeError):
            return "runtime_error", None, "runtime_error"
        if isinstance(raw, GroundingParseResult):
            if raw.status == "found":
                box = _valid_bbox(raw.bbox_xyxy)
                return ("found", box, "ok") if box else ("invalid_response", None, "invalid_box_geometry")
            return (raw.status, None, raw.reason_code)
        box = _valid_bbox(raw)
        return ("found", box, "ok") if box else ("invalid_response", None, "bbox_missing")

    # First inspect the whole image, with a meaningfully different second prompt if needed.
    for attempt_idx in range(selected.max_attempts):
        status, box, reason = call_locator(source, retry_target)
        last_status, last_reason = status, reason
        if status == "found" and box is not None:
            emit("Grounding succeeded with a validated bounding box.")
            return GroundingResult("found", box, target_text, attempts, "ok", "A valid target box was found.", elapsed(), (box,))
        if status == "not_found":
            emit("Grounding model did not locate the target in the current image region.")
            # Do not spend retries with near-identical whole-image prompts after an
            # explicit negative. Multiscale profiles proceed to the bounded tile pass
            # below; Fast has no tiles and retains a clear not_found result.
            break
        if status in {"timeout", "runtime_error", "cancelled"}:
            emit(f"Grounding stopped: {reason}.")
            summary = {"timeout": "Grounding exceeded its time budget.", "runtime_error": "The local grounding runtime failed.", "cancelled": "Grounding was cancelled."}[status]
            return GroundingResult(status, None, target_text, attempts, reason, summary, elapsed())
        if attempt_idx + 1 < selected.max_attempts:
            if attempt_idx == 0:
                retry_target = (
                    f"Clearly visible instance of: {target_text}. Return only a tight bounding box "
                    "for this exact target; do not substitute another object."
                )
            else:
                retry_target = (
                    f"Inspect the image carefully for the exact visual target: {target_text}. "
                    "Use the smallest box that contains the whole visible target. If uncertain, "
                    "return an explicit not-found result rather than guessing."
                )
            emit(f"No valid whole-image box; bounded retry {attempt_idx + 2} of {selected.max_attempts}.")

    # Optional semantic expansion uses the already-selected local language model only.
    # Candidate text is never treated as a detection: every description must still produce
    # a validated image-grounding box. Both request count and wall-clock time stay bounded.
    expanded_targets: tuple[str, ...] = ()
    expansion_hits: list[tuple[BBox, str]] = []
    if (last_status == "not_found" and expansion_enabled and expansion_cap > 0
            and attempts < max_total_calls and time.monotonic() < deadline):
        if should_cancel and should_cancel():
            return GroundingResult("cancelled", None, target_text, attempts, "cancelled", "Grounding was cancelled.", elapsed())
        remaining = deadline - time.monotonic()
        if remaining > 0:
            attempts += 1  # Count the text-only expansion request in the model-call budget.
            try:
                raw_descriptions = expand_target(target_text, min(selected.per_request_timeout_seconds, remaining))
                expanded_targets = _sanitize_expanded_descriptions(raw_descriptions, target_text, expansion_cap)
                emit(f"Semantic target expansion produced {len(expanded_targets)} bounded alternative descriptions.")
            except TimeoutError:
                last_reason = "semantic_expansion_timeout"
                emit("Semantic target expansion timed out; continuing with direct multiscale grounding where budget remains.")
            except (OSError, RuntimeError):
                last_reason = "semantic_expansion_runtime_error"
                emit("Semantic target expansion was unavailable; continuing with direct multiscale grounding where budget remains.")

    for description in expanded_targets:
        status, box, reason = call_locator(source, description)
        last_status, last_reason = status, reason
        if status == "found" and box is not None:
            expansion_hits.append((box, description))
            if not selected.review_ambiguous:
                return GroundingResult("found", box, target_text, attempts, "semantic_alternative_found",
                                       "A related model-generated description was visually grounded.", elapsed(), (box,), description)
        elif status in {"timeout", "runtime_error", "cancelled"}:
            emit(f"Semantic alternative search stopped: {reason}.")
            summary = {"timeout": "Grounding exceeded its time budget.", "runtime_error": "The local grounding runtime failed.", "cancelled": "Grounding was cancelled."}[status]
            return GroundingResult(status, None, target_text, attempts, reason, summary, elapsed())

    if expansion_hits:
        expanded_boxes, expanded_labels = _deduplicate_labeled_candidates(expansion_hits)
        if len(expanded_boxes) == 1:
            picked_box = expanded_boxes[0]
            description = next((desc for box, desc in expansion_hits if _bbox_iou(box, picked_box) >= 0.55), target_text)
            emit("One unique model-generated target description was visually grounded.")
            return GroundingResult("found", picked_box, target_text, attempts, "semantic_alternative_found",
                                   "A related model-generated description was visually grounded.", elapsed(), expanded_boxes, description)
        if len(expanded_boxes) > 1:
            emit("Several semantic alternatives produced distinct boxes; user review is required.")
            return GroundingResult("ambiguous", None, target_text, attempts, "multiple_semantic_candidates",
                                   "Related target descriptions produced multiple distinct visual boxes; review is required.",
                                   elapsed(), expanded_boxes, "", expanded_labels)

    # Multiscale is a fallback, not an unconditional cost on successful images. When
    # expansion was useful but whole-image alternatives failed, distribute tile requests
    # across the dynamic alternatives and the original user wording within the same cap.
    tiles = plan_grounding_tiles(source.size, selected)
    tile_descriptions = expanded_targets + (target_text,) if expanded_targets else (target_text,)
    for tile_index, tile in enumerate(tiles):
        if should_cancel and should_cancel():
            emit("Grounding cancelled before the next tile.")
            cancel_boxes, cancel_labels = _deduplicate_labeled_candidates(tile_candidates)
            return GroundingResult("cancelled", None, target_text, attempts, "cancelled", "Grounding was cancelled.", elapsed(), cancel_boxes, candidate_descriptions=cancel_labels)
        if time.monotonic() >= deadline:
            last_status, last_reason = "timeout", "total_budget_exhausted"
            break
        tile_image = source.crop(tuple(tile))
        tiles_attempted += 1
        description = tile_descriptions[tile_index % len(tile_descriptions)]
        status, box, reason = call_locator(tile_image, description)
        last_status, last_reason = status, reason
        if status == "found" and box is not None:
            try:
                tile_candidates.append((map_tile_bbox_to_image(tuple(tile), source.size, box), description))
            except ValueError:
                continue
        elif status == "not_found":
            tile_not_found += 1
        elif status in {"timeout", "runtime_error", "cancelled"}:
            emit(f"Multiscale grounding stopped: {reason}.")
            summary = {"timeout": "Grounding exceeded its time budget.", "runtime_error": "The local grounding runtime failed.", "cancelled": "Grounding was cancelled."}[status]
            partial_boxes, partial_labels = _deduplicate_labeled_candidates(tile_candidates)
            return GroundingResult(status, None, target_text, attempts, reason, summary, elapsed(), partial_boxes, candidate_descriptions=partial_labels)

    candidates, candidate_labels = _deduplicate_labeled_candidates(tile_candidates)
    if len(candidates) == 1:
        picked = candidates[0]
        matched = next((desc for box, desc in tile_candidates if _bbox_iou(box, picked) >= 0.55), target_text)
        emit("Multiscale grounding produced one unique validated box.")
        reason = "semantic_alternative_found" if matched.casefold() != target_text.casefold() else "ok"
        summary = "A related model-generated description was visually grounded across image regions." if reason != "ok" else "One unique target box was found across image regions."
        return GroundingResult("found", picked, target_text, attempts, reason, summary, elapsed(), candidates, matched)
    if len(candidates) > 1:
        emit("Multiscale grounding found multiple distinct candidates; user review is required.")
        return GroundingResult("ambiguous", None, target_text, attempts, "multiple_distinct_candidates", "Several distinct target boxes remain; review is required.", elapsed(), candidates, "", candidate_labels)
    if tiles and tiles_attempted > 0 and tile_not_found == tiles_attempted:
        last_status, last_reason = "not_found", "target_not_found_in_tiles"
    elif last_status not in {"timeout", "runtime_error", "cancelled", "not_found"}:
        last_status, last_reason = "invalid_response", "bbox_missing"
    summary = {
        "timeout": "Grounding did not finish within its time budget.",
        "runtime_error": "The local grounding runtime failed.",
        "not_found": "The model did not locate the requested target.",
        "cancelled": "Grounding was cancelled.",
        "invalid_response": "The model did not return a valid target box.",
    }.get(last_status, "Grounding failed.")
    emit(f"Grounding failed: {last_reason}.")
    return GroundingResult(last_status, None, target_text, attempts, last_reason, summary, elapsed(), candidates)
