"""Bounded, immutable performance budgets for semantic grounding."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingProfile:
    name: str
    max_attempts: int
    max_tiles: int
    per_request_timeout_seconds: float
    total_budget_seconds: float
    multiscale_enabled: bool
    review_ambiguous: bool


GROUNDING_PROFILE_NAMES = ("fast", "balanced", "accuracy")
_PROFILE_DESCRIPTIONS = {
    "fast": "Fast: one full-image request, no tiling; lowest expected latency and compute cost, but small targets may be missed.",
    "balanced": "Balanced: limited retry and a small tiled fallback; moderate latency and memory use for a practical compromise.",
    "accuracy": "Accuracy: more bounded tile attempts and review of ambiguous candidates; higher latency and memory use, but cannot guarantee a better semantic result.",
}


def normalize_grounding_profile(name: str, default: str = "balanced") -> str:
    """Normalize persisted/UI profile values without ever creating an unbounded profile."""
    key = str(name or "").strip().lower()
    if key in GROUNDING_PROFILE_NAMES:
        return key
    fallback = str(default or "balanced").strip().lower()
    return fallback if fallback in GROUNDING_PROFILE_NAMES else "balanced"


def grounding_profile_description(name: str) -> str:
    """Return user-facing tradeoffs without promising model accuracy."""
    return _PROFILE_DESCRIPTIONS[normalize_grounding_profile(name)]


_PROFILES = {
    "fast": GroundingProfile("fast", 1, 0, 240.0, 300.0, False, False),
    "balanced": GroundingProfile("balanced", 2, 4, 180.0, 720.0, True, False),
    "accuracy": GroundingProfile("accuracy", 3, 9, 180.0, 1500.0, True, True),
}


def get_grounding_profile(name: str) -> GroundingProfile:
    """Return a fixed profile; unknown values never become unbounded defaults."""
    key = str(name or "").strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported grounding profile: {key or '<empty>'}") from exc


def enhance_detail_mode_for_ultimate(ultimate_enabled: bool) -> str:
    """Ultimate is the only mode allowed to split/refine an Enhance result into tiles."""
    return "ultimate" if bool(ultimate_enabled) else "fast"
