from __future__ import annotations

import pytest

from pai_studio.ui.grounding_review_dialog import (
    GroundingReviewDecision,
    map_preview_drag_to_image_box,
    should_accept_review_job,
)


def test_review_accept_requires_validated_box():
    decision = GroundingReviewDecision("accept", (0.1, 0.2, 0.7, 0.8))
    assert decision.action == "accept"
    assert decision.bbox_xyxy == (0.1, 0.2, 0.7, 0.8)


def test_review_rejects_invalid_accepted_box():
    with pytest.raises(ValueError):
        GroundingReviewDecision("accept", (0.8, 0.2, 0.1, 0.9))


def test_retry_and_cancel_do_not_return_a_box():
    assert GroundingReviewDecision("retry").bbox_xyxy is None
    assert GroundingReviewDecision("cancel").bbox_xyxy is None
    with pytest.raises(ValueError):
        GroundingReviewDecision("retry", (0.1, 0.2, 0.7, 0.8))


def test_review_drag_maps_letterboxed_preview_to_normalized_image_coordinates():
    # The actual image occupies (50, 25, 200, 100), not the whole widget.
    box = map_preview_drag_to_image_box((100, 50), (250, 100), (50, 25, 200, 100), (400, 200))
    assert box == pytest.approx((0.25, 0.25, 1.0, 0.75))


def test_review_drag_clamps_coordinates_to_image_content_rect():
    box = map_preview_drag_to_image_box((0, 20), (300, 140), (50, 20, 200, 100), (400, 200))
    assert box == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_late_result_from_previous_job_is_ignored():
    assert should_accept_review_job("job-new", "job-new") is True
    assert should_accept_review_job("job-old", "job-new") is False
    assert should_accept_review_job("job-new", "job-new", cancelled=True) is False


def test_manual_focus_contract_remains_independent_of_review_ui():
    from pai_studio.engine import Job

    job = Job(workflow="lora_closeup_1024", focus_point=(0.3, 0.7), auto_zoom=True)
    assert job.focus_point == (0.3, 0.7)


def test_semantic_candidate_label_includes_model_generated_description():
    from pai_studio.ui.grounding_review_dialog import grounding_candidate_label

    assert grounding_candidate_label(1, "eyes and nose", (0.1, 0.2, 0.4, 0.6)).startswith("eyes and nose · Candidate 1")
    assert grounding_candidate_label(2, "", (0.1, 0.2, 0.4, 0.6)).startswith("Candidate 2 ·")
    assert "\n" not in grounding_candidate_label(1, "eyes\nand nose", (0.1, 0.2, 0.4, 0.6))
