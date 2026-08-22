import copy

import pytest

from video_harness.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceValidationError,
    compose_evidence_record,
    evidence_is_trainable,
    mock_call2_record,
    mock_evidence_record,
    mock_inspection_record,
    validate_call2_record,
    validate_evidence_record,
    validate_inspection_record,
)


def test_call2_record_requires_three_descriptions_per_endpoint(call2_record) -> None:
    normalized = validate_call2_record(call2_record)
    assert set(normalized["endpoint_observation"]["before"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    del call2_record["endpoint_observation"]["after"]["cam_right_wrist"]
    with pytest.raises(EvidenceValidationError, match="must have exactly"):
        validate_call2_record(call2_record)


def test_canonical_evidence_composes_call1_and_call2(call2_record) -> None:
    evidence = compose_evidence_record(
        "The right gripper approaches the object and closes around it.",
        call2_record,
        review_status="accepted",
    )
    assert EVIDENCE_SCHEMA_VERSION == "video-harness.evidence"
    assert evidence["motion_summary"].startswith("The right gripper")
    assert evidence["review_status"] == "accepted"
    assert validate_evidence_record(copy.deepcopy(evidence)) == evidence
    assert evidence_is_trainable(evidence)


def test_review_status_must_match_causal_validation(call2_record) -> None:
    with pytest.raises(EvidenceValidationError, match="accepted exactly"):
        compose_evidence_record(
            "The robot moves.", call2_record, review_status="needs_review"
        )
    call2_record["causal_validation"]["status"] = "retry"
    evidence = compose_evidence_record(
        "The robot moves.", call2_record, review_status="needs_review"
    )
    assert not evidence_is_trainable(evidence)


def test_inspection_motion_summary_has_no_word_or_sentence_limit() -> None:
    record = mock_inspection_record()
    record["motion_summary"] = (
        "The right arm approaches the object. It pauses near the object; contact is "
        "not visible, and this deliberately exceeds the former summary limit while "
        "remaining useful temporal evidence for the next call."
    )
    assert validate_inspection_record(record)["motion_summary"] == record["motion_summary"]


def test_inspection_has_no_active_end_effector_field() -> None:
    record = mock_inspection_record()
    record["active_end_effector"] = "right"
    with pytest.raises(EvidenceValidationError, match="must have exactly"):
        validate_inspection_record(record)


def test_inspection_record_validates_detail_contract() -> None:
    record = {
        "motion_summary": "The left gripper approaches one small yellow object.",
        "interaction_window": {"start_frame": 7, "end_frame": 19},
        "needs_detail": True,
        "detail_request": {
            "x_min": 0.05,
            "y_min": 0.25,
            "x_max": 0.40,
            "y_max": 0.80,
            "reason": "gripper_object",
        },
    }
    assert validate_inspection_record(record) == record


def test_inspection_needs_detail_and_roi_must_agree() -> None:
    record = mock_inspection_record()
    record["needs_detail"] = True
    with pytest.raises(EvidenceValidationError, match="must agree"):
        validate_inspection_record(record)


def test_inspection_rejects_reversed_window() -> None:
    record = mock_inspection_record()
    record["interaction_window"] = {"start_frame": 12, "end_frame": 4}
    with pytest.raises(EvidenceValidationError, match="reversed"):
        validate_inspection_record(record)


def test_mock_records_are_valid_but_need_review() -> None:
    assert validate_call2_record(mock_call2_record()) == mock_call2_record()
    evidence = mock_evidence_record()
    assert validate_evidence_record(copy.deepcopy(evidence)) == evidence
    assert evidence["review_status"] == "needs_review"
    assert not evidence_is_trainable(evidence)
