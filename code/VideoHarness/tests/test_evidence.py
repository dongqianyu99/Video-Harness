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


def test_call2_record_requires_three_descriptions_per_boundary(call2_record) -> None:
    normalized = validate_call2_record(call2_record)
    assert set(normalized["before_boundary_observation"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    del call2_record["after_boundary_observation"]["cam_right_wrist"]
    with pytest.raises(EvidenceValidationError, match="must have exactly"):
        validate_call2_record(call2_record)


def test_canonical_evidence_uses_call2_motion_summary(call2_record) -> None:
    evidence = compose_evidence_record(
        call2_record,
        quality_status="accepted",
    )
    assert EVIDENCE_SCHEMA_VERSION == "video-harness.evidence.v3"
    assert evidence["motion_summary"] == call2_record["motion_summary"]
    assert evidence["quality_status"] == "accepted"
    assert validate_evidence_record(copy.deepcopy(evidence)) == evidence
    assert evidence_is_trainable(evidence)


def test_shared_boundary_is_represented_once_in_adjacent_units() -> None:
    """Adjacent units point at one boundary id; they do not duplicate state text."""
    from video_harness.robodojo import EpisodeRecord, VideoSlice
    from video_harness.sampling import plan_document

    views = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    source = EpisodeRecord(
        episode_index=0,
        task_index=0,
        task_instruction="x",
        task_kind="x",
        length=51,
        dataset_from_index=0,
        dataset_to_index=51,
        data_path="x",
        videos=tuple(
            VideoSlice(
                key=f"observation.images.{view}",
                path=f"{view}.mp4",
                from_timestamp=0.0,
                to_timestamp=2.0,
            )
            for view in views
        ),
    )
    document = plan_document(source, build_id="test")
    assert len(document["boundary_states"]) == len(document["evidence_units"]) + 1
    for left, right in zip(document["evidence_units"], document["evidence_units"][1:]):
        assert left["after_boundary_id"] == right["before_boundary_id"]


def test_quality_status_must_match_causal_validation(call2_record) -> None:
    with pytest.raises(EvidenceValidationError, match="accepted exactly"):
        compose_evidence_record(
            call2_record,
            quality_status="quarantined",
        )
    call2_record["causal_validation"]["status"] = "retry"
    evidence = compose_evidence_record(
        call2_record,
        quality_status="quarantined",
    )
    assert not evidence_is_trainable(evidence)


def test_inspection_sentence_contract_is_prompt_guidance_not_parser_logic() -> None:
    record = mock_inspection_record()
    record["motion_summary"] = (
        "The right arm approaches the object. It pauses near the object; contact is "
        "not visible, and this deliberately exceeds the former summary limit while "
        "remaining useful temporal evidence for the next call."
    )
    assert (
        validate_inspection_record(record)["motion_summary"] == record["motion_summary"]
    )


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
            "reason": "fine_spatial_detail",
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


def test_mock_records_are_valid_but_not_trainable() -> None:
    assert validate_call2_record(mock_call2_record()) == mock_call2_record()
    evidence = mock_evidence_record()
    assert validate_evidence_record(copy.deepcopy(evidence)) == evidence
    assert evidence["quality_status"] == "quarantined"
    assert not evidence_is_trainable(evidence)
