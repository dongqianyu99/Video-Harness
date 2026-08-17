import copy

import pytest

from video_harness.evidence import (
    EvidenceValidationError,
    evidence_is_trainable,
    mock_evidence_record,
    validate_evidence_record,
)


def test_changed_evidence_is_normalized_and_trainable(changed_evidence) -> None:
    changed_evidence["visibility_limits"] = [
        "grasp_contact",
        "precise_pose",
        "motion_path",
        "force",
    ]
    normalized = validate_evidence_record(changed_evidence)
    assert normalized["visibility_limits"] == [
        "motion_path",
        "force",
        "precise_pose",
        "grasp_contact",
    ]
    assert evidence_is_trainable(normalized)


def test_missing_fixed_visibility_limit_is_rejected(changed_evidence) -> None:
    changed_evidence["visibility_limits"].remove("force")
    with pytest.raises(EvidenceValidationError, match="missing fixed endpoint limits"):
        validate_evidence_record(changed_evidence)


def test_no_change_cannot_contain_operation_hint(changed_evidence) -> None:
    changed_evidence["change_status"] = "no_task_relevant_change"
    changed_evidence["visual_observation"]["change"] = None
    with pytest.raises(EvidenceValidationError, match="operation_hint must be null"):
        validate_evidence_record(changed_evidence)


def test_ambiguous_change_requires_explicit_opt_in(changed_evidence) -> None:
    changed_evidence["visual_observation"]["support"] = "ambiguous"
    assert not evidence_is_trainable(changed_evidence)
    assert evidence_is_trainable(changed_evidence, allow_ambiguous=True)


def test_changed_evidence_must_be_task_relevant(changed_evidence) -> None:
    changed_evidence["task_relevance"] = "incidental"
    with pytest.raises(EvidenceValidationError, match="requires task_relevance"):
        validate_evidence_record(changed_evidence)


def test_mock_record_is_valid_but_not_trainable() -> None:
    record = mock_evidence_record()
    assert validate_evidence_record(copy.deepcopy(record)) == record
    assert not evidence_is_trainable(record)
