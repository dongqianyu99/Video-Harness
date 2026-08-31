from __future__ import annotations

import copy
import json

import pytest
from _support import annotate_boundaries

from video_harness.annotations import (
    AnnotationError,
    RepairResult,
    SequenceAuditResult,
)
from video_harness.config import HarnessConfig
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.pipeline import ExistingRepairOutcome
from video_harness.quality import accepted_transition_chain
from video_harness.reconciliation import (
    _repair_targets,
    build_sequence_projection,
    reconcile_document,
    sequence_projection_sha256,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document, validate_document


def _document(evidence: dict, *, length: int = 51) -> dict:
    record = EpisodeRecord(
        episode_index=0,
        task_index=0,
        task_instruction="Perform the task.",
        task_kind="benchmark",
        length=length,
        dataset_from_index=0,
        dataset_to_index=length,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=key,
                path=f"videos/{key}.mp4",
                from_timestamp=0.0,
                to_timestamp=length / 25,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ),
    )
    document = plan_document(record, build_id="test-build")
    annotate_boundaries(document)
    for unit in document["evidence_units"]:
        unit["annotation"] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "complete",
            "record": copy.deepcopy(evidence),
            "provenance": {
                "call1": {
                    "provider": "test",
                    "model": "motion",
                    "prompt_version": "inspection",
                },
                "call2": {
                    "provider": "test",
                    "model": "evidence",
                    "prompt_version": "evidence",
                },
                "repair": None,
            },
        }
    document["status"] = "annotated"
    return document


class FakeAuditBackend:
    provider = "test-auditor"
    model = "auditor"

    def __init__(self, audits) -> None:
        self.audits = list(audits)
        self.requests = []

    def audit_sequence(self, request):
        self.requests.append(request)
        outcome = self.audits.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SequenceAuditResult(outcome, self.provider, self.model)

    def repair(self, request):  # pragma: no cover - fake pipeline owns repairs
        raise AssertionError(request)


class FakeRepairPipeline:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    def repair_target(
        self,
        document,
        unit,
        *,
        issue_reason,
        attempts=None,
        allowed_boundary_replacements=frozenset(),
        required_boundary_replacements=frozenset(),
    ):
        self.calls.append(
            (
                document["document_id"],
                unit["unit_id"],
                issue_reason,
                allowed_boundary_replacements,
                required_boundary_replacements,
            )
        )
        return self.outcomes.pop(0)


def _successful_repair(
    evidence: dict,
    *,
    before_boundary: dict | None = None,
    after_boundary: dict | None = None,
) -> ExistingRepairOutcome:
    repaired = copy.deepcopy(evidence)
    repaired["motion_summary"] = "Automatically resolved motion."
    result = RepairResult(
        {
            "evidence_sufficient": True,
            "reason": "The bounded evidence resolves the issue.",
            "resolved_call2": {"motion_summary": "Automatically resolved motion."},
        },
        "test-repair",
        "repair-model",
    )
    return ExistingRepairOutcome(
        repaired,
        before_boundary,
        after_boundary,
        1,
        result,
    )


def test_sequence_projection_is_stable_and_deduplicates_boundaries(changed_evidence):
    document = _document(changed_evidence)
    first = build_sequence_projection(document)
    second = build_sequence_projection(copy.deepcopy(document))
    payload = json.loads(first)

    assert first == second
    assert len(payload["boundaries"]) == 3
    assert len(payload["evidence_units"]) == 2
    assert "detail_observation" in payload["evidence_units"][0]
    assert "causal_validation" not in payload["evidence_units"][0]
    assert len(sequence_projection_sha256(document)) == 64


def test_empty_sequence_audit_accepts_complete_document(changed_evidence):
    document = _document(changed_evidence)
    backend = FakeAuditBackend([{"issues": []}])
    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=FakeRepairPipeline([]),
        config=HarnessConfig(sequence_repair_rounds=1),
    )

    assert outcome.document["quality_status"] == "accepted"
    assert outcome.issues == ()
    assert outcome.document["quality_provenance"]["audit_attempts"] == 1
    assert len(list(accepted_transition_chain(outcome.document))) == 2
    validate_document(outcome.document)

    incomplete = copy.deepcopy(outcome.document)
    incomplete["evidence_units"][0]["annotation"].update(
        {"status": "failed", "record": None, "provenance": None}
    )
    with pytest.raises(ValueError, match="not complete"):
        list(accepted_transition_chain(incomplete))


def test_sequence_issue_is_repaired_and_reaudited(changed_evidence):
    document = _document(changed_evidence)
    backend = FakeAuditBackend(
        [
            {
                "issues": [
                    {
                        "target_type": "unit",
                        "target_id": "u0000",
                        "reason": "Missing release.",
                    }
                ]
            },
            {"issues": []},
        ]
    )
    pipeline = FakeRepairPipeline(
        [
            _successful_repair(changed_evidence),
            _successful_repair(changed_evidence),
        ]
    )
    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=pipeline,
        config=HarnessConfig(sequence_repair_rounds=1),
    )

    assert outcome.document["quality_status"] == "accepted"
    assert outcome.repair_rounds == 1
    assert pipeline.calls[0][1] == "u0000"
    assert "unit u0000: Missing release." in pipeline.calls[0][2]
    assert (
        outcome.document["evidence_units"][0]["annotation"]["record"]["motion_summary"]
        == "Automatically resolved motion."
    )
    validate_document(outcome.document)


def test_unresolved_issue_quarantines_document_without_human_queue(changed_evidence):
    document = _document(changed_evidence)
    backend = FakeAuditBackend(
        [
            {
                "issues": [
                    {
                        "target_type": "unit",
                        "target_id": "u0001",
                        "reason": "Unresolved continuity.",
                    }
                ]
            }
        ]
    )
    pipeline = FakeRepairPipeline([ExistingRepairOutcome(None, None, None, 2, None)])
    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=pipeline,
        config=HarnessConfig(sequence_repair_rounds=1),
    )

    assert outcome.document["quality_status"] == "quarantined"
    assert outcome.document["quality_provenance"] is not None
    assert outcome.issues[0]["target_id"] == "u0001"
    validate_document(outcome.document)


def test_repair_provider_failure_is_exposed_for_resume(changed_evidence):
    document = _document(changed_evidence)
    backend = FakeAuditBackend(
        [
            {
                "issues": [
                    {"target_type": "unit", "target_id": "u0001", "reason": "Retry."}
                ]
            }
        ]
    )
    pipeline = FakeRepairPipeline(
        [ExistingRepairOutcome(None, None, None, 2, None, "TimeoutError: offline")]
    )
    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=pipeline,
        config=HarnessConfig(sequence_repair_rounds=1),
    )

    assert outcome.document["quality_status"] == "pending"
    assert outcome.document["quality_provenance"] is None
    assert outcome.technical_failures == (
        {
            "target_id": "u0001",
            "error": "Targeted repair provider failed: TimeoutError: offline",
        },
    )


def test_unit_only_repair_does_not_replace_boundary(changed_evidence):
    document = _document(changed_evidence)
    original_boundary = copy.deepcopy(document["boundary_states"][1])
    backend = FakeAuditBackend(
        [
            {
                "issues": [
                    {
                        "target_type": "unit",
                        "target_id": "u0000",
                        "reason": "Boundary conflict.",
                    }
                ]
            },
            {"issues": []},
        ]
    )
    pipeline = FakeRepairPipeline(
        [
            _successful_repair(changed_evidence),
            _successful_repair(changed_evidence),
        ]
    )
    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=pipeline,
        config=HarnessConfig(sequence_repair_rounds=1),
    )

    assert outcome.document["quality_status"] == "accepted"
    assert outcome.document["boundary_states"][1] == original_boundary
    assert pipeline.calls[0][3:] == (frozenset(), frozenset())


def test_boundary_issue_repairs_owner_without_eager_adjacent_rewrite(
    changed_evidence,
):
    document = _document(changed_evidence)
    replacement = copy.deepcopy(document["boundary_states"][1]["annotation"]["record"])
    replacement["observation"]["cam_high"] = "Corrected shared state."
    backend = FakeAuditBackend(
        [
            {
                "issues": [
                    {
                        "target_type": "boundary",
                        "target_id": "b0001",
                        "reason": "The shared Boundary is visually incorrect.",
                    }
                ]
            },
            {"issues": []},
        ]
    )
    pipeline = FakeRepairPipeline(
        [
            _successful_repair(changed_evidence, after_boundary=replacement),
        ]
    )

    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=pipeline,
        config=HarnessConfig(sequence_repair_rounds=1),
    )

    assert outcome.document["quality_status"] == "accepted"
    assert outcome.document["boundary_states"][1]["annotation"]["record"] == replacement
    assert [call[1] for call in pipeline.calls] == ["u0000"]
    assert pipeline.calls[0][3:] == (
        frozenset({"after"}),
        frozenset({"after"}),
    )


def test_repair_targets_merge_issues_by_owner(changed_evidence):
    document = _document(changed_evidence)
    targets = _repair_targets(
        document,
        [
            {"target_type": "unit", "target_id": "u0000", "reason": "unit"},
            {"target_type": "boundary", "target_id": "b0000", "reason": "start"},
            {"target_type": "boundary", "target_id": "b0001", "reason": "shared"},
            {"target_type": "boundary", "target_id": "b0002", "reason": "final"},
        ],
    )

    assert [(target.owner_unit_id, target.boundary_roles) for target in targets] == [
        ("u0000", frozenset({"before", "after"})),
        ("u0001", frozenset({"after"})),
    ]
    assert len(targets[0].reasons) == 3


def test_unknown_audit_target_is_retried(changed_evidence):
    document = _document(changed_evidence)
    backend = FakeAuditBackend(
        [
            {
                "issues": [
                    {"target_type": "unit", "target_id": "u9999", "reason": "bad"}
                ]
            },
            {"issues": []},
        ]
    )

    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=FakeRepairPipeline([]),
        config=HarnessConfig(sequence_audit_max_attempts=2),
    )

    assert outcome.document["quality_status"] == "accepted"
    assert outcome.audit_attempts == 2


def test_audit_provider_failure_stays_pending_for_resume(changed_evidence):
    document = _document(changed_evidence)
    backend = FakeAuditBackend([AnnotationError("offline"), AnnotationError("offline")])
    outcome = reconcile_document(
        document,
        backend=backend,
        pipeline=FakeRepairPipeline([]),
        config=HarnessConfig(
            sequence_audit_max_attempts=2,
            sequence_repair_rounds=0,
        ),
    )

    assert outcome.document["quality_status"] == "pending"
    assert outcome.document["quality_provenance"] is None
    assert outcome.audit_attempts == 2
    assert outcome.technical_failures
    validate_document(outcome.document)
