from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .annotations import (
    RepairBackend,
    SequenceAuditRequest,
    SequenceAuditResult,
)
from .config import HarnessConfig
from .pipeline import EvidenceUnitPipeline, ExistingRepairOutcome
from .run_tracking import ApiCallBudgetExceeded
from .sampling import unit_boundary_states


@dataclass(frozen=True)
class DocumentReconciliationResult:
    document: dict[str, Any]
    issues: tuple[dict[str, str], ...]
    audit_attempts: int
    repair_rounds: int
    audit: SequenceAuditResult | None


def build_sequence_projection(document: dict[str, Any]) -> str:
    boundaries = []
    for boundary in document["boundary_states"]:
        annotation = boundary["annotation"]
        record = annotation.get("record")
        boundaries.append(
            {
                "boundary_id": boundary["boundary_id"],
                "frame": boundary["frame"],
                "observation": (
                    record.get("observation") if isinstance(record, dict) else None
                ),
                "quality_status": (
                    record.get("quality_status") if isinstance(record, dict) else None
                ),
            }
        )

    units = []
    for unit in document["evidence_units"]:
        annotation = unit["annotation"]
        record = annotation.get("record")
        units.append(
            {
                "unit_id": unit["unit_id"],
                "before_boundary_id": unit["before_boundary_id"],
                "after_boundary_id": unit["after_boundary_id"],
                "motion_summary": (
                    record.get("motion_summary") if isinstance(record, dict) else None
                ),
                "action_description": (
                    record.get("unit_interpretation", {}).get("action_description")
                    if isinstance(record, dict)
                    else None
                ),
                "task_role": (
                    record.get("unit_interpretation", {}).get("task_role")
                    if isinstance(record, dict)
                    else None
                ),
                "causal_validation": (
                    record.get("causal_validation")
                    if isinstance(record, dict)
                    else None
                ),
                "quality_status": (
                    record.get("quality_status") if isinstance(record, dict) else None
                ),
            }
        )
    return json.dumps(
        {
            "document_id": document["document_id"],
            "task_instruction": document["task_instruction"],
            "boundaries": boundaries,
            "evidence_units": units,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sequence_projection_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        build_sequence_projection(document).encode("utf-8")
    ).hexdigest()


def _intrinsic_issues(document: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for unit in document["evidence_units"]:
        annotation = unit["annotation"]
        record = annotation.get("record")
        if annotation["status"] != "complete" or not isinstance(record, dict):
            issues.append(
                {
                    "unit_id": unit["unit_id"],
                    "reason": "Evidence Unit annotation is incomplete or failed.",
                }
            )
            continue
        if record["quality_status"] != "accepted":
            issues.append(
                {
                    "unit_id": unit["unit_id"],
                    "reason": record["causal_validation"]["reason"],
                }
            )
        for boundary in unit_boundary_states(document, unit):
            boundary_record = boundary["annotation"].get("record")
            if (
                boundary["annotation"]["status"] != "complete"
                or not isinstance(boundary_record, dict)
                or boundary_record["quality_status"] != "accepted"
            ):
                issues.append(
                    {
                        "unit_id": unit["unit_id"],
                        "reason": (
                            f"Adjacent Boundary {boundary['boundary_id']} is not "
                            "quality-accepted."
                        ),
                    }
                )
                break
    return issues


def _deduplicate_issues(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, list[str]] = {}
    for issue in issues:
        merged.setdefault(issue["unit_id"], []).append(issue["reason"])
    return [
        {"unit_id": unit_id, "reason": " ".join(dict.fromkeys(reasons))}
        for unit_id, reasons in sorted(merged.items())
    ]


def _audit_document(
    document: dict[str, Any],
    backend: RepairBackend,
    config: HarnessConfig,
) -> tuple[SequenceAuditResult | None, int]:
    request = SequenceAuditRequest(
        canonical_sequence=build_sequence_projection(document),
        task_instruction=document["task_instruction"],
    )
    for attempt in range(1, config.sequence_audit_max_attempts + 1):
        try:
            return backend.audit_sequence(request), attempt
        except ApiCallBudgetExceeded:
            raise
        except Exception:  # noqa: BLE001,S112 - audit failure quarantines this document
            continue
    return None, config.sequence_audit_max_attempts


def _apply_repair(
    document: dict[str, Any],
    unit: dict[str, Any],
    issue_reason: str,
    outcome: ExistingRepairOutcome,
) -> None:
    assert outcome.canonical_evidence is not None
    assert outcome.result is not None
    source = {
        "provider": outcome.result.provider,
        "model": outcome.result.trace.response_model or outcome.result.requested_model,
        "prompt_version": outcome.result.prompt_version,
    }
    unit["annotation"]["record"] = outcome.canonical_evidence
    unit["annotation"]["status"] = "complete"
    unit["annotation"]["provenance"]["repair"] = {
        **source,
        "attempts": outcome.attempts,
        "reason": issue_reason,
    }
    before_boundary, after_boundary = unit_boundary_states(document, unit)
    for boundary, role, replacement in (
        (before_boundary, "before", outcome.before_boundary_record),
        (after_boundary, "after", outcome.after_boundary_record),
    ):
        if replacement is not None:
            boundary["annotation"]["record"] = replacement
            boundary["annotation"]["status"] = "complete"
            boundary["annotation"]["provenance"] = {
                **source,
                "source_unit_id": unit["unit_id"],
                "boundary_role": role,
            }


def reconcile_document(
    document: dict[str, Any],
    *,
    backend: RepairBackend,
    pipeline: EvidenceUnitPipeline,
    config: HarnessConfig,
) -> DocumentReconciliationResult:
    working = copy.deepcopy(document)
    total_audit_attempts = 0
    last_audit: SequenceAuditResult | None = None
    last_issues: list[dict[str, str]] = []
    repair_rounds = 0

    for round_index in range(config.sequence_repair_rounds + 1):
        audit, attempts = _audit_document(working, backend, config)
        total_audit_attempts += attempts
        last_audit = audit
        audit_issues = (
            [{"unit_id": "<document>", "reason": "Sequence audit provider failed."}]
            if audit is None
            else list(audit.audit["issues"])
        )
        valid_unit_ids = {unit["unit_id"] for unit in working["evidence_units"]}
        unknown = [
            issue for issue in audit_issues if issue["unit_id"] not in valid_unit_ids
        ]
        if unknown:
            audit_issues = [
                {
                    "unit_id": "<document>",
                    "reason": "Sequence audit returned an unknown Evidence Unit.",
                }
            ]
        last_issues = _deduplicate_issues(_intrinsic_issues(working) + audit_issues)
        if not last_issues:
            working["quality_status"] = "accepted"
            working["quality_provenance"] = _quality_provenance(
                backend,
                last_audit,
                total_audit_attempts,
                repair_rounds,
                working,
                [],
            )
            return DocumentReconciliationResult(
                working,
                (),
                total_audit_attempts,
                repair_rounds,
                last_audit,
            )
        if round_index >= config.sequence_repair_rounds:
            break

        repaired_any = False
        units_by_id = {unit["unit_id"]: unit for unit in working["evidence_units"]}
        for issue in last_issues:
            unit = units_by_id.get(issue["unit_id"])
            if unit is None or not isinstance(unit["annotation"].get("record"), dict):
                continue
            outcome = pipeline.repair_existing(
                working,
                unit,
                issue_reason=issue["reason"],
            )
            if outcome.canonical_evidence is None:
                continue
            _apply_repair(working, unit, issue["reason"], outcome)
            repaired_any = True
        repair_rounds += 1
        if not repaired_any:
            break

    working["quality_status"] = "quarantined"
    working["quality_provenance"] = _quality_provenance(
        backend,
        last_audit,
        total_audit_attempts,
        repair_rounds,
        working,
        last_issues,
    )
    return DocumentReconciliationResult(
        working,
        tuple(last_issues),
        total_audit_attempts,
        repair_rounds,
        last_audit,
    )


def _quality_provenance(
    backend: RepairBackend,
    audit: SequenceAuditResult | None,
    audit_attempts: int,
    repair_rounds: int,
    document: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "provider": backend.provider,
        "model": (
            audit.trace.response_model or audit.requested_model
            if audit is not None
            else backend.model
        ),
        "prompt_version": (
            audit.prompt_version
            if audit is not None
            else "video-harness.sequence-audit.v3"
        ),
        "audit_attempts": audit_attempts,
        "repair_rounds": repair_rounds,
        "sequence_sha256": sequence_projection_sha256(document),
        "issues": issues,
    }
