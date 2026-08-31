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
    technical_failures: tuple[dict[str, str], ...] = ()


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
                "detail_observation": (
                    record.get("detail_observation")
                    if isinstance(record, dict)
                    else None
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
    if document["status"] != "annotated":
        return [
            {
                "target_type": "document",
                "target_id": "<document>",
                "reason": "Only a fully compiled real document can be audited.",
            }
        ]
    for unit in document["evidence_units"]:
        annotation = unit["annotation"]
        record = annotation.get("record")
        if annotation["status"] != "complete" or not isinstance(record, dict):
            issues.append(
                {
                    "target_type": "unit",
                    "target_id": unit["unit_id"],
                    "reason": "Evidence Unit annotation is incomplete or failed.",
                }
            )
    for boundary in document["boundary_states"]:
        boundary_record = boundary["annotation"].get("record")
        if (
            boundary["annotation"]["status"] != "complete"
            or not isinstance(boundary_record, dict)
        ):
            issues.append(
                {
                    "target_type": "boundary",
                    "target_id": boundary["boundary_id"],
                    "reason": "Boundary annotation is incomplete or failed.",
                }
            )
    return issues


def _deduplicate_issues(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], list[str]] = {}
    for issue in issues:
        key = (issue["target_type"], issue["target_id"])
        merged.setdefault(key, []).append(issue["reason"])
    return [
        {
            "target_type": target_type,
            "target_id": target_id,
            "reason": " ".join(dict.fromkeys(reasons)),
        }
        for (target_type, target_id), reasons in sorted(merged.items())
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
    valid_targets = {
        "unit": {unit["unit_id"] for unit in document["evidence_units"]},
        "boundary": {
            boundary["boundary_id"] for boundary in document["boundary_states"]
        },
    }
    for attempt in range(1, config.sequence_audit_max_attempts + 1):
        try:
            result = backend.audit_sequence(request)
        except ApiCallBudgetExceeded:
            raise
        except Exception:  # noqa: BLE001,S112 - audit failure remains resumable
            continue
        if all(
            issue["target_id"] in valid_targets[issue["target_type"]]
            for issue in result.audit["issues"]
        ):
            return result, attempt
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


@dataclass(frozen=True)
class RepairTarget:
    owner_unit_id: str
    reasons: tuple[str, ...]
    boundary_roles: frozenset[str]


def _repair_targets(
    document: dict[str, Any],
    issues: list[dict[str, str]],
) -> tuple[RepairTarget, ...]:
    grouped: dict[str, dict[str, Any]] = {}
    units = document["evidence_units"]
    boundary_orders = {
        boundary["boundary_id"]: boundary["order"]
        for boundary in document["boundary_states"]
    }
    for issue in issues:
        roles: set[str] = set()
        if issue["target_type"] == "unit":
            owner = issue["target_id"]
        else:
            order = boundary_orders[issue["target_id"]]
            if order == 0:
                owner = units[0]["unit_id"]
                roles.add("before")
            else:
                owner = units[order - 1]["unit_id"]
                roles.add("after")
        item = grouped.setdefault(owner, {"reasons": [], "roles": set()})
        item["reasons"].append(
            f"{issue['target_type']} {issue['target_id']}: {issue['reason']}"
        )
        item["roles"].update(roles)
    return tuple(
        RepairTarget(
            owner_unit_id=owner,
            reasons=tuple(dict.fromkeys(item["reasons"])),
            boundary_roles=frozenset(item["roles"]),
        )
        for owner, item in sorted(grouped.items())
    )


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
    technical_failures: list[dict[str, str]] = []

    if working["status"] != "annotated":
        last_issues = _intrinsic_issues(working)
        working["quality_status"] = "quarantined"
        working["quality_provenance"] = _quality_provenance(
            backend, None, 0, 0, working, last_issues
        )
        return DocumentReconciliationResult(
            working,
            tuple(last_issues),
            0,
            0,
            None,
        )

    for round_index in range(config.sequence_repair_rounds + 1):
        audit, attempts = _audit_document(working, backend, config)
        total_audit_attempts += attempts
        last_audit = audit
        if audit is None:
            technical_failures.append(
                {
                    "target_id": "<document>",
                    "error": "Sequence audit provider failed after retries.",
                }
            )
            working["quality_status"] = "pending"
            working["quality_provenance"] = None
            return DocumentReconciliationResult(
                working,
                (),
                total_audit_attempts,
                repair_rounds,
                None,
                tuple(technical_failures),
            )
        last_issues = _deduplicate_issues(list(audit.audit["issues"]))
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
                tuple(technical_failures),
            )
        if round_index >= config.sequence_repair_rounds:
            break

        repaired_any = False
        units_by_id = {unit["unit_id"]: unit for unit in working["evidence_units"]}
        round_failures: list[dict[str, str]] = []
        for target in _repair_targets(working, last_issues):
            unit = units_by_id[target.owner_unit_id]
            reason = " ".join(target.reasons)
            outcome = pipeline.repair_target(
                working,
                unit,
                issue_reason=reason,
                allowed_boundary_replacements=target.boundary_roles,
                required_boundary_replacements=target.boundary_roles,
            )
            if outcome.error is not None:
                round_failures.append(
                    {
                        "target_id": unit["unit_id"],
                        "error": f"Targeted repair provider failed: {outcome.error}",
                    }
                )
            if outcome.canonical_evidence is None:
                continue
            _apply_repair(working, unit, reason, outcome)
            repaired_any = True
        repair_rounds += 1
        if round_failures:
            technical_failures.extend(round_failures)
            working["quality_status"] = "pending"
            working["quality_provenance"] = None
            return DocumentReconciliationResult(
                working,
                tuple(last_issues),
                total_audit_attempts,
                repair_rounds,
                last_audit,
                tuple(technical_failures),
            )
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
        tuple(technical_failures),
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
            else "video-harness.sequence-audit.v6"
        ),
        "audit_attempts": audit_attempts,
        "repair_rounds": repair_rounds,
        "sequence_sha256": sequence_projection_sha256(document),
        "issues": issues,
    }
