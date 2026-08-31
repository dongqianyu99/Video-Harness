from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .annotations import (
    RepairBackend,
    SequenceAuditRequest,
    SequenceAuditResult,
)
from .config import HarnessConfig
from .debug_artifacts import (
    SEQUENCE_AUDIT_REPORT_SCHEMA_VERSION,
    sequence_audit_report_path,
    write_sequence_audit_report,
)
from .pipeline import EvidenceUnitPipeline, ExistingRepairOutcome
from .prompts import SEQUENCE_AUDIT_PROMPT_VERSION
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
    boundaries = document["boundary_states"]
    units = document["evidence_units"]
    if len(boundaries) != len(units) + 1:
        raise ValueError("Sequence projection requires N Units and N+1 Boundaries")
    sequence: list[dict[str, Any]] = []
    for index, boundary in enumerate(boundaries):
        annotation = boundary["annotation"]
        record = annotation.get("record")
        sequence.append(
            {
                "type": "boundary",
                "boundary_id": boundary["boundary_id"],
                "frame": boundary["frame"],
                "observation": (
                    record.get("observation") if isinstance(record, dict) else None
                ),
            }
        )
        if index >= len(units):
            continue
        unit = units[index]
        annotation = unit["annotation"]
        record = annotation.get("record")
        sequence.append(
            {
                "type": "evidence_unit",
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
            "sequence": sequence,
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
    canonical_sequence: str,
) -> tuple[SequenceAuditResult | None, int]:
    request = SequenceAuditRequest(
        canonical_sequence=canonical_sequence,
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
    issues: tuple[dict[str, str], ...]
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
        item = grouped.setdefault(
            owner, {"issues": [], "reasons": [], "roles": set()}
        )
        item["issues"].append(issue)
        item["reasons"].append(
            f"{issue['target_type']} {issue['target_id']}: {issue['reason']}"
        )
        item["roles"].update(roles)
    return tuple(
        RepairTarget(
            owner_unit_id=owner,
            issues=tuple(item["issues"]),
            reasons=tuple(dict.fromkeys(item["reasons"])),
            boundary_roles=frozenset(item["roles"]),
        )
        for owner, item in sorted(grouped.items())
    )


def _issue_key(issue: dict[str, str]) -> tuple[str, str]:
    return issue["target_type"], issue["target_id"]


def _repair_committed(
    outcome: ExistingRepairOutcome, boundary_roles: frozenset[str]
) -> bool:
    if (
        outcome.error is not None
        or outcome.canonical_evidence is None
        or outcome.result is None
    ):
        return False
    return not (
        ("before" in boundary_roles and outcome.before_boundary_record is None)
        or ("after" in boundary_roles and outcome.after_boundary_record is None)
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
    sticky_issues: dict[tuple[str, str], dict[str, str]] = {}
    report_path = sequence_audit_report_path(
        enabled=config.debug,
        root=config.debug_root,
        document_id=working["document_id"],
    )
    debug_report: dict[str, Any] = {
        "schema_version": SEQUENCE_AUDIT_REPORT_SCHEMA_VERSION,
        "document_id": working["document_id"],
        "rounds": [],
        "final": None,
    }

    def save_debug(final: dict[str, Any] | None = None) -> None:
        if final is not None:
            debug_report["final"] = final
        write_sequence_audit_report(report_path, debug_report)

    if working["status"] != "annotated":
        last_issues = _intrinsic_issues(working)
        working["quality_status"] = "quarantined"
        working["quality_provenance"] = _quality_provenance(
            backend, None, 0, 0, working, last_issues
        )
        save_debug(
            {
                "quality_status": "quarantined",
                "active_issues": last_issues,
                "technical_failures": [],
            }
        )
        return DocumentReconciliationResult(
            working,
            tuple(last_issues),
            0,
            0,
            None,
        )

    for round_index in range(config.sequence_repair_rounds + 1):
        canonical_sequence = build_sequence_projection(working)
        projection_sha256 = hashlib.sha256(
            canonical_sequence.encode("utf-8")
        ).hexdigest()
        audit, attempts = _audit_document(
            working, backend, config, canonical_sequence
        )
        total_audit_attempts += attempts
        last_audit = audit
        reported_issues = (
            []
            if audit is None
            else _deduplicate_issues(list(audit.audit["issues"]))
        )
        round_report: dict[str, Any] = {
            "round_index": round_index,
            "sequence_sha256": projection_sha256,
            "canonical_sequence": json.loads(canonical_sequence),
            "audit": {
                "attempts": attempts,
                "provider": backend.provider if audit is None else audit.provider,
                "model": (
                    backend.model
                    if audit is None
                    else audit.trace.response_model or audit.requested_model
                ),
                "prompt_version": (
                    SEQUENCE_AUDIT_PROMPT_VERSION
                    if audit is None
                    else audit.prompt_version
                ),
                "trace": None if audit is None else asdict(audit.trace),
                "structured_output": None if audit is None else audit.audit,
                "reported_issues": reported_issues,
            },
            "carried_issues": [
                sticky_issues[key] for key in sorted(sticky_issues)
            ],
            "active_issues": [],
            "repair_targets": [],
        }
        debug_report["rounds"].append(round_report)
        if audit is None:
            technical_failures.append(
                {
                    "target_id": "<document>",
                    "error": "Sequence audit provider failed after retries.",
                }
            )
            working["quality_status"] = "pending"
            working["quality_provenance"] = None
            round_report["active_issues"] = [
                sticky_issues[key] for key in sorted(sticky_issues)
            ]
            save_debug(
                {
                    "quality_status": "pending",
                    "active_issues": round_report["active_issues"],
                    "technical_failures": technical_failures,
                }
            )
            return DocumentReconciliationResult(
                working,
                (),
                total_audit_attempts,
                repair_rounds,
                None,
                tuple(technical_failures),
            )
        active_issues = dict(sticky_issues)
        active_issues.update({_issue_key(issue): issue for issue in reported_issues})
        last_issues = [active_issues[key] for key in sorted(active_issues)]
        round_report["active_issues"] = last_issues
        save_debug()
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
            save_debug(
                {
                    "quality_status": "accepted",
                    "active_issues": [],
                    "technical_failures": technical_failures,
                }
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

        units_by_id = {unit["unit_id"]: unit for unit in working["evidence_units"]}
        round_failures: list[dict[str, str]] = []
        next_sticky_issues: dict[tuple[str, str], dict[str, str]] = {}
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
            committed = _repair_committed(outcome, target.boundary_roles)
            status = "committed" if committed else "unresolved"
            if outcome.error is not None:
                status = "technical_failed"
                round_failures.append(
                    {
                        "target_id": unit["unit_id"],
                        "error": f"Targeted repair provider failed: {outcome.error}",
                    }
                )
            if committed:
                _apply_repair(working, unit, reason, outcome)
            else:
                next_sticky_issues.update(
                    {_issue_key(issue): issue for issue in target.issues}
                )
            round_report["repair_targets"].append(
                {
                    "owner_unit_id": target.owner_unit_id,
                    "issues": list(target.issues),
                    "required_boundary_replacements": sorted(
                        target.boundary_roles
                    ),
                    "status": status,
                    "attempts": outcome.attempts,
                    "error": outcome.error,
                    "result": (
                        None if outcome.result is None else asdict(outcome.result)
                    ),
                }
            )
        repair_rounds += 1
        sticky_issues = next_sticky_issues
        save_debug()
        if round_failures:
            technical_failures.extend(round_failures)
            working["quality_status"] = "pending"
            working["quality_provenance"] = None
            save_debug(
                {
                    "quality_status": "pending",
                    "active_issues": last_issues,
                    "technical_failures": technical_failures,
                }
            )
            return DocumentReconciliationResult(
                working,
                tuple(last_issues),
                total_audit_attempts,
                repair_rounds,
                last_audit,
                tuple(technical_failures),
            )

    working["quality_status"] = "quarantined"
    working["quality_provenance"] = _quality_provenance(
        backend,
        last_audit,
        total_audit_attempts,
        repair_rounds,
        working,
        last_issues,
    )
    save_debug(
        {
            "quality_status": "quarantined",
            "active_issues": last_issues,
            "technical_failures": technical_failures,
        }
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
            else SEQUENCE_AUDIT_PROMPT_VERSION
        ),
        "audit_attempts": audit_attempts,
        "repair_rounds": repair_rounds,
        "sequence_sha256": sequence_projection_sha256(document),
        "issues": issues,
    }
