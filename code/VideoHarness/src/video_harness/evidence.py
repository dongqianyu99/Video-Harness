from __future__ import annotations

import copy
from typing import Any

from .camera_contract import CAMERA_VIEWS

EVIDENCE_SCHEMA_VERSION = "video-harness.evidence.v3"
INSPECTION_SCHEMA_VERSION = "video-harness.inspection"
BOUNDARY_STATE_SCHEMA_VERSION = "video-harness.boundary-state"

CAUSAL_VALIDATION_STATUSES = ("pass", "retry")
QUALITY_STATUSES = ("accepted", "quarantined")
DETAIL_REASONS = (
    "fine_spatial_detail",
    "temporal_disambiguation",
    "occlusion",
    "other",
)


class EvidenceValidationError(ValueError):
    """Raised when a provider output violates a Harness evidence contract."""


def _exact_object(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EvidenceValidationError(
            f"{field} must have exactly {sorted(keys)}, got {actual}"
        )
    return value


def _enum(value: Any, field: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise EvidenceValidationError(
            f"{field} must be one of {choices}, got {value!r}"
        )
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise EvidenceValidationError(
            f"{field} must be an integer in [{minimum}, {maximum}], got {value!r}"
        )
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(f"{field} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise EvidenceValidationError(f"{field} must be in [0, 1], got {number}")
    return number


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _view_descriptions(value: Any, field: str) -> dict[str, str]:
    descriptions = _exact_object(value, field, set(CAMERA_VIEWS))
    return {view: _text(descriptions[view], f"{field}.{view}") for view in CAMERA_VIEWS}


def compose_boundary_state_record(
    observation: Any,
    *,
    quality_status: str,
) -> dict[str, Any]:
    return {
        "observation": _view_descriptions(observation, "boundary_state.observation"),
        "quality_status": _enum(
            quality_status,
            "boundary_state.quality_status",
            QUALITY_STATUSES,
        ),
    }


def validate_boundary_state_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "boundary_state",
        {"observation", "quality_status"},
    )
    return compose_boundary_state_record(
        record["observation"],
        quality_status=record["quality_status"],
    )


def boundary_state_is_usable(record: Any) -> bool:
    normalized = validate_boundary_state_record(copy.deepcopy(record))
    return normalized["quality_status"] == "accepted"


def validate_inspection_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "inspection",
        {
            "motion_summary",
            "interaction_window",
            "needs_detail",
            "detail_request",
        },
    )
    motion_summary = _text(record["motion_summary"], "inspection.motion_summary")
    window = _exact_object(
        record["interaction_window"],
        "inspection.interaction_window",
        {"start_frame", "end_frame"},
    )
    start_frame = _integer(
        window["start_frame"],
        "inspection.interaction_window.start_frame",
        minimum=0,
        maximum=25,
    )
    end_frame = _integer(
        window["end_frame"],
        "inspection.interaction_window.end_frame",
        minimum=0,
        maximum=25,
    )
    if start_frame > end_frame:
        raise EvidenceValidationError("inspection interaction window is reversed")

    needs_detail = record["needs_detail"]
    if not isinstance(needs_detail, bool):
        raise EvidenceValidationError("inspection.needs_detail must be boolean")

    detail = record["detail_request"]
    normalized_detail: dict[str, Any] | None = None
    if detail is not None:
        detail = _exact_object(
            detail,
            "inspection.detail_request",
            {"x_min", "y_min", "x_max", "y_max", "reason"},
        )
        x_min = _number(detail["x_min"], "inspection.detail_request.x_min")
        y_min = _number(detail["y_min"], "inspection.detail_request.y_min")
        x_max = _number(detail["x_max"], "inspection.detail_request.x_max")
        y_max = _number(detail["y_max"], "inspection.detail_request.y_max")
        if x_min >= x_max or y_min >= y_max:
            raise EvidenceValidationError("inspection detail ROI is empty or reversed")
        normalized_detail = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "reason": _enum(
                detail["reason"],
                "inspection.detail_request.reason",
                DETAIL_REASONS,
            ),
        }
    if needs_detail != (normalized_detail is not None):
        raise EvidenceValidationError(
            "inspection needs_detail and detail_request must agree"
        )
    return {
        "motion_summary": motion_summary,
        "interaction_window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        "needs_detail": needs_detail,
        "detail_request": normalized_detail,
    }


def validate_call2_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "call2",
        {
            "motion_summary",
            "before_boundary_observation",
            "after_boundary_observation",
            "boundary_conflicts",
            "detail_observation",
            "unit_interpretation",
            "causal_validation",
        },
    )
    before_boundary = record["before_boundary_observation"]
    if before_boundary is not None:
        before_boundary = _view_descriptions(
            before_boundary,
            "call2.before_boundary_observation",
        )
    after_boundary = record["after_boundary_observation"]
    if after_boundary is not None:
        after_boundary = _view_descriptions(
            after_boundary,
            "call2.after_boundary_observation",
        )
    conflicts = _normalize_boundary_conflicts(
        record["boundary_conflicts"],
        "call2.boundary_conflicts",
    )
    transition = _normalize_transition_fields(record, "call2")
    return {
        "motion_summary": _text(record["motion_summary"], "call2.motion_summary"),
        "before_boundary_observation": before_boundary,
        "after_boundary_observation": after_boundary,
        "boundary_conflicts": conflicts,
        **transition,
    }


def _normalize_boundary_conflicts(value: Any, field: str) -> dict[str, str | None]:
    conflicts = _exact_object(value, field, {"before", "after"})
    return {
        role: (
            None
            if conflicts[role] is None
            else _text(conflicts[role], f"{field}.{role}")
        )
        for role in ("before", "after")
    }


def _normalize_transition_fields(
    record: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    detail = record["detail_observation"]
    if detail is not None:
        detail = _text(detail, f"{field}.detail_observation")
    interpretation = _exact_object(
        record["unit_interpretation"],
        f"{field}.unit_interpretation",
        {"action_description", "task_role"},
    )
    validation = _exact_object(
        record["causal_validation"],
        f"{field}.causal_validation",
        {"status", "reason"},
    )
    return {
        "detail_observation": detail,
        "unit_interpretation": {
            "action_description": _text(
                interpretation["action_description"],
                f"{field}.unit_interpretation.action_description",
            ),
            "task_role": _text(
                interpretation["task_role"],
                f"{field}.unit_interpretation.task_role",
            ),
        },
        "causal_validation": {
            "status": _enum(
                validation["status"],
                f"{field}.causal_validation.status",
                CAUSAL_VALIDATION_STATUSES,
            ),
            "reason": _text(validation["reason"], f"{field}.causal_validation.reason"),
        },
    }


def compose_evidence_record(
    call2_record: Any,
    *,
    quality_status: str,
) -> dict[str, Any]:
    call2 = validate_call2_record(copy.deepcopy(call2_record))
    status = _enum(quality_status, "quality_status", QUALITY_STATUSES)
    causal_status = call2["causal_validation"]["status"]
    if (status == "accepted") != (causal_status == "pass"):
        raise EvidenceValidationError(
            "quality_status must be accepted exactly when causal_validation passes"
        )
    return {
        "motion_summary": call2["motion_summary"],
        "detail_observation": call2["detail_observation"],
        "boundary_conflicts": call2["boundary_conflicts"],
        "unit_interpretation": call2["unit_interpretation"],
        "causal_validation": call2["causal_validation"],
        "quality_status": status,
    }


def validate_evidence_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "evidence",
        {
            "motion_summary",
            "boundary_conflicts",
            "detail_observation",
            "unit_interpretation",
            "causal_validation",
            "quality_status",
        },
    )
    transition = _normalize_transition_fields(record, "evidence")
    conflicts = _normalize_boundary_conflicts(
        record["boundary_conflicts"],
        "evidence.boundary_conflicts",
    )
    status = _enum(record["quality_status"], "quality_status", QUALITY_STATUSES)
    causal_status = transition["causal_validation"]["status"]
    if (status == "accepted") != (causal_status == "pass"):
        raise EvidenceValidationError(
            "quality_status must be accepted exactly when causal_validation passes"
        )
    return {
        "motion_summary": _text(record["motion_summary"], "motion_summary"),
        "boundary_conflicts": conflicts,
        **transition,
        "quality_status": status,
    }


def mock_call2_record(
    *,
    include_before_boundary: bool = True,
    include_after_boundary: bool = True,
) -> dict[str, Any]:
    unavailable = "The mock backend does not interpret this Boundary State."
    return {
        "motion_summary": "The mock backend does not refine the demonstrated motion.",
        "before_boundary_observation": (
            {view: unavailable for view in CAMERA_VIEWS}
            if include_before_boundary
            else None
        ),
        "after_boundary_observation": (
            {view: unavailable for view in CAMERA_VIEWS}
            if include_after_boundary
            else None
        ),
        "boundary_conflicts": {"before": None, "after": None},
        "detail_observation": None,
        "unit_interpretation": {
            "action_description": "The mock backend does not infer the demonstrated action.",
            "task_role": "The mock backend does not infer what this Evidence Unit contributes to the task.",
        },
        "causal_validation": {
            "status": "retry",
            "reason": "The mock backend cannot validate a causal interpretation.",
        },
    }


def mock_evidence_record() -> dict[str, Any]:
    return compose_evidence_record(
        mock_call2_record(),
        quality_status="quarantined",
    )


def mock_inspection_record() -> dict[str, Any]:
    return {
        "motion_summary": "The mock backend does not summarize the demonstrated motion.",
        "interaction_window": {"start_frame": 0, "end_frame": 25},
        "needs_detail": False,
        "detail_request": None,
    }


def evidence_is_trainable(record: Any) -> bool:
    normalized = validate_evidence_record(copy.deepcopy(record))
    return normalized["quality_status"] == "accepted"
