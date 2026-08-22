from __future__ import annotations

import copy
from typing import Any

from .camera_contract import CAMERA_VIEWS


EVIDENCE_SCHEMA_VERSION = "video-harness.evidence"
INSPECTION_SCHEMA_VERSION = "video-harness.inspection"

CAUSAL_VALIDATION_STATUSES = ("pass", "retry")
REVIEW_STATUSES = ("accepted", "needs_review")
DETAIL_REASONS = (
    "gripper_object",
    "insertion",
    "button_contact",
    "release",
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
    return {
        view: _text(descriptions[view], f"{field}.{view}")
        for view in CAMERA_VIEWS
    }


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
            "endpoint_observation",
            "detail_observation",
            "unit_interpretation",
            "causal_validation",
        },
    )
    endpoint = _exact_object(
        record["endpoint_observation"],
        "call2.endpoint_observation",
        {"before", "after"},
    )
    detail = record["detail_observation"]
    if detail is not None:
        detail = _text(detail, "call2.detail_observation")
    interpretation = _exact_object(
        record["unit_interpretation"],
        "call2.unit_interpretation",
        {"action_description", "task_role"},
    )
    validation = _exact_object(
        record["causal_validation"],
        "call2.causal_validation",
        {"status", "reason"},
    )
    return {
        "endpoint_observation": {
            "before": _view_descriptions(
                endpoint["before"], "call2.endpoint_observation.before"
            ),
            "after": _view_descriptions(
                endpoint["after"], "call2.endpoint_observation.after"
            ),
        },
        "detail_observation": detail,
        "unit_interpretation": {
            "action_description": _text(
                interpretation["action_description"],
                "call2.unit_interpretation.action_description",
            ),
            "task_role": _text(
                interpretation["task_role"],
                "call2.unit_interpretation.task_role",
            ),
        },
        "causal_validation": {
            "status": _enum(
                validation["status"],
                "call2.causal_validation.status",
                CAUSAL_VALIDATION_STATUSES,
            ),
            "reason": _text(
                validation["reason"], "call2.causal_validation.reason"
            ),
        },
    }


def compose_evidence_record(
    motion_summary: str,
    call2_record: Any,
    *,
    review_status: str,
) -> dict[str, Any]:
    call2 = validate_call2_record(copy.deepcopy(call2_record))
    status = _enum(review_status, "review_status", REVIEW_STATUSES)
    causal_status = call2["causal_validation"]["status"]
    if (status == "accepted") != (causal_status == "pass"):
        raise EvidenceValidationError(
            "review_status must be accepted exactly when causal_validation passes"
        )
    return {
        "motion_summary": _text(motion_summary, "motion_summary"),
        **call2,
        "review_status": status,
    }


def validate_evidence_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "evidence",
        {
            "motion_summary",
            "endpoint_observation",
            "detail_observation",
            "unit_interpretation",
            "causal_validation",
            "review_status",
        },
    )
    call2 = {
        field: record[field]
        for field in (
            "endpoint_observation",
            "detail_observation",
            "unit_interpretation",
            "causal_validation",
        )
    }
    return compose_evidence_record(
        record["motion_summary"],
        call2,
        review_status=record["review_status"],
    )


def mock_call2_record() -> dict[str, Any]:
    unavailable = "The mock backend does not interpret this camera endpoint."
    return {
        "endpoint_observation": {
            "before": {view: unavailable for view in CAMERA_VIEWS},
            "after": {view: unavailable for view in CAMERA_VIEWS},
        },
        "detail_observation": None,
        "unit_interpretation": {
            "action_description": "The mock backend does not infer the demonstrated action.",
            "task_role": "The mock backend does not infer this Unit's task role.",
        },
        "causal_validation": {
            "status": "retry",
            "reason": "The mock backend cannot validate a causal interpretation.",
        },
    }


def mock_evidence_record() -> dict[str, Any]:
    return compose_evidence_record(
        "The mock backend does not summarize the demonstrated motion.",
        mock_call2_record(),
        review_status="needs_review",
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
    return normalized["review_status"] == "accepted"
