from __future__ import annotations

import copy
from typing import Any

from .camera_contract import CAMERA_VIEWS

EVIDENCE_SCHEMA_VERSION = "video-harness.evidence.v6"
INSPECTION_SCHEMA_VERSION = "video-harness.inspection"
BOUNDARY_STATE_SCHEMA_VERSION = "video-harness.boundary-state.v2"


class EvidenceValidationError(ValueError):
    """Raised when a provider output violates a Harness evidence contract."""


def _exact_object(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EvidenceValidationError(
            f"{field} must have exactly {sorted(keys)}, got {actual}"
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
) -> dict[str, Any]:
    return {
        "observation": _view_descriptions(observation, "boundary_state.observation"),
    }


def validate_boundary_state_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "boundary_state",
        {"observation"},
    )
    return compose_boundary_state_record(record["observation"])


def validate_inspection_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "inspection",
        {
            "motion_summary",
            "interaction_window",
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

    detail = _exact_object(
        record["detail_request"],
        "inspection.detail_request",
        {"x_min", "y_min", "x_max", "y_max"},
    )
    x_min = _number(detail["x_min"], "inspection.detail_request.x_min")
    y_min = _number(detail["y_min"], "inspection.detail_request.y_min")
    x_max = _number(detail["x_max"], "inspection.detail_request.x_max")
    y_max = _number(detail["y_max"], "inspection.detail_request.y_max")
    if x_min >= x_max or y_min >= y_max:
        raise EvidenceValidationError("inspection detail ROI is empty or reversed")
    return {
        "motion_summary": motion_summary,
        "interaction_window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        "detail_request": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
    }


def validate_call2_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "call2",
        {
            "motion_summary",
            "before_boundary_observation",
            "after_boundary_observation",
            "detail_observation",
            "unit_interpretation",
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
    transition = _normalize_transition_fields(record, "call2")
    return {
        "motion_summary": _text(record["motion_summary"], "call2.motion_summary"),
        "before_boundary_observation": before_boundary,
        "after_boundary_observation": after_boundary,
        **transition,
    }


def _normalize_transition_fields(
    record: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    detail = _text(record["detail_observation"], f"{field}.detail_observation")
    interpretation = _exact_object(
        record["unit_interpretation"],
        f"{field}.unit_interpretation",
        {"action_description", "task_role"},
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
    }


def compose_evidence_record(
    call2_record: Any,
) -> dict[str, Any]:
    call2 = validate_call2_record(copy.deepcopy(call2_record))
    return {
        "motion_summary": call2["motion_summary"],
        "detail_observation": call2["detail_observation"],
        "unit_interpretation": call2["unit_interpretation"],
    }


def validate_evidence_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        "evidence",
        {
            "motion_summary",
            "detail_observation",
            "unit_interpretation",
        },
    )
    transition = _normalize_transition_fields(record, "evidence")
    return {
        "motion_summary": _text(record["motion_summary"], "motion_summary"),
        **transition,
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
        "detail_observation": "The mandatory detail sheet remains inconclusive.",
        "unit_interpretation": {
            "action_description": "The mock backend does not infer the demonstrated action.",
            "task_role": "The mock backend does not infer what this Evidence Unit contributes to the task.",
        },
    }


def mock_evidence_record() -> dict[str, Any]:
    return compose_evidence_record(mock_call2_record())


def mock_inspection_record() -> dict[str, Any]:
    return {
        "motion_summary": "The mock backend does not summarize the demonstrated motion.",
        "interaction_window": {"start_frame": 0, "end_frame": 25},
        "detail_request": {
            "x_min": 0.1,
            "y_min": 0.1,
            "x_max": 0.85,
            "y_max": 0.85,
        },
    }
