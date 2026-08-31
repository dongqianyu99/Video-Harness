from __future__ import annotations

from typing import Any

from .camera_contract import CAMERA_VIEWS
from .evidence import (
    validate_boundary_state_record,
    validate_evidence_record,
)


def render_boundary_view_texts(record: dict[str, Any]) -> tuple[str, str, str]:
    boundary = validate_boundary_state_record(record)
    observations = boundary["observation"]
    return (
        observations[CAMERA_VIEWS[0]],
        observations[CAMERA_VIEWS[1]],
        observations[CAMERA_VIEWS[2]],
    )


def render_boundary_text(record: dict[str, Any]) -> str:
    return "\n".join(
        f"{view}: {description}"
        for view, description in zip(
            CAMERA_VIEWS,
            render_boundary_view_texts(record),
            strict=True,
        )
    )


def render_transition_text(record: dict[str, Any]) -> str:
    """Render the single model-facing transition representation."""

    evidence = validate_evidence_record(record)
    interpretation = evidence["unit_interpretation"]
    motion_summary = evidence["motion_summary"]
    return "\n".join(
        (
            f"Motion: {motion_summary}",
            f"Detail: {evidence['detail_observation']}",
            f"Action: {interpretation['action_description']}",
            f"Task role: {interpretation['task_role']}",
        )
    )
