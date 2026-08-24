from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .evidence import (
    boundary_state_is_usable,
    evidence_is_trainable,
    validate_boundary_state_record,
    validate_evidence_record,
)
from .sampling import unit_boundary_states, validate_document

RENDER_PROFILES = (
    "brief",
    "state-change",
    "instructional",
    "evidence-card",
    "actuator",
)


def render_boundary_text(record: dict[str, Any]) -> str:
    boundary = validate_boundary_state_record(record)
    return "\n".join(
        f"{view}: {description}"
        for view, description in boundary["observation"].items()
    )


def render_evidence_text(
    record: dict[str, Any],
    profile: str = "brief",
) -> str:
    """Render one canonical evidence record without changing its semantics."""

    if profile not in RENDER_PROFILES:
        raise ValueError(f"Unknown renderer profile: {profile}")
    evidence = validate_evidence_record(record)
    interpretation = evidence["unit_interpretation"]
    motion_summary = evidence["resolved_motion_summary"] or evidence["motion_summary"]
    if profile == "brief":
        return interpretation["action_description"]
    if profile == "state-change":
        return (
            f"Motion: {motion_summary} Action: {interpretation['action_description']}"
        )

    if profile == "actuator":
        lines = [f"Motion: {motion_summary}"]
        if evidence["detail_observation"] is not None:
            lines.append(f"Detail: {evidence['detail_observation']}")
        lines.extend(
            [
                f"Action: {interpretation['action_description']}",
                f"Task role: {interpretation['task_role']}",
            ]
        )
        return "\n".join(lines)
    if profile == "instructional":
        return (
            f"Action: {interpretation['action_description']} "
            f"Task role: {interpretation['task_role']}"
        )

    return (
        f"Motion: {motion_summary} "
        f"Action: {interpretation['action_description']} "
        f"Task role: {interpretation['task_role']} "
        f"Causal validation: {evidence['causal_validation']['status']} — "
        f"{evidence['causal_validation']['reason']}"
    )


def render_interleaved(
    document: dict[str, Any],
    frame_loader: Callable[[dict[str, Any], dict[str, Any]], Any],
    *,
    profile: str = "brief",
    allow_mock: bool = False,
) -> list[dict[str, Any]]:
    """Resolve a sidecar into image-text-image elements for an Actuator adapter."""

    if profile not in RENDER_PROFILES:
        raise ValueError(f"Unknown renderer profile: {profile}")
    validate_document(document)
    if document["quality_status"] != "accepted" and not (
        allow_mock and document["status"] == "mock-annotated"
    ):
        raise ValueError("Behavior Document is not quality-accepted")
    rendered: list[dict[str, Any]] = []
    usable_statuses = {"complete", "mock"} if allow_mock else {"complete"}
    emitted_boundaries: set[str] = set()
    for unit in document["evidence_units"]:
        before_boundary, after_boundary = unit_boundary_states(document, unit)
        annotation = unit.get("annotation")
        if (
            not isinstance(annotation, dict)
            or annotation.get("status") not in usable_statuses
        ):
            raise ValueError(
                f"Evidence Unit {unit['unit_id']} has no usable evidence annotation"
            )
        record = annotation.get("record")
        if not evidence_is_trainable(record) and not (
            allow_mock and annotation.get("status") == "mock"
        ):
            raise ValueError(
                f"Evidence Unit {unit['unit_id']} has no trainable transition evidence"
            )

        for boundary in (before_boundary,):
            if boundary["boundary_id"] in emitted_boundaries:
                continue
            boundary_annotation = boundary["annotation"]
            if boundary_annotation["status"] not in usable_statuses or (
                not boundary_state_is_usable(boundary_annotation["record"])
                and not (allow_mock and boundary_annotation["status"] == "mock")
            ):
                raise ValueError(
                    f"Boundary State {boundary['boundary_id']} has no usable observation"
                )
            rendered.extend(
                [
                    {
                        "type": "image",
                        "role": "boundary",
                        "boundary_id": boundary["boundary_id"],
                        "frame_ref": boundary["frame"],
                        "value": frame_loader(document, boundary["frame"]),
                    },
                    {
                        "type": "text",
                        "role": "boundary_state",
                        "boundary_id": boundary["boundary_id"],
                        "value": render_boundary_text(boundary_annotation["record"]),
                    },
                ]
            )
            emitted_boundaries.add(boundary["boundary_id"])
        rendered.append(
            {
                "type": "text",
                "role": "transition",
                "unit_id": unit["unit_id"],
                "profile": profile,
                "value": render_evidence_text(
                    record,
                    profile,
                ),
            }
        )
        if after_boundary["boundary_id"] not in emitted_boundaries:
            boundary_annotation = after_boundary["annotation"]
            if boundary_annotation["status"] not in usable_statuses or (
                not boundary_state_is_usable(boundary_annotation["record"])
                and not (allow_mock and boundary_annotation["status"] == "mock")
            ):
                raise ValueError(
                    f"Boundary State {after_boundary['boundary_id']} has no usable observation"
                )
            rendered.extend(
                [
                    {
                        "type": "image",
                        "role": "boundary",
                        "boundary_id": after_boundary["boundary_id"],
                        "frame_ref": after_boundary["frame"],
                        "value": frame_loader(document, after_boundary["frame"]),
                    },
                    {
                        "type": "text",
                        "role": "boundary_state",
                        "boundary_id": after_boundary["boundary_id"],
                        "value": render_boundary_text(boundary_annotation["record"]),
                    },
                ]
            )
            emitted_boundaries.add(after_boundary["boundary_id"])
    return rendered
