from __future__ import annotations

from typing import Any, Callable

from .evidence import evidence_is_trainable, validate_evidence_record
from .sampling import validate_document


RENDER_PROFILES = ("brief", "state-change", "instructional", "stage-card", "actuator")


def render_evidence_text(record: dict[str, Any], profile: str = "brief") -> str:
    """Render one canonical evidence record without changing its semantics."""

    if profile not in RENDER_PROFILES:
        raise ValueError(f"Unknown renderer profile: {profile}")
    evidence = validate_evidence_record(record)
    endpoint = evidence["endpoint_observation"]
    interpretation = evidence["unit_interpretation"]
    if profile == "brief":
        return interpretation["action_description"]
    if profile == "state-change":
        return " ".join(
            [
                *(f"Before {view}: {endpoint['before'][view]}" for view in endpoint["before"]),
                *(f"After {view}: {endpoint['after'][view]}" for view in endpoint["after"]),
            ]
        )

    if profile == "actuator":
        lines = [f"Motion: {evidence['motion_summary']}"]
        lines.extend(
            f"Before {view}: {endpoint['before'][view]}" for view in endpoint["before"]
        )
        lines.extend(
            f"After {view}: {endpoint['after'][view]}" for view in endpoint["after"]
        )
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
        f"Motion: {evidence['motion_summary']} "
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
    rendered: list[dict[str, Any]] = []
    usable_statuses = {"complete", "mock"} if allow_mock else {"complete"}
    previous_after: dict[str, Any] | None = None
    for unit in document["guidance_units"]:
        before_ref = unit["before"]
        after_ref = unit["after"]
        annotation = unit.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("status") not in usable_statuses:
            raise ValueError(f"Unit {unit['unit_id']} has no usable evidence annotation")
        record = annotation.get("record")
        if not evidence_is_trainable(record):
            raise ValueError(f"Unit {unit['unit_id']} has no trainable transition evidence")

        if previous_after != before_ref:
            rendered.append(
                {
                    "type": "image",
                    "role": "before",
                    "frame_ref": before_ref,
                    "value": frame_loader(document, before_ref),
                }
            )
        rendered.append(
            {
                "type": "text",
                "role": "transition",
                "unit_id": unit["unit_id"],
                "profile": profile,
                "value": render_evidence_text(record, profile),
            }
        )
        rendered.append(
            {
                "type": "image",
                "role": "after",
                "frame_ref": after_ref,
                "value": frame_loader(document, after_ref),
            }
        )
        previous_after = after_ref
    return rendered
