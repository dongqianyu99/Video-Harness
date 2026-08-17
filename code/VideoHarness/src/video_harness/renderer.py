from __future__ import annotations

from typing import Any, Callable

from .evidence import evidence_is_trainable, validate_evidence_record
from .sampling import validate_document


RENDER_PROFILES = ("brief", "state-change", "instructional", "stage-card", "actuator-v0")


def render_evidence_text(record: dict[str, Any], profile: str = "brief") -> str:
    """Render one canonical evidence record without changing its semantics."""

    if profile not in RENDER_PROFILES:
        raise ValueError(f"Unknown renderer profile: {profile}")
    evidence = validate_evidence_record(record)
    observation = evidence["visual_observation"]
    if evidence["change_status"] != "changed" or observation["change"] is None:
        raise ValueError("Evidence does not contain a renderable visible change")

    if profile == "brief":
        return observation["change"]
    if profile == "state-change":
        return (
            f"Before: {observation['before']} "
            f"After: {observation['after']} "
            f"Visible change: {observation['change']}"
        )

    operation = evidence["operation_hint"]
    if profile == "actuator-v0":
        entity_text = (
            "; ".join(
                f"{entity['role']}={entity['name']} "
                f"[grounding={entity['grounding']}, support={entity['support']}]"
                for entity in evidence["entities"]
            )
            or "none recorded"
        )

        if operation is None:
            operation_text = (
                "Operation inference [support=none recorded]: none recorded."
            )
        else:
            operation_text = (
                f"Operation inference [support={operation['support']}]: "
                f"{operation['label']} — "
                f"{operation['description'].rstrip('.')}."
            )

        visibility_text = ", ".join(evidence["visibility_limits"]) or "none recorded"

        return "\n".join(
            [
                f"Observed before: {observation['before']}",
                f"Observed after: {observation['after']}",
                f"Visible change: {observation['change']}",
                f"Relevant entities: {entity_text}",
                operation_text,
                f"Visible end effector: {evidence['visible_end_effector']}.",
                f"Unobserved details: {visibility_text}.",
            ]
        )
    if profile == "instructional":
        if operation is None or operation["support"] != "visible_interaction":
            return f"Visible change: {observation['change']} Visible result: {observation['after']}"
        return (
            f"Operation hint ({operation['support']}): {operation['description']} "
            f"Visible result: {observation['after']}"
        )

    entity_text = "; ".join(
        f"{entity['role']}={entity['name']} ({entity['visual_description']})"
        for entity in evidence["entities"]
    ) or "none recorded"
    operation_text = (
        f"{operation['label']} / {operation['support']}: {operation['description']}"
        if operation is not None
        else "none recorded"
    )
    return (
        f"Before: {observation['before']} After: {observation['after']} "
        f"Visible change: {observation['change']} Entities: {entity_text}. "
        f"Operation hypothesis: {operation_text}. "
        f"Visible end effector: {evidence['visible_end_effector']}. "
        f"Task relevance: {evidence['task_relevance']}. "
        f"Visibility limits: {', '.join(evidence['visibility_limits'])}."
    )


def render_interleaved(
    document: dict[str, Any],
    frame_loader: Callable[[dict[str, Any], dict[str, Any]], Any],
    *,
    profile: str = "brief",
    allow_mock: bool = False,
    allow_ambiguous: bool = False,
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
        if not evidence_is_trainable(record, allow_ambiguous=allow_ambiguous):
            raise ValueError(f"Unit {unit['unit_id']} has no trainable visible change")

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
