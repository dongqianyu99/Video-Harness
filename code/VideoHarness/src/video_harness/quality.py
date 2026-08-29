from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .evidence import boundary_state_is_usable, evidence_is_trainable
from .sampling import unit_boundary_states


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def accepted_transition_chain(
    document: Mapping[str, Any],
) -> Iterator[
    tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]
]:
    """Yield accepted transitions from a quality-accepted document."""

    if document["quality_status"] != "accepted":
        raise ValueError(
            f"document {document['document_id']!r} is not quality-accepted"
        )
    for unit in document["evidence_units"]:
        annotation = unit["annotation"]
        record = _plain_json(annotation["record"])
        if annotation["status"] != "complete" or not evidence_is_trainable(record):
            continue
        before, after = unit_boundary_states(document, unit)
        boundary_records = tuple(
            _plain_json(boundary["annotation"]["record"])
            for boundary in (before, after)
        )
        if any(
            boundary["annotation"]["status"] != "complete"
            or not boundary_state_is_usable(boundary_record)
            for boundary, boundary_record in zip(
                (before, after),
                boundary_records,
                strict=True,
            )
        ):
            raise ValueError(
                f"Evidence Unit {unit['unit_id']} references an unaccepted Boundary"
            )
        yield unit, before, after, record, boundary_records[0], boundary_records[1]
