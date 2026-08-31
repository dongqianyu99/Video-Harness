from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .evidence import validate_boundary_state_record, validate_evidence_record
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
        if annotation["status"] != "complete":
            raise ValueError(f"Evidence Unit {unit['unit_id']} is not complete")
        record = validate_evidence_record(record)
        before, after = unit_boundary_states(document, unit)
        if any(
            boundary["annotation"]["status"] != "complete"
            for boundary in (before, after)
        ):
            raise ValueError(
                f"Evidence Unit {unit['unit_id']} references an incomplete Boundary"
            )
        boundary_records = tuple(
            validate_boundary_state_record(
                _plain_json(boundary["annotation"]["record"])
            )
            for boundary in (before, after)
        )
        yield unit, before, after, record, boundary_records[0], boundary_records[1]
