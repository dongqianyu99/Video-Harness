from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .quality import accepted_transition_chain
from .renderer import render_boundary_view_texts, render_transition_text
from .sampling import validate_document


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: empty line"
                )
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}, "
                    f"column {exc.colno}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record in {path} at line {line_number} must be an object"
                )
            records.append(value)
    return records


def _read_document_directory(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not path.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {path}")
    files = sorted(path.glob("*/*.document.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"Document directory contains no episode files: {path}"
        )

    documents: list[tuple[Path, dict[str, Any]]] = []
    for document_path in files:
        records = _read_jsonl(document_path)
        if len(records) != 1:
            raise ValueError(
                "Episode Document file must contain exactly one record: "
                f"{document_path}"
            )
        documents.append((document_path.resolve(), records[0]))
    return documents


@dataclass(frozen=True, slots=True)
class GuideDocument:
    document_id: str
    document_path: Path
    document_sha256: str
    build_id: str
    source_episode_index: int
    task_index: int
    task_instruction: str
    document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GuideExclusion:
    document_id: str
    document_path: Path
    document_sha256: str
    build_id: str
    source_episode_index: int
    task_index: int
    task_instruction: str
    quality_status: str
    reason: str


@dataclass(frozen=True, slots=True)
class GuideDocumentCatalog:
    build_id: str
    documents: tuple[GuideDocument, ...]
    exclusions: tuple[GuideExclusion, ...]
    catalog_digest: str
    _by_document_id: Mapping[str, GuideDocument] = field(repr=False)
    _by_task: Mapping[int, tuple[GuideDocument, ...]] = field(repr=False)

    def by_document_id(self, document_id: str) -> GuideDocument:
        document_id = _require_non_empty_string(document_id, "document_id")
        try:
            return self._by_document_id[document_id]
        except KeyError as exc:
            raise ValueError(
                f"No accepted Guide Document has document_id={document_id!r}"
            ) from exc

    def documents_for_task(self, task_index: int) -> tuple[GuideDocument, ...]:
        task_index = _require_non_negative_int(task_index, "task_index")
        return self._by_task.get(task_index, ())


def _catalog_digest(
    documents: list[GuideDocument],
    exclusions: list[GuideExclusion],
) -> str:
    records = [
        {
            "document_id": document.document_id,
            "source_episode_index": document.source_episode_index,
            "task_index": document.task_index,
            "quality_status": "accepted",
            "document_sha256": document.document_sha256,
        }
        for document in documents
    ]
    records.extend(
        {
            "document_id": exclusion.document_id,
            "source_episode_index": exclusion.source_episode_index,
            "task_index": exclusion.task_index,
            "quality_status": exclusion.quality_status,
            "document_sha256": exclusion.document_sha256,
        }
        for exclusion in exclusions
    )
    records.sort(
        key=lambda record: (
            record["task_index"],
            record["source_episode_index"],
            record["document_id"],
        )
    )
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_guide_document_catalog(documents_root: Path) -> GuideDocumentCatalog:
    """Load the one canonical, document-only Guide catalog."""

    raw_documents = _read_document_directory(documents_root)
    documents: list[GuideDocument] = []
    exclusions: list[GuideExclusion] = []
    seen_document_ids: set[str] = set()
    seen_task_episodes: set[tuple[int, int]] = set()
    instructions_by_task: dict[int, str] = {}
    build_id: str | None = None

    for document_path, raw_document in raw_documents:
        try:
            document = validate_document(raw_document)
        except Exception as exc:
            raise ValueError(
                f"Invalid Behavior Document in {document_path}: {exc}"
            ) from exc

        document_id = _require_non_empty_string(
            document["document_id"], "document.document_id"
        )
        current_build_id = _require_non_empty_string(
            document["build_id"], "document.build_id"
        )
        source = document["source"]
        source_episode_index = _require_non_negative_int(
            source["episode_index"], "document.source.episode_index"
        )
        task_index = _require_non_negative_int(
            source["task_index"], "document.source.task_index"
        )
        task_instruction = _require_non_empty_string(
            document["task_instruction"], "document.task_instruction"
        )

        if build_id is None:
            build_id = current_build_id
        elif current_build_id != build_id:
            raise ValueError(
                f"Document {document_id!r} build_id mismatch: "
                f"expected {build_id!r}, got {current_build_id!r}"
            )
        if document_id in seen_document_ids:
            raise ValueError(f"Duplicate document_id {document_id!r}")
        seen_document_ids.add(document_id)
        task_episode = (task_index, source_episode_index)
        if task_episode in seen_task_episodes:
            raise ValueError(
                "Duplicate task/source episode Document: "
                f"task_index={task_index}, episode_index={source_episode_index}"
            )
        seen_task_episodes.add(task_episode)
        known_instruction = instructions_by_task.setdefault(
            task_index, task_instruction
        )
        if known_instruction != task_instruction:
            raise ValueError(
                f"Task {task_index} has inconsistent task instructions"
            )
        quality_status = document["quality_status"]
        document_sha256 = _canonical_json_sha256(document)
        identity = {
            "document_id": document_id,
            "document_path": document_path,
            "document_sha256": document_sha256,
            "build_id": current_build_id,
            "source_episode_index": source_episode_index,
            "task_index": task_index,
            "task_instruction": task_instruction,
        }

        if quality_status == "quarantined":
            exclusions.append(
                GuideExclusion(
                    **identity,
                    quality_status=quality_status,
                    reason="Document quality_status is quarantined",
                )
            )
            continue
        if quality_status != "accepted":
            raise ValueError(
                f"Final Document {document_id!r} has nonterminal "
                f"quality_status={quality_status!r}"
            )

        try:
            chain = tuple(accepted_transition_chain(document))
        except Exception as exc:
            raise ValueError(
                f"Accepted Document {document_id!r} has an invalid transition chain: {exc}"
            ) from exc
        if not chain:
            raise ValueError(
                f"Accepted Document {document_id!r} has no trainable Evidence Units"
            )

        documents.append(
            GuideDocument(
                **identity,
                document=_freeze_json(document),
            )
        )

    if build_id is None:
        raise ValueError("Guide document catalog is empty")
    if not documents:
        raise ValueError("Guide document catalog contains no accepted Documents")

    documents.sort(
        key=lambda document: (
            document.task_index,
            document.source_episode_index,
            document.document_id,
        )
    )
    exclusions.sort(
        key=lambda exclusion: (
            exclusion.task_index,
            exclusion.source_episode_index,
            exclusion.document_id,
        )
    )
    by_task: dict[int, list[GuideDocument]] = {}
    for document in documents:
        by_task.setdefault(document.task_index, []).append(document)

    return GuideDocumentCatalog(
        build_id=build_id,
        documents=tuple(documents),
        exclusions=tuple(exclusions),
        catalog_digest=_catalog_digest(documents, exclusions),
        _by_document_id=MappingProxyType(
            {document.document_id: document for document in documents}
        ),
        _by_task=MappingProxyType(
            {task: tuple(values) for task, values in by_task.items()}
        ),
    )


@dataclass(frozen=True, slots=True)
class GuideBoundary:
    boundary_id: str
    order: int
    slot: int
    episode_frame_index: int
    timestamp_s: float
    view_texts: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class GuidePlanUnit:
    unit_id: str
    order: int
    before_slot: int
    after_slot: int
    transition_text: str


@dataclass(frozen=True, slots=True)
class GuidePlan:
    document_id: str
    source_episode_index: int
    task_index: int
    task_instruction: str
    boundaries: tuple[GuideBoundary, ...]
    units: tuple[GuidePlanUnit, ...]


def build_guide_plan(
    catalog: GuideDocumentCatalog,
    *,
    document_id: str,
) -> GuidePlan:
    """Project one accepted canonical Document into a token-neutral GuidePlan."""

    source = catalog.by_document_id(document_id)
    boundaries: list[GuideBoundary] = []
    boundary_slots: dict[str, int] = {}
    units: list[GuidePlanUnit] = []

    def slot_for(
        boundary: Mapping[str, Any],
        boundary_record: dict[str, Any],
    ) -> int:
        boundary_id = _require_non_empty_string(
            boundary["boundary_id"], "boundary.boundary_id"
        )
        frame = boundary["frame"]
        episode_frame_index = _require_non_negative_int(
            frame["episode_frame_index"], "boundary.frame.episode_frame_index"
        )
        timestamp_s = float(frame["timestamp_s"])
        view_texts = render_boundary_view_texts(boundary_record)

        if boundary_id in boundary_slots:
            slot = boundary_slots[boundary_id]
            existing = boundaries[slot]
            if (
                existing.order != boundary["order"]
                or existing.episode_frame_index != episode_frame_index
                or existing.timestamp_s != timestamp_s
                or existing.view_texts != view_texts
            ):
                raise ValueError(
                    f"Shared Boundary {boundary_id!r} changed within one GuidePlan"
                )
            return slot

        slot = len(boundaries)
        boundary_slots[boundary_id] = slot
        boundaries.append(
            GuideBoundary(
                boundary_id=boundary_id,
                order=_require_non_negative_int(
                    boundary["order"], "boundary.order"
                ),
                slot=slot,
                episode_frame_index=episode_frame_index,
                timestamp_s=timestamp_s,
                view_texts=view_texts,
            )
        )
        return slot

    for (
        raw_unit,
        before_boundary,
        after_boundary,
        record,
        before_record,
        after_record,
    ) in accepted_transition_chain(source.document):
        units.append(
            GuidePlanUnit(
                unit_id=_require_non_empty_string(
                    raw_unit["unit_id"], "evidence_unit.unit_id"
                ),
                order=_require_non_negative_int(
                    raw_unit["order"], "evidence_unit.order"
                ),
                before_slot=slot_for(before_boundary, before_record),
                after_slot=slot_for(after_boundary, after_record),
                transition_text=render_transition_text(record),
            )
        )

    if not units:
        raise ValueError(
            f"Accepted Document {source.document_id!r} has no trainable Evidence Units"
        )

    return GuidePlan(
        document_id=source.document_id,
        source_episode_index=source.source_episode_index,
        task_index=source.task_index,
        task_instruction=source.task_instruction,
        boundaries=tuple(boundaries),
        units=tuple(units),
    )
