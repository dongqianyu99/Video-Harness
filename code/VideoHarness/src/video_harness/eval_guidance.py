"""Deterministic task-level Behavior Documents for policy evaluation."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .evidence import evidence_is_trainable
from .reader import GuideFrameRef, GuidePlanUnit
from .renderer import RENDER_PROFILES, render_evidence_text
from .sampling import validate_document


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Eval guidance artifact does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Empty JSONL line in {path} at line {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise TypeError(f"JSONL record in {path} at line {line_number} must be an object")
                records.append(record)
        return records

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"JSON artifact {path} must contain one object or a list of objects")


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class EvalGuidance:
    """The immutable first support document selected for one task."""

    build_id: str
    document_id: str
    task_index: int
    task_instruction: str
    source_episode_index: int
    document: Mapping[str, Any]


@dataclass(frozen=True)
class EvalGuidePlan:
    """Token-neutral model-facing plan for one task-level Guide."""

    support_document_id: str
    support_episode_index: int
    task_index: int
    task_instruction: str
    profile: str
    frames: tuple[GuideFrameRef, ...]
    units: tuple[GuidePlanUnit, ...]


@dataclass(frozen=True)
class EvalGuidanceCatalog:
    """Read-only task index for deterministic evaluation guidance."""

    build_id: str
    entries: tuple[EvalGuidance, ...]
    _by_task: Mapping[int, EvalGuidance]
    _by_instruction: Mapping[str, EvalGuidance]

    def resolve(
        self,
        *,
        task_index: int | None = None,
        task_instruction: str | None = None,
    ) -> EvalGuidance:
        if (task_index is None) == (task_instruction is None):
            raise ValueError("Provide exactly one of task_index or task_instruction")
        if task_index is not None:
            task_index = _require_nonnegative_int(task_index, "task_index")
            result = self._by_task.get(task_index)
        else:
            if not isinstance(task_instruction, str) or not task_instruction.strip():
                raise ValueError("task_instruction must be a non-empty string")
            result = self._by_instruction.get(task_instruction)
        if result is None:
            identity = task_index if task_index is not None else task_instruction
            raise ValueError(f"No evaluation guidance is available for task {identity!r}")
        return result

    def build_plan(
        self,
        *,
        task_index: int | None = None,
        task_instruction: str | None = None,
        profile: str = "actuator",
    ) -> EvalGuidePlan:
        if profile not in RENDER_PROFILES:
            raise ValueError(f"Unknown renderer profile: {profile}")
        guidance = self.resolve(task_index=task_index, task_instruction=task_instruction)

        frames: list[GuideFrameRef] = []
        frame_slots: dict[tuple[int, float], int] = {}
        units: list[GuidePlanUnit] = []

        def frame_slot(frame: Mapping[str, Any]) -> int:
            frame_index = _require_nonnegative_int(frame.get("episode_frame_index"), "episode_frame_index")
            timestamp = frame.get("timestamp_s")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise TypeError("timestamp_s must be numeric")
            key = (frame_index, float(timestamp))
            if key not in frame_slots:
                frame_slots[key] = len(frames)
                frames.append(
                    GuideFrameRef(
                        document_id=guidance.document_id,
                        episode_index=guidance.source_episode_index,
                        episode_frame_index=frame_index,
                        timestamp_s=float(timestamp),
                    )
                )
            return frame_slots[key]

        for raw_unit in guidance.document["guidance_units"]:
            annotation = raw_unit["annotation"]
            if annotation["status"] != "complete":
                continue
            record = _thaw_json(annotation["record"])
            if not evidence_is_trainable(record):
                continue
            units.append(
                GuidePlanUnit(
                    unit_id=raw_unit["unit_id"],
                    order=raw_unit["order"],
                    before_slot=frame_slot(raw_unit["before"]),
                    after_slot=frame_slot(raw_unit["after"]),
                    transition_text=render_evidence_text(record, profile),
                    provenance=annotation["provenance"],
                )
            )

        if not units:
            raise ValueError(f"Evaluation document {guidance.document_id!r} has no trainable guidance units")
        return EvalGuidePlan(
            support_document_id=guidance.document_id,
            support_episode_index=guidance.source_episode_index,
            task_index=guidance.task_index,
            task_instruction=guidance.task_instruction,
            profile=profile,
            frames=tuple(frames),
            units=tuple(units),
        )


def _dataset_first_episode_by_task(episodes: Iterable[Mapping[str, Any]]) -> dict[int, tuple[int, str]]:
    first: dict[int, tuple[int, str]] = {}
    instructions: dict[int, str] = {}
    seen_episodes: set[int] = set()
    for record in episodes:
        task_index = _require_nonnegative_int(record.get("task_index"), "episodes.task_index")
        episode_index = _require_nonnegative_int(record.get("episode_index"), "episodes.episode_index")
        instruction = record.get("task_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("episodes.task_instruction must be a non-empty string")
        if episode_index in seen_episodes:
            raise ValueError(f"Duplicate episode_index in episodes artifact: {episode_index}")
        seen_episodes.add(episode_index)
        known_instruction = instructions.setdefault(task_index, instruction)
        if known_instruction != instruction:
            raise ValueError(f"Task {task_index} has inconsistent task instructions")
        current = first.get(task_index)
        if current is None or episode_index < current[0]:
            first[task_index] = (episode_index, instruction)
    return first


def build_eval_guidance_catalog(
    documents: Iterable[Mapping[str, Any]],
    *,
    episodes: Iterable[Mapping[str, Any]] | None = None,
) -> EvalGuidanceCatalog:
    validated: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    seen_task_episodes: set[tuple[int, int]] = set()
    build_id: str | None = None
    for raw_document in documents:
        document = validate_document(copy.deepcopy(dict(raw_document)))
        if build_id is None:
            build_id = document["build_id"]
        elif document["build_id"] != build_id:
            raise ValueError("Evaluation documents must share one build_id")
        if document["document_id"] in seen_document_ids:
            raise ValueError(f"Duplicate evaluation document_id: {document['document_id']!r}")
        seen_document_ids.add(document["document_id"])
        key = (document["source"]["task_index"], document["source"]["episode_index"])
        if key in seen_task_episodes:
            raise ValueError(f"Duplicate evaluation task/episode document: {key}")
        seen_task_episodes.add(key)
        validated.append(document)

    if not validated or build_id is None:
        raise ValueError("Evaluation guidance requires at least one document")
    documents_by_task: dict[int, list[dict[str, Any]]] = {}
    for document in validated:
        documents_by_task.setdefault(document["source"]["task_index"], []).append(document)

    expected_first = _dataset_first_episode_by_task(episodes) if episodes is not None else None
    entries: list[EvalGuidance] = []
    seen_instructions: set[str] = set()
    for task_index in sorted(documents_by_task):
        task_documents = sorted(documents_by_task[task_index], key=lambda item: item["source"]["episode_index"])
        selected = task_documents[0]
        instruction = selected["task_instruction"]
        if any(document["task_instruction"] != instruction for document in task_documents):
            raise ValueError(f"Task {task_index} has inconsistent task instructions")
        if instruction in seen_instructions:
            raise ValueError(f"Task instruction is ambiguous across tasks: {instruction!r}")
        seen_instructions.add(instruction)
        if expected_first is not None:
            expected = expected_first.get(task_index)
            if expected is None:
                raise ValueError(f"Task {task_index} is absent from the episodes artifact")
            if selected["source"]["episode_index"] != expected[0]:
                raise ValueError(
                    f"Task {task_index} guidance episode {selected['source']['episode_index']} "
                    f"is not the dataset-first episode {expected[0]}"
                )
            if instruction != expected[1]:
                raise ValueError(f"Task {task_index} instruction disagrees with the episodes artifact")
        entries.append(
            EvalGuidance(
                build_id=build_id,
                document_id=selected["document_id"],
                task_index=task_index,
                task_instruction=instruction,
                source_episode_index=selected["source"]["episode_index"],
                document=_freeze_json(selected),
            )
        )

    entries_tuple = tuple(entries)
    return EvalGuidanceCatalog(
        build_id=build_id,
        entries=entries_tuple,
        _by_task=MappingProxyType({entry.task_index: entry for entry in entries_tuple}),
        _by_instruction=MappingProxyType({entry.task_instruction: entry for entry in entries_tuple}),
    )


def load_eval_guidance_catalog(
    documents_path: Path,
    *,
    episodes_path: Path | None = None,
) -> EvalGuidanceCatalog:
    documents = _read_json_records(documents_path)
    episodes = _read_json_records(episodes_path) if episodes_path is not None else None
    return build_eval_guidance_catalog(documents, episodes=episodes)
