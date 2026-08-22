from __future__ import annotations

from collections.abc import Mapping as MappingABC
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .evidence import evidence_is_trainable
from .renderer import RENDER_PROFILES, render_evidence_text
from .sampling import BEHAVIOR_DOCUMENT_SCHEMA_VERSION, validate_document

DATASET_SCHEMA_VERSION = "video-harness.robodojo-source"
PAIR_SCHEMA_VERSION = "video-harness.support-query-pair"

_DATASET_KEYS = frozenset(
    {
        "schema_version",
        "task_scope",
        "episodes",
        "tasks",
        "frames",
        "fps",
        "episode_counts_by_task_index",
        "build_id",
        "source_dataset",
        "source_revision",
        "sample_hz",
        "supports_per_query",
        "document_camera",
        "benchmark_source_episodes",
        "selection",
    }
)

_PAIR_KEYS = frozenset(
    {
        "schema_version",
        "build_id",
        "pair_id",
        "task_index",
        "task_instruction",
        "query_episode_index",
        "support_episode_index",
        "support_rank",
        "support_document_id",
        "guide_schema_version",
    }
)


def _exact_object(
    value: Any,
    field: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    """Require one object to have exactly the expected keys."""

    if not isinstance(value, dict) or set(value) != expected_keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{field} must have exactly {sorted(expected_keys)}, got {actual}"
        )

    return value


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")

    return value


def _require_non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")

    return value


def _freeze_json(value: Any) -> Any:
    """Recursively convert JSON containers into immutable containers."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Make a temporary mutable copy for existing evidence validators/renderers."""

    if isinstance(value, MappingABC):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_dataset_metadata(
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Validate the build-level dataset metadata."""

    value = _exact_object(
        dataset,
        "dataset",
        _DATASET_KEYS,
    )

    if value["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unexpected dataset schema {value['schema_version']!r}")

    _require_non_empty_string(value["build_id"], "dataset.build_id")

    supports_per_query = value["supports_per_query"]
    if (
        not isinstance(supports_per_query, int)
        or isinstance(supports_per_query, bool)
        or supports_per_query != 1
    ):
        raise ValueError("dataset.supports_per_query must be exactly 1")

    return value


def _validate_support_binding(
    pair: dict[str, Any],
    *,
    build_id: str,
    guide_schema_version: str,
) -> SupportBinding:
    """Validate one static support-query pair."""

    value = _exact_object(
        pair,
        "pair",
        _PAIR_KEYS,
    )

    if value["schema_version"] != PAIR_SCHEMA_VERSION:
        raise ValueError(f"unexpected pair schema {value['schema_version']!r}")

    if value["build_id"] != build_id:
        raise ValueError(
            f"pair {value['pair_id']} build_id mismatch: "
            f"expected {build_id!r}, got {value['build_id']!r}"
        )

    if value["guide_schema_version"] != guide_schema_version:
        raise ValueError(
            f"pair {value['pair_id']} guide_schema_version mismatch: "
            f"expected {guide_schema_version!r}, "
            f"got {value['guide_schema_version']!r}"
        )

    pair_id = _require_non_empty_string(value["pair_id"], "pair.pair_id")
    task_instruction = _require_non_empty_string(
        value["task_instruction"],
        "pair.task_instruction",
    )
    support_document_id = _require_non_empty_string(
        value["support_document_id"],
        "pair.support_document_id",
    )

    query_episode_index = _require_non_negative_int(
        value["query_episode_index"],
        "pair.query_episode_index",
    )
    support_episode_index = _require_non_negative_int(
        value["support_episode_index"],
        "pair.support_episode_index",
    )

    if query_episode_index == support_episode_index:
        raise ValueError(
            f"pair {pair_id} must use different query and support episodes"
        )

    support_rank = _require_non_negative_int(
        value["support_rank"],
        "pair.support_rank",
    )
    if support_rank != 0:
        raise ValueError(
            f"pair {pair_id} must use support_rank=0 for supports_per_query=1"
        )

    task_index = _require_non_negative_int(
        value["task_index"],
        "pair.task_index",
    )

    return SupportBinding(
        build_id=value["build_id"],
        pair_id=pair_id,
        query_episode_index=query_episode_index,
        support_episode_index=support_episode_index,
        support_document_id=support_document_id,
        support_rank=support_rank,
        task_index=task_index,
        task_instruction=task_instruction,
        guide_schema_version=value["guide_schema_version"],
    )


def _require_artifact_file(path: Path) -> None:
    """Require one explicitly provided artifact file."""

    if not path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with path-aware parse errors."""

    _require_artifact_file(path)

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL objects with path- and line-aware parse errors."""

    _require_artifact_file(path)
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
                raise ValueError(
                    f"JSONL record in {path} at line {line_number} "
                    "must be a JSON object"
                )

            records.append(value)

    return records


@dataclass(frozen=True)
class GuideFrameRef:
    """One support-document frame referenced by a token-neutral GuidePlan."""

    document_id: str
    episode_index: int
    episode_frame_index: int
    timestamp_s: float


@dataclass(frozen=True)
class GuideSource:
    """Validated support document source metadata and canonical document."""

    document_id: str
    build_id: str
    schema_version: str
    episode_index: int
    task_index: int
    task_instruction: str
    document: Mapping[str, Any]


@dataclass(frozen=True)
class SupportBinding:
    """Static query-episode to support-document binding."""

    build_id: str
    pair_id: str
    query_episode_index: int
    support_episode_index: int
    support_document_id: str
    support_rank: int
    task_index: int
    task_instruction: str
    guide_schema_version: str


@dataclass(frozen=True)
class GuideArtifactBundle:
    """Validated, read-only artifact bundle for one build."""

    build_id: str
    supports_per_query: int
    dataset: Mapping[str, Any]
    documents: tuple[GuideSource, ...]
    support_bindings: tuple[SupportBinding, ...]


@dataclass(frozen=True)
class GuidePlanUnit:
    """One selected evidence transition in an ordered GuidePlan."""

    unit_id: str
    order: int
    before_slot: int
    after_slot: int
    transition_text: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class GuidePlan:
    """Token-neutral ordered Guide plan for one query episode."""

    query_episode_index: int
    support_document_id: str
    support_episode_index: int
    task_index: int
    task_instruction: str
    profile: str
    frames: tuple[GuideFrameRef, ...]
    units: tuple[GuidePlanUnit, ...]


def _validate_document_source(
    document: dict[str, Any],
    *,
    build_id: str,
) -> GuideSource:
    """Validate one canonical behavior document and wrap its source metadata."""

    value = validate_document(document)

    if value["build_id"] != build_id:
        raise ValueError(
            f"document {value['document_id']} build_id mismatch: "
            f"expected {build_id!r}, got {value['build_id']!r}"
        )

    source = value["source"]

    return GuideSource(
        document_id=value["document_id"],
        build_id=value["build_id"],
        schema_version=value["schema_version"],
        episode_index=source["episode_index"],
        task_index=source["task_index"],
        task_instruction=value["task_instruction"],
        document=_freeze_json(value),
    )


def _index_document_sources(
    documents: list[dict[str, Any]],
    *,
    build_id: str,
) -> tuple[GuideSource, ...]:
    """Validate and index all documents while preserving file order."""

    sources: list[GuideSource] = []
    seen_document_ids: set[str] = set()

    for line_number, document in enumerate(documents, start=1):
        source = _validate_document_source(
            document,
            build_id=build_id,
        )

        if source.document_id in seen_document_ids:
            raise ValueError(
                f"duplicate document_id {source.document_id!r} "
                f"at JSONL record {line_number}"
            )

        seen_document_ids.add(source.document_id)
        sources.append(source)

    return tuple(sources)


def _validate_binding_relationships(
    bindings: tuple[SupportBinding, ...],
    documents: tuple[GuideSource, ...],
) -> None:
    """Validate cross-artifact support-document relationships."""

    documents_by_id = {source.document_id: source for source in documents}
    seen_pair_ids: set[str] = set()
    seen_query_episodes: set[int] = set()

    for binding in bindings:
        if binding.pair_id in seen_pair_ids:
            raise ValueError(f"duplicate pair_id {binding.pair_id!r}")
        seen_pair_ids.add(binding.pair_id)

        if binding.query_episode_index in seen_query_episodes:
            raise ValueError(
                f"query episode {binding.query_episode_index} has more than one support"
            )
        seen_query_episodes.add(binding.query_episode_index)

        support = documents_by_id.get(binding.support_document_id)
        if support is None:
            raise ValueError(
                f"pair {binding.pair_id} references missing support document "
                f"{binding.support_document_id!r}"
            )

        if support.episode_index != binding.support_episode_index:
            raise ValueError(
                f"pair {binding.pair_id} support episode mismatch: "
                f"document has {support.episode_index}, "
                f"pair has {binding.support_episode_index}"
            )

        if support.task_index != binding.task_index:
            raise ValueError(
                f"pair {binding.pair_id} task index mismatch: "
                f"document has {support.task_index}, "
                f"pair has {binding.task_index}"
            )

        if support.task_instruction != binding.task_instruction:
            raise ValueError(
                f"pair {binding.pair_id} task instruction mismatch"
            )


def load_guide_artifact_bundle(
    *,
    dataset_path: Path,
    documents_path: Path,
    pairs_path: Path,
) -> GuideArtifactBundle:
    """Load and cross-validate one explicit VideoHarness artifact bundle."""

    dataset = _validate_dataset_metadata(_read_json(dataset_path))
    build_id = dataset["build_id"]

    documents = _index_document_sources(
        _read_jsonl(documents_path),
        build_id=build_id,
    )

    raw_pairs = _read_jsonl(pairs_path)
    bindings = tuple(
        _validate_support_binding(
            pair,
            build_id=build_id,
            guide_schema_version=BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
        )
        for pair in raw_pairs
    )
    _validate_binding_relationships(bindings, documents)

    return GuideArtifactBundle(
        build_id=build_id,
        supports_per_query=dataset["supports_per_query"],
        dataset=_freeze_json(dataset),
        documents=documents,
        support_bindings=bindings,
    )


def build_guide_plan(
    bundle: GuideArtifactBundle,
    *,
    query_episode_index: int,
    profile: str = "actuator",
) -> GuidePlan:
    """Select trainable support evidence and build a token-neutral GuidePlan."""

    if profile not in RENDER_PROFILES:
        raise ValueError(f"Unknown renderer profile: {profile}")

    binding = next(
        (
            candidate
            for candidate in bundle.support_bindings
            if candidate.query_episode_index == query_episode_index
        ),
        None,
    )
    if binding is None:
        raise ValueError(
            f"no support binding found for query episode {query_episode_index}"
        )

    source = next(
        (
            candidate
            for candidate in bundle.documents
            if candidate.document_id == binding.support_document_id
        ),
        None,
    )
    if source is None:
        raise ValueError(
            f"support document {binding.support_document_id!r} is not loaded"
        )

    frames: list[GuideFrameRef] = []
    frame_slots: dict[tuple[int, float], int] = {}
    units: list[GuidePlanUnit] = []

    def slot_for(frame: MappingABC[str, Any]) -> int:
        key = (
            frame["episode_frame_index"],
            float(frame["timestamp_s"]),
        )
        if key not in frame_slots:
            frame_slots[key] = len(frames)
            frames.append(
                GuideFrameRef(
                    document_id=source.document_id,
                    episode_index=source.episode_index,
                    episode_frame_index=frame["episode_frame_index"],
                    timestamp_s=float(frame["timestamp_s"]),
                )
            )
        return frame_slots[key]

    document = source.document
    for raw_unit in document["guidance_units"]:
        annotation = raw_unit["annotation"]
        if annotation["status"] != "complete":
            continue

        record = _thaw_json(annotation["record"])
        if not evidence_is_trainable(record):
            continue

        before_slot = slot_for(raw_unit["before"])
        after_slot = slot_for(raw_unit["after"])
        units.append(
            GuidePlanUnit(
                unit_id=raw_unit["unit_id"],
                order=raw_unit["order"],
                before_slot=before_slot,
                after_slot=after_slot,
                transition_text=render_evidence_text(
                    record,
                    profile,
                ),
                provenance=annotation["provenance"],
            )
        )

    if not units:
        raise ValueError(
            f"support document {source.document_id!r} has no trainable units"
        )

    return GuidePlan(
        query_episode_index=query_episode_index,
        support_document_id=source.document_id,
        support_episode_index=source.episode_index,
        task_index=source.task_index,
        task_instruction=source.task_instruction,
        profile=profile,
        frames=tuple(frames),
        units=tuple(units),
    )
