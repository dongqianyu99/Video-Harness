from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .evidence import evidence_is_trainable
from .robodojo import EpisodeRecord, VideoSlice
from .sampling import BEHAVIOR_DOCUMENT_SCHEMA_VERSION, validate_document

TRAINING_SPLIT_SCHEMA_VERSION = "video-harness.training-split"
PAIR_SCHEMA_VERSION = "video-harness.support-query-pair"


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _rank(seed: int, namespace: str, task_index: int, *episode_indices: int) -> bytes:
    payload = ":".join(
        [str(seed), namespace, str(task_index), *(str(value) for value in episode_indices)]
    ).encode()
    return hashlib.sha256(payload).digest()


def episode_record_from_dict(value: Mapping[str, Any]) -> EpisodeRecord:
    try:
        videos = tuple(VideoSlice(**dict(video)) for video in value["videos"])
        return EpisodeRecord(
            episode_index=int(value["episode_index"]),
            task_index=int(value["task_index"]),
            task_instruction=str(value["task_instruction"]),
            task_kind=str(value["task_kind"]),
            length=int(value["length"]),
            dataset_from_index=int(value["dataset_from_index"]),
            dataset_to_index=int(value["dataset_to_index"]),
            data_path=str(value["data_path"]),
            videos=videos,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid EpisodeRecord JSON object") from exc


def _trainable_units(document: Mapping[str, Any]) -> int:
    count = 0
    for unit in document["guidance_units"]:
        annotation = unit["annotation"]
        if (
            annotation["status"] == "complete"
            and annotation["record"] is not None
            and evidence_is_trainable(annotation["record"])
        ):
            count += 1
    return count


def _index_documents(
    documents: Iterable[dict[str, Any]],
    *,
    build_id: str,
) -> dict[int, tuple[dict[str, Any], int]]:
    result: dict[int, tuple[dict[str, Any], int]] = {}
    document_ids: set[str] = set()
    for document in documents:
        validate_document(document)
        if document["build_id"] != build_id:
            raise ValueError(
                f"document {document['document_id']!r} build_id does not match {build_id!r}"
            )
        document_id = str(document["document_id"])
        if document_id in document_ids:
            raise ValueError(f"duplicate document_id={document_id!r}")
        document_ids.add(document_id)
        episode_index = int(document["source"]["episode_index"])
        if episode_index in result:
            raise ValueError(f"duplicate document source episode_index={episode_index}")
        result[episode_index] = (document, _trainable_units(document))
    return result


def _validate_records(records: Sequence[EpisodeRecord]) -> dict[int, list[EpisodeRecord]]:
    if not records:
        raise ValueError("records must not be empty")
    by_task: dict[int, list[EpisodeRecord]] = {}
    seen_episodes: set[int] = set()
    instructions: dict[int, str] = {}
    for record in records:
        if record.episode_index in seen_episodes:
            raise ValueError(f"duplicate episode_index={record.episode_index}")
        seen_episodes.add(record.episode_index)
        if record.dataset_to_index - record.dataset_from_index != record.length:
            raise ValueError(
                f"episode {record.episode_index} has inconsistent frame bounds"
            )
        expected_instruction = instructions.setdefault(
            record.task_index, record.task_instruction
        )
        if expected_instruction != record.task_instruction:
            raise ValueError(
                f"task_index={record.task_index} has inconsistent instructions"
            )
        by_task.setdefault(record.task_index, []).append(record)
    for task_records in by_task.values():
        task_records.sort(key=lambda record: record.episode_index)
    return by_task


def _balanced_assign(
    queries: Sequence[EpisodeRecord],
    supports: Sequence[tuple[EpisodeRecord, dict[str, Any], int]],
    *,
    seed: int,
    task_index: int,
) -> dict[int, tuple[EpisodeRecord, dict[str, Any], int]]:
    query_order = sorted(
        queries,
        key=lambda record: (
            -record.length,
            _rank(seed, "query-assignment-order", task_index, record.episode_index),
        ),
    )
    support_order = sorted(
        supports,
        key=lambda item: _rank(
            seed, "support-assignment-order", task_index, item[0].episode_index
        ),
    )
    counts = {support[0].episode_index: 0 for support in support_order}
    frame_loads = {support[0].episode_index: 0 for support in support_order}
    assignments: dict[int, tuple[EpisodeRecord, dict[str, Any], int]] = {}

    for query in query_order:
        support = min(
            support_order,
            key=lambda item: (
                counts[item[0].episode_index],
                frame_loads[item[0].episode_index],
                _rank(
                    seed,
                    "support-tie-break",
                    task_index,
                    query.episode_index,
                    item[0].episode_index,
                ),
            ),
        )
        support_episode = support[0].episode_index
        assignments[query.episode_index] = support
        counts[support_episode] += 1
        frame_loads[support_episode] += query.length
    return assignments


def build_training_split(
    records: Iterable[EpisodeRecord],
    documents: Iterable[dict[str, Any]],
    *,
    build_id: str,
    support_documents_per_task: int,
    heldout_documents_per_task: int,
    query_episodes_per_task: int | None = None,
    min_trainable_units: int = 1,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(build_id, str) or not build_id.strip():
        raise ValueError("build_id must be a non-empty string")
    support_documents_per_task = _require_nonnegative_int(
        support_documents_per_task, name="support_documents_per_task"
    )
    heldout_documents_per_task = _require_nonnegative_int(
        heldout_documents_per_task, name="heldout_documents_per_task"
    )
    min_trainable_units = _require_nonnegative_int(
        min_trainable_units, name="min_trainable_units"
    )
    seed = _require_nonnegative_int(seed, name="seed")
    if support_documents_per_task < 1:
        raise ValueError("support_documents_per_task must be at least one")
    if heldout_documents_per_task < 1:
        raise ValueError("heldout_documents_per_task must be at least one")
    if min_trainable_units < 1:
        raise ValueError("min_trainable_units must be at least one")
    if query_episodes_per_task is not None:
        query_episodes_per_task = _require_nonnegative_int(
            query_episodes_per_task, name="query_episodes_per_task"
        )
        if query_episodes_per_task < 1:
            raise ValueError("query_episodes_per_task must be at least one")

    records_tuple = tuple(records)
    by_task = _validate_records(records_tuple)
    documents_by_episode = _index_documents(documents, build_id=build_id)
    records_by_episode = {record.episode_index: record for record in records_tuple}

    for episode_index, (document, _) in documents_by_episode.items():
        record = records_by_episode.get(episode_index)
        if record is None:
            raise ValueError(
                f"document {document['document_id']!r} source episode is absent"
            )
        if document["source"]["task_index"] != record.task_index:
            raise ValueError(
                f"document {document['document_id']!r} task does not match episode metadata"
            )

    split_id = (
        f"{build_id}__support-{support_documents_per_task}__"
        f"heldout-{heldout_documents_per_task}__"
        f"query-{query_episodes_per_task if query_episodes_per_task is not None else 'remaining'}__"
        f"min-units-{min_trainable_units}__seed-{seed}"
    )
    task_manifests: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for task_index, task_records in sorted(by_task.items()):
        eligible: list[tuple[EpisodeRecord, dict[str, Any], int]] = []
        for record in task_records:
            document_info = documents_by_episode.get(record.episode_index)
            if document_info is None:
                continue
            document, trainable_units = document_info
            if trainable_units >= min_trainable_units:
                eligible.append((record, document, trainable_units))
        required_documents = support_documents_per_task + heldout_documents_per_task
        if len(eligible) < required_documents:
            raise ValueError(
                f"task_index={task_index} has {len(eligible)} eligible documents, "
                f"fewer than required {required_documents}"
            )

        ranked_eligible = sorted(
            eligible,
            key=lambda item: _rank(
                seed, "document-role", task_index, item[0].episode_index
            ),
        )
        heldout = ranked_eligible[:heldout_documents_per_task]
        supports = ranked_eligible[
            heldout_documents_per_task : heldout_documents_per_task
            + support_documents_per_task
        ]
        reserved_episodes = {
            item[0].episode_index for item in (*heldout, *supports)
        }
        query_candidates = [
            record
            for record in task_records
            if record.episode_index not in reserved_episodes
        ]
        ranked_queries = sorted(
            query_candidates,
            key=lambda record: _rank(
                seed, "query-role", task_index, record.episode_index
            ),
        )
        if query_episodes_per_task is not None:
            if len(ranked_queries) < query_episodes_per_task:
                raise ValueError(
                    f"task_index={task_index} has only {len(ranked_queries)} query candidates, "
                    f"fewer than requested {query_episodes_per_task}"
                )
            queries = ranked_queries[:query_episodes_per_task]
            unused = ranked_queries[query_episodes_per_task:]
        else:
            queries = ranked_queries
            unused = []
        if not queries:
            raise ValueError(f"task_index={task_index} has no train query episodes")

        assignments = _balanced_assign(
            queries,
            supports,
            seed=seed,
            task_index=task_index,
        )
        query_entries: list[dict[str, Any]] = []
        support_query_counts: Counter[int] = Counter()
        support_frame_counts: Counter[int] = Counter()
        for query in sorted(queries, key=lambda record: record.episode_index):
            support, document, _ = assignments[query.episode_index]
            pair_id = (
                f"t{task_index:02d}-q{query.episode_index:07d}-"
                f"s{support.episode_index:07d}"
            )
            support_query_counts[support.episode_index] += 1
            support_frame_counts[support.episode_index] += query.length
            query_entries.append(
                {
                    "episode_index": query.episode_index,
                    "length": query.length,
                    "pair_id": pair_id,
                    "support_episode_index": support.episode_index,
                    "support_document_id": document["document_id"],
                }
            )
            pairs.append(
                {
                    "schema_version": PAIR_SCHEMA_VERSION,
                    "build_id": build_id,
                    "pair_id": pair_id,
                    "task_index": task_index,
                    "task_instruction": query.task_instruction,
                    "query_episode_index": query.episode_index,
                    "support_episode_index": support.episode_index,
                    "support_rank": 0,
                    "support_document_id": document["document_id"],
                    "guide_schema_version": BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
                }
            )

        support_entries = [
            {
                "episode_index": record.episode_index,
                "document_id": document["document_id"],
                "trainable_units": trainable_units,
                "assigned_query_episodes": support_query_counts[record.episode_index],
                "assigned_query_frames": support_frame_counts[record.episode_index],
            }
            for record, document, trainable_units in sorted(
                supports, key=lambda item: item[0].episode_index
            )
        ]
        heldout_entries = [
            {
                "episode_index": record.episode_index,
                "document_id": document["document_id"],
                "trainable_units": trainable_units,
            }
            for record, document, trainable_units in sorted(
                heldout, key=lambda item: item[0].episode_index
            )
        ]
        query_loads = [entry["assigned_query_episodes"] for entry in support_entries]
        frame_loads = [entry["assigned_query_frames"] for entry in support_entries]
        task_manifests.append(
            {
                "task_index": task_index,
                "task_instruction": task_records[0].task_instruction,
                "train_support_documents": support_entries,
                "heldout_documents": heldout_entries,
                "train_queries": query_entries,
                "unused_episode_indices": sorted(
                    record.episode_index for record in unused
                ),
                "assignment_summary": {
                    "min_queries_per_support": min(query_loads),
                    "max_queries_per_support": max(query_loads),
                    "query_count_gap": max(query_loads) - min(query_loads),
                    "min_query_frames_per_support": min(frame_loads),
                    "max_query_frames_per_support": max(frame_loads),
                    "query_frame_gap": max(frame_loads) - min(frame_loads),
                },
            }
        )

    pairs.sort(key=lambda pair: (pair["task_index"], pair["query_episode_index"]))
    all_supports = {
        entry["episode_index"]
        for task in task_manifests
        for entry in task["train_support_documents"]
    }
    all_heldout = {
        entry["episode_index"]
        for task in task_manifests
        for entry in task["heldout_documents"]
    }
    all_queries = {
        entry["episode_index"]
        for task in task_manifests
        for entry in task["train_queries"]
    }
    if all_supports & all_heldout or all_supports & all_queries or all_heldout & all_queries:
        raise AssertionError("training split roles overlap")

    manifest = {
        "schema_version": TRAINING_SPLIT_SCHEMA_VERSION,
        "split_id": split_id,
        "build_id": build_id,
        "guide_schema_version": BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "pairing_strategy": "balanced-static-assignment",
        "binding_scope": "query_episode",
        "seed": seed,
        "task_scope": "all" if len(by_task) == 34 else "partial",
        "config": {
            "support_documents_per_task": support_documents_per_task,
            "heldout_documents_per_task": heldout_documents_per_task,
            "query_episodes_per_task": query_episodes_per_task,
            "min_trainable_units": min_trainable_units,
        },
        "artifacts": {"train_pairs_file": "train-pairs.jsonl"},
        "tasks": task_manifests,
        "totals": {
            "tasks": len(task_manifests),
            "train_support_documents": len(all_supports),
            "heldout_documents": len(all_heldout),
            "train_queries": len(all_queries),
            "train_pairs": len(pairs),
        },
    }
    validate_training_split_manifest(manifest)
    return manifest, pairs


def validate_training_split_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "split_id",
        "build_id",
        "guide_schema_version",
        "pair_schema_version",
        "pairing_strategy",
        "binding_scope",
        "seed",
        "task_scope",
        "config",
        "artifacts",
        "tasks",
        "totals",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("training split manifest has unexpected top-level fields")
    if manifest["schema_version"] != TRAINING_SPLIT_SCHEMA_VERSION:
        raise ValueError("unexpected training split schema")
    if manifest["guide_schema_version"] != BEHAVIOR_DOCUMENT_SCHEMA_VERSION:
        raise ValueError("training split Guide schema mismatch")
    if manifest["pair_schema_version"] != PAIR_SCHEMA_VERSION:
        raise ValueError("training split pair schema mismatch")
    if manifest["pairing_strategy"] != "balanced-static-assignment":
        raise ValueError("unsupported pairing strategy")
    if manifest["binding_scope"] != "query_episode":
        raise ValueError("training split bindings must be episode-scoped")
    if manifest["task_scope"] not in {"all", "partial"}:
        raise ValueError("training split task_scope must be all or partial")
    expected_config = {
        "support_documents_per_task",
        "heldout_documents_per_task",
        "query_episodes_per_task",
        "min_trainable_units",
    }
    if not isinstance(manifest["config"], Mapping) or set(manifest["config"]) != expected_config:
        raise ValueError("training split config has unexpected fields")
    if not isinstance(manifest["artifacts"], Mapping) or dict(manifest["artifacts"]) != {
        "train_pairs_file": "train-pairs.jsonl"
    }:
        raise ValueError("training split must reference train-pairs.jsonl")
    for field in ("split_id", "build_id"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ValueError(f"training split {field} must be a non-empty string")

    tasks = manifest["tasks"]
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence) or not tasks:
        raise ValueError("training split tasks must be a non-empty sequence")
    task_indices: set[int] = set()
    support_episodes: set[int] = set()
    heldout_episodes: set[int] = set()
    query_episodes: set[int] = set()
    pair_ids: set[str] = set()
    unused_episodes: set[int] = set()
    for task in tasks:
        expected_task_fields = {
            "task_index",
            "task_instruction",
            "train_support_documents",
            "heldout_documents",
            "train_queries",
            "unused_episode_indices",
            "assignment_summary",
        }
        if not isinstance(task, Mapping) or set(task) != expected_task_fields:
            raise ValueError("training split task entry has unexpected fields")
        task_index = _require_nonnegative_int(task["task_index"], name="task_index")
        if task_index in task_indices:
            raise ValueError(f"duplicate task_index={task_index}")
        task_indices.add(task_index)
        supports = task["train_support_documents"]
        heldout = task["heldout_documents"]
        queries = task["train_queries"]
        if not supports or not heldout or not queries:
            raise ValueError(f"task_index={task_index} has an empty split role")
        support_fields = {
            "episode_index",
            "document_id",
            "trainable_units",
            "assigned_query_episodes",
            "assigned_query_frames",
        }
        heldout_fields = {"episode_index", "document_id", "trainable_units"}
        query_fields = {
            "episode_index",
            "length",
            "pair_id",
            "support_episode_index",
            "support_document_id",
        }
        if any(not isinstance(entry, Mapping) or set(entry) != support_fields for entry in supports):
            raise ValueError("training support entry has unexpected fields")
        if any(not isinstance(entry, Mapping) or set(entry) != heldout_fields for entry in heldout):
            raise ValueError("held-out document entry has unexpected fields")
        if any(not isinstance(entry, Mapping) or set(entry) != query_fields for entry in queries):
            raise ValueError("training query entry has unexpected fields")
        task_supports = {
            _require_nonnegative_int(entry["episode_index"], name="support episode_index")
            for entry in supports
        }
        task_heldout = {
            _require_nonnegative_int(entry["episode_index"], name="heldout episode_index")
            for entry in heldout
        }
        task_queries = {
            _require_nonnegative_int(entry["episode_index"], name="query episode_index")
            for entry in queries
        }
        task_unused = {
            _require_nonnegative_int(value, name="unused episode_index")
            for value in task["unused_episode_indices"]
        }
        if (
            task_supports & task_heldout
            or task_supports & task_queries
            or task_heldout & task_queries
            or task_unused & (task_supports | task_heldout | task_queries)
        ):
            raise ValueError(f"task_index={task_index} split roles overlap")
        if (
            task_supports & (support_episodes | heldout_episodes | query_episodes | unused_episodes)
            or task_heldout & (support_episodes | heldout_episodes | query_episodes | unused_episodes)
            or task_queries & (support_episodes | heldout_episodes | query_episodes | unused_episodes)
            or task_unused & (support_episodes | heldout_episodes | query_episodes | unused_episodes)
        ):
            raise ValueError("episode index appears in multiple task manifests")
        support_documents = {
            entry["episode_index"]: entry["document_id"] for entry in supports
        }
        observed_query_counts: Counter[int] = Counter()
        observed_frame_counts: Counter[int] = Counter()
        for query in queries:
            support_episode = _require_nonnegative_int(
                query["support_episode_index"], name="query support_episode_index"
            )
            if support_episode not in task_supports:
                raise ValueError("query assignment references a non-support episode")
            if query["support_document_id"] != support_documents[support_episode]:
                raise ValueError("query assignment support episode/document mismatch")
            query_length = _require_nonnegative_int(query["length"], name="query length")
            if query_length < 1:
                raise ValueError("query length must be positive")
            observed_query_counts[support_episode] += 1
            observed_frame_counts[support_episode] += query_length
            pair_id = query["pair_id"]
            if pair_id in pair_ids:
                raise ValueError(f"duplicate pair_id={pair_id!r}")
            pair_ids.add(pair_id)
        for support in supports:
            support_episode = support["episode_index"]
            if support["assigned_query_episodes"] != observed_query_counts[support_episode]:
                raise ValueError("support assigned_query_episodes does not match queries")
            if support["assigned_query_frames"] != observed_frame_counts[support_episode]:
                raise ValueError("support assigned_query_frames does not match queries")
        query_loads = [observed_query_counts[episode] for episode in task_supports]
        expected_summary = {
            "min_queries_per_support": min(query_loads),
            "max_queries_per_support": max(query_loads),
            "query_count_gap": max(query_loads) - min(query_loads),
            "min_query_frames_per_support": min(observed_frame_counts.values()),
            "max_query_frames_per_support": max(observed_frame_counts.values()),
            "query_frame_gap": max(observed_frame_counts.values())
            - min(observed_frame_counts.values()),
        }
        if dict(task["assignment_summary"]) != expected_summary:
            raise ValueError("task assignment_summary does not match query assignments")
        if expected_summary["query_count_gap"] > 1:
            raise ValueError("support document assignment is not balanced")
        support_episodes.update(task_supports)
        heldout_episodes.update(task_heldout)
        query_episodes.update(task_queries)
        unused_episodes.update(task_unused)

    totals = manifest["totals"]
    expected_totals = {
        "tasks": len(task_indices),
        "train_support_documents": len(support_episodes),
        "heldout_documents": len(heldout_episodes),
        "train_queries": len(query_episodes),
        "train_pairs": len(pair_ids),
    }
    if dict(totals) != expected_totals:
        raise ValueError(
            f"training split totals mismatch: expected {expected_totals}, got {dict(totals)}"
        )


def load_training_split_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training split manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid training split JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError("training split manifest must contain one JSON object")
    validate_training_split_manifest(value)
    return value
