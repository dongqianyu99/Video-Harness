from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidatedTrainingSplit:
    split_id: str
    query_episode_indices: tuple[int, ...]
    support_episode_indices: tuple[int, ...]
    heldout_episode_indices: tuple[int, ...]
    heldout_document_ids: tuple[str, ...]
    task_indices: tuple[int, ...]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training split manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid training split JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("training split manifest must contain one JSON object")
    expected = {
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
    if set(manifest) != expected:
        raise ValueError("training split manifest has unexpected top-level fields")
    if manifest["schema_version"] != "video-harness.training-split":
        raise ValueError("unexpected training split schema")
    if manifest["guide_schema_version"] != "video-harness.behavior-document":
        raise ValueError("training split Guide schema mismatch")
    if manifest["pair_schema_version"] != "video-harness.support-query-pair":
        raise ValueError("training split pair schema mismatch")
    if manifest["pairing_strategy"] != "balanced-static-assignment":
        raise ValueError("unsupported pairing strategy")
    if manifest["binding_scope"] != "query_episode":
        raise ValueError("training split bindings must be episode-scoped")
    return manifest


def _document_index(bundle: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for document in getattr(bundle, "documents", ()):
        if document.document_id in result:
            raise ValueError(f"duplicate loaded document_id={document.document_id!r}")
        result[document.document_id] = document
    return result


def load_and_validate_training_split(
    path: Path,
    *,
    bundle: Any,
    episode_records: list[Any],
    require_all_tasks: bool,
) -> ValidatedTrainingSplit:
    manifest = _load_manifest(path)
    if manifest["build_id"] != bundle.build_id:
        raise ValueError(
            f"training split build_id {manifest['build_id']!r} does not match "
            f"artifact build_id {bundle.build_id!r}"
        )

    records_by_episode = {int(record.episode_index): record for record in episode_records}
    dataset_tasks = {int(record.task_index) for record in episode_records}
    documents_by_id = _document_index(bundle)
    bindings_by_query = {}
    for binding in getattr(bundle, "support_bindings", ()):
        if binding.query_episode_index in bindings_by_query:
            raise ValueError(
                f"duplicate loaded binding for query episode {binding.query_episode_index}"
            )
        bindings_by_query[binding.query_episode_index] = binding

    query_episodes: set[int] = set()
    support_episodes: set[int] = set()
    heldout_episodes: set[int] = set()
    heldout_document_ids: set[str] = set()
    task_indices: set[int] = set()

    for task in manifest["tasks"]:
        task_index = int(task["task_index"])
        task_indices.add(task_index)
        support_by_episode = {
            int(entry["episode_index"]): entry
            for entry in task["train_support_documents"]
        }
        heldout_by_episode = {
            int(entry["episode_index"]): entry
            for entry in task["heldout_documents"]
        }

        for episode_index, entry in {**support_by_episode, **heldout_by_episode}.items():
            record = records_by_episode.get(episode_index)
            if record is None or int(record.task_index) != task_index:
                raise ValueError(
                    f"split document episode {episode_index} is absent or has the wrong task"
                )
            document = documents_by_id.get(entry["document_id"])
            if document is None:
                raise ValueError(
                    f"split document {entry['document_id']!r} is absent from the artifact"
                )
            if document.episode_index != episode_index or document.task_index != task_index:
                raise ValueError(
                    f"split document {entry['document_id']!r} source identity mismatch"
                )

        support_episodes.update(support_by_episode)
        heldout_episodes.update(heldout_by_episode)
        heldout_document_ids.update(
            entry["document_id"] for entry in heldout_by_episode.values()
        )

        observed_counts = dict.fromkeys(support_by_episode, 0)
        observed_frames = dict.fromkeys(support_by_episode, 0)
        for query in task["train_queries"]:
            query_episode = int(query["episode_index"])
            if query_episode in query_episodes:
                raise ValueError(f"duplicate split query episode {query_episode}")
            query_episodes.add(query_episode)
            record = records_by_episode.get(query_episode)
            if record is None or int(record.task_index) != task_index:
                raise ValueError(
                    f"split query episode {query_episode} is absent or has the wrong task"
                )
            if int(query["length"]) != int(record.length):
                raise ValueError(f"split query episode {query_episode} length mismatch")

            support_episode = int(query["support_episode_index"])
            if support_episode not in support_by_episode:
                raise ValueError(
                    f"query episode {query_episode} references a non-support episode"
                )
            binding = bindings_by_query.get(query_episode)
            if binding is None:
                raise ValueError(
                    f"train-pairs artifact has no binding for query episode {query_episode}"
                )
            expected = {
                "pair_id": query["pair_id"],
                "support_episode_index": support_episode,
                "support_document_id": query["support_document_id"],
                "task_index": task_index,
            }
            for field, expected_value in expected.items():
                if getattr(binding, field) != expected_value:
                    raise ValueError(
                        f"query episode {query_episode} pair field {field} mismatch"
                    )
            observed_counts[support_episode] += 1
            observed_frames[support_episode] += int(record.length)

        for support_episode, entry in support_by_episode.items():
            if int(entry["assigned_query_episodes"]) != observed_counts[support_episode]:
                raise ValueError(
                    f"support episode {support_episode} assigned query count mismatch"
                )
            if int(entry["assigned_query_frames"]) != observed_frames[support_episode]:
                raise ValueError(
                    f"support episode {support_episode} assigned frame count mismatch"
                )
        if observed_counts and max(observed_counts.values()) - min(observed_counts.values()) > 1:
            raise ValueError(f"task_index={task_index} support assignment is not balanced")

    if set(bindings_by_query) != query_episodes:
        extra = sorted(set(bindings_by_query) - query_episodes)
        missing = sorted(query_episodes - set(bindings_by_query))
        raise ValueError(
            f"train-pairs/split query set mismatch: extra={extra}, missing={missing}"
        )
    if query_episodes & support_episodes or query_episodes & heldout_episodes or support_episodes & heldout_episodes:
        raise ValueError("training split roles overlap after dataset validation")
    if heldout_document_ids & {
        binding.support_document_id for binding in bindings_by_query.values()
    }:
        raise ValueError("held-out documents appear in train-pairs")

    if require_all_tasks:
        if manifest["task_scope"] != "all":
            raise ValueError("formal training requires task_scope='all'")
        if task_indices != dataset_tasks or len(task_indices) != 34:
            raise ValueError(
                "formal training split must cover all 34 RoboDojo demonstration tasks"
            )

    return ValidatedTrainingSplit(
        split_id=manifest["split_id"],
        query_episode_indices=tuple(sorted(query_episodes)),
        support_episode_indices=tuple(sorted(support_episodes)),
        heldout_episode_indices=tuple(sorted(heldout_episodes)),
        heldout_document_ids=tuple(sorted(heldout_document_ids)),
        task_indices=tuple(sorted(task_indices)),
    )
