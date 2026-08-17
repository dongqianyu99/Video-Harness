from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .robodojo import EpisodeRecord
from .sampling import BEHAVIOR_DOCUMENT_SCHEMA_VERSION


def _rank(seed: int, query_episode: int, support_episode: int) -> bytes:
    payload = f"{seed}:{query_episode}:{support_episode}".encode()
    return hashlib.sha256(payload).digest()


def build_pairs(
    records: Iterable[EpisodeRecord],
    *,
    build_id: str,
    supports_per_query: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if not build_id.strip():
        raise ValueError("build_id must be non-empty")
    if supports_per_query < 1:
        raise ValueError("supports_per_query must be at least one")
    by_task: dict[int, list[EpisodeRecord]] = {}
    for record in records:
        by_task.setdefault(record.task_index, []).append(record)

    pairs: list[dict[str, Any]] = []
    for task_index, task_records in sorted(by_task.items()):
        task_records = sorted(task_records, key=lambda record: record.episode_index)
        if supports_per_query >= len(task_records):
            raise ValueError(
                f"Task {task_index} has {len(task_records)} episodes, so it cannot provide "
                f"{supports_per_query} distinct supports per query"
            )
        for query in task_records:
            candidates = [record for record in task_records if record.episode_index != query.episode_index]
            candidates.sort(key=lambda support: _rank(seed, query.episode_index, support.episode_index))
            for support_rank, support in enumerate(candidates[:supports_per_query]):
                pairs.append(
                    {
                        "schema_version": "video-harness.support-query-pair.v0.1",
                        "build_id": build_id,
                        "pair_id": (
                            f"t{task_index:02d}-q{query.episode_index:07d}-"
                            f"s{support.episode_index:07d}"
                        ),
                        "task_index": task_index,
                        "task_instruction": query.task_instruction,
                        "query_episode_index": query.episode_index,
                        "support_episode_index": support.episode_index,
                        "support_rank": support_rank,
                        "support_document_id": f"robodojo/episode-{support.episode_index:07d}",
                        "guide_schema_version": BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
                    }
                )
    return pairs
