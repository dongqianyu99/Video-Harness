from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from openpi.training.guide_split import load_and_validate_training_split


@dataclass(frozen=True)
class _Binding:
    pair_id: str
    query_episode_index: int
    support_episode_index: int
    support_document_id: str
    task_index: int


def _manifest() -> dict:
    tasks = []
    bindings = []
    documents = []
    episodes = []
    for task_index in range(2):
        base = task_index * 10
        support = base
        heldout = base + 1
        queries = (base + 2, base + 3)
        documents.extend(
            [
                SimpleNamespace(
                    document_id=f"doc-{support}",
                    episode_index=support,
                    task_index=task_index,
                ),
                SimpleNamespace(
                    document_id=f"doc-{heldout}",
                    episode_index=heldout,
                    task_index=task_index,
                ),
            ]
        )
        episodes.extend(
            SimpleNamespace(episode_index=index, task_index=task_index, length=10 + index)
            for index in (support, heldout, *queries)
        )
        query_entries = []
        for query in queries:
            pair_id = f"pair-{query}"
            query_entries.append(
                {
                    "episode_index": query,
                    "length": 10 + query,
                    "pair_id": pair_id,
                    "support_episode_index": support,
                    "support_document_id": f"doc-{support}",
                }
            )
            bindings.append(
                _Binding(pair_id, query, support, f"doc-{support}", task_index)
            )
        tasks.append(
            {
                "task_index": task_index,
                "task_instruction": f"task-{task_index}",
                "train_support_documents": [
                    {
                        "episode_index": support,
                        "document_id": f"doc-{support}",
                        "trainable_units": 1,
                        "assigned_query_episodes": 2,
                        "assigned_query_frames": sum(10 + query for query in queries),
                    }
                ],
                "heldout_documents": [
                    {
                        "episode_index": heldout,
                        "document_id": f"doc-{heldout}",
                        "trainable_units": 1,
                    }
                ],
                "train_queries": query_entries,
                "unused_episode_indices": [],
                "assignment_summary": {
                    "min_queries_per_support": 2,
                    "max_queries_per_support": 2,
                    "query_count_gap": 0,
                    "min_query_frames_per_support": sum(10 + query for query in queries),
                    "max_query_frames_per_support": sum(10 + query for query in queries),
                    "query_frame_gap": 0,
                },
            }
        )
    manifest = {
        "schema_version": "video-harness.training-split.v0",
        "split_id": "test-split",
        "build_id": "test-build",
        "guide_schema_version": "video-harness.behavior-document.v0.2",
        "pair_schema_version": "video-harness.support-query-pair.v0.1",
        "pairing_strategy": "balanced-static-assignment-v1",
        "binding_scope": "query_episode",
        "seed": 0,
        "task_scope": "partial",
        "config": {
            "support_documents_per_task": 1,
            "heldout_documents_per_task": 1,
            "query_episodes_per_task": None,
            "min_trainable_units": 1,
        },
        "artifacts": {"train_pairs_file": "train-pairs.jsonl"},
        "tasks": tasks,
        "totals": {
            "tasks": 2,
            "train_support_documents": 2,
            "heldout_documents": 2,
            "train_queries": 4,
            "train_pairs": 4,
        },
    }
    return manifest, bindings, documents, episodes


def _write(tmp_path, manifest):
    path = tmp_path / "training-split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_split_validator_matches_pairs_documents_and_episode_metadata(tmp_path):
    manifest, bindings, documents, episodes = _manifest()
    bundle = SimpleNamespace(
        build_id="test-build",
        support_bindings=tuple(bindings),
        documents=tuple(documents),
    )

    result = load_and_validate_training_split(
        _write(tmp_path, manifest),
        bundle=bundle,
        episode_records=episodes,
        require_all_tasks=False,
    )

    assert result.split_id == "test-split"
    assert result.query_episode_indices == (2, 3, 12, 13)
    assert result.support_episode_indices == (0, 10)
    assert result.heldout_episode_indices == (1, 11)
    assert result.heldout_document_ids == ("doc-1", "doc-11")


def test_split_validator_rejects_pair_drift_and_heldout_leakage(tmp_path):
    manifest, bindings, documents, episodes = _manifest()
    drifted = list(bindings)
    drifted[0] = _Binding(
        drifted[0].pair_id,
        drifted[0].query_episode_index,
        drifted[0].support_episode_index,
        "doc-1",
        drifted[0].task_index,
    )
    bundle = SimpleNamespace(
        build_id="test-build",
        support_bindings=tuple(drifted),
        documents=tuple(documents),
    )

    with pytest.raises(ValueError, match="field support_document_id mismatch"):
        load_and_validate_training_split(
            _write(tmp_path, manifest),
            bundle=bundle,
            episode_records=episodes,
            require_all_tasks=False,
        )


def test_formal_split_requires_all_34_dataset_tasks(tmp_path):
    manifest, bindings, documents, episodes = _manifest()
    bundle = SimpleNamespace(
        build_id="test-build",
        support_bindings=tuple(bindings),
        documents=tuple(documents),
    )

    with pytest.raises(ValueError, match="task_scope='all'"):
        load_and_validate_training_split(
            _write(tmp_path, manifest),
            bundle=bundle,
            episode_records=episodes,
            require_all_tasks=True,
        )


def test_split_validator_rejects_manifest_build_drift(tmp_path):
    manifest, bindings, documents, episodes = _manifest()
    invalid = copy.deepcopy(manifest)
    invalid["build_id"] = "other-build"
    bundle = SimpleNamespace(
        build_id="test-build",
        support_bindings=tuple(bindings),
        documents=tuple(documents),
    )

    with pytest.raises(ValueError, match="build_id"):
        load_and_validate_training_split(
            _write(tmp_path, invalid),
            bundle=bundle,
            episode_records=episodes,
            require_all_tasks=False,
        )
