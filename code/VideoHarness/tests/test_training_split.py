from __future__ import annotations

import copy
import json

import pytest
from _support import annotate_boundaries, set_document_quality

from video_harness.cli import _make_training_split
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document
from video_harness.training_split import (
    TRAINING_SPLIT_SCHEMA_VERSION,
    build_training_split,
    load_training_split_manifest,
    validate_training_split_manifest,
)


def _record(episode: int, task: int, *, length: int | None = None) -> EpisodeRecord:
    frame_count = 26 + episode if length is None else length
    videos = tuple(
        VideoSlice(
            key=key,
            path=f"videos/{key}/file-000.mp4",
            from_timestamp=episode * 2.0,
            to_timestamp=episode * 2.0 + frame_count / 25,
        )
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    )
    return EpisodeRecord(
        episode_index=episode,
        task_index=task,
        task_instruction=f"task-{task}",
        task_kind="benchmark",
        length=frame_count,
        dataset_from_index=episode * 100,
        dataset_to_index=episode * 100 + frame_count,
        data_path="data/chunk-000/file-000.parquet",
        videos=videos,
    )


def _annotated_document(record: EpisodeRecord, evidence: dict) -> dict:
    document = plan_document(record, build_id="test-build", sample_hz=1)
    annotate_boundaries(document)
    for unit in document["evidence_units"]:
        unit["annotation"] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "complete",
            "record": copy.deepcopy(evidence),
            "provenance": {
                "call1": {
                    "provider": "test",
                    "model": "test-motion",
                    "prompt_version": "test-inspection",
                },
                "call2": {
                    "provider": "test",
                    "model": "test-evidence",
                    "prompt_version": "test-evidence",
                },
                "repair": None,
            },
        }
    document["status"] = "annotated"
    set_document_quality(document, "accepted")
    return document


def _corpus(changed_evidence):
    records = [
        _record(task * 100 + offset, task) for task in range(2) for offset in range(8)
    ]
    documents = [_annotated_document(record, changed_evidence) for record in records]
    return records, documents


def test_split_is_role_disjoint_balanced_and_deterministic(changed_evidence):
    records, documents = _corpus(changed_evidence)

    manifest, pairs = build_training_split(
        records,
        documents,
        build_id="test-build",
        support_documents_per_task=2,
        heldout_documents_per_task=1,
        seed=7,
    )
    reversed_manifest, reversed_pairs = build_training_split(
        reversed(records),
        reversed(documents),
        build_id="test-build",
        support_documents_per_task=2,
        heldout_documents_per_task=1,
        seed=7,
    )

    assert manifest == reversed_manifest
    assert pairs == reversed_pairs
    assert manifest["schema_version"] == TRAINING_SPLIT_SCHEMA_VERSION
    assert manifest["task_scope"] == "partial"
    assert manifest["totals"] == {
        "tasks": 2,
        "train_support_documents": 4,
        "heldout_documents": 2,
        "train_queries": 10,
        "train_pairs": 10,
    }
    assert len(pairs) == 10

    for task in manifest["tasks"]:
        support = {entry["episode_index"] for entry in task["train_support_documents"]}
        heldout = {entry["episode_index"] for entry in task["heldout_documents"]}
        queries = {entry["episode_index"] for entry in task["train_queries"]}
        assert not support & heldout
        assert not support & queries
        assert not heldout & queries
        assert task["assignment_summary"]["query_count_gap"] <= 1
        for query in task["train_queries"]:
            assert query["support_episode_index"] in support

    heldout_all = {
        entry["episode_index"]
        for task in manifest["tasks"]
        for entry in task["heldout_documents"]
    }
    assert not heldout_all & {pair["query_episode_index"] for pair in pairs}
    assert not heldout_all & {pair["support_episode_index"] for pair in pairs}


def test_split_uses_only_trainable_documents_for_support_and_heldout(changed_evidence):
    records, documents = _corpus(changed_evidence)
    for document in documents[:2]:
        for unit in document["evidence_units"]:
            unit["annotation"] = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "status": "pending",
                "record": None,
                "provenance": None,
            }
        document["status"] = "planned"
        annotate_boundaries(document, status="pending")
        set_document_quality(document, "pending")

    manifest, _ = build_training_split(
        records,
        documents,
        build_id="test-build",
        support_documents_per_task=2,
        heldout_documents_per_task=1,
        seed=2,
    )

    selected_documents = {
        entry["document_id"]
        for task in manifest["tasks"]
        for role in ("train_support_documents", "heldout_documents")
        for entry in task[role]
    }
    assert documents[0]["document_id"] not in selected_documents
    assert documents[1]["document_id"] not in selected_documents


def test_split_excludes_quarantined_document_as_a_whole(changed_evidence):
    records, documents = _corpus(changed_evidence)
    set_document_quality(documents[0], "quarantined")
    documents[0]["evidence_units"][0]["annotation"]["record"]["quality_status"] = (
        "quarantined"
    )
    documents[0]["evidence_units"][0]["annotation"]["record"]["causal_validation"] = {
        "status": "retry",
        "reason": "Automatic repair was unresolved.",
    }
    manifest, _ = build_training_split(
        records,
        documents,
        build_id="test-build",
        support_documents_per_task=2,
        heldout_documents_per_task=1,
        seed=2,
    )
    selected_documents = {
        entry["document_id"]
        for task in manifest["tasks"]
        for role in ("train_support_documents", "heldout_documents")
        for entry in task[role]
    }
    assert documents[0]["document_id"] not in selected_documents


def test_split_can_limit_queries_without_reusing_unused_episodes(changed_evidence):
    records, documents = _corpus(changed_evidence)
    manifest, pairs = build_training_split(
        records,
        documents,
        build_id="test-build",
        support_documents_per_task=2,
        heldout_documents_per_task=1,
        query_episodes_per_task=3,
        seed=5,
    )

    assert len(pairs) == 6
    assert all(len(task["train_queries"]) == 3 for task in manifest["tasks"])
    assert all(len(task["unused_episode_indices"]) == 2 for task in manifest["tasks"])


def test_split_manifest_loader_rejects_role_overlap(tmp_path, changed_evidence):
    records, documents = _corpus(changed_evidence)
    manifest, _ = build_training_split(
        records,
        documents,
        build_id="test-build",
        support_documents_per_task=2,
        heldout_documents_per_task=1,
    )
    invalid = copy.deepcopy(manifest)
    invalid["tasks"][0]["train_queries"][0]["episode_index"] = invalid["tasks"][0][
        "train_support_documents"
    ][0]["episode_index"]

    with pytest.raises(ValueError, match="overlap"):
        validate_training_split_manifest(invalid)

    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_training_split_manifest(path) == manifest


def test_split_marks_exact_34_task_corpus_as_formal_all_task_scope(changed_evidence):
    records = [
        _record(task * 100 + offset, task, length=26)
        for task in range(34)
        for offset in range(4)
    ]
    documents = [_annotated_document(record, changed_evidence) for record in records]

    manifest, pairs = build_training_split(
        records,
        documents,
        build_id="test-build",
        support_documents_per_task=1,
        heldout_documents_per_task=1,
        seed=11,
    )

    assert manifest["task_scope"] == "all"
    assert manifest["totals"] == {
        "tasks": 34,
        "train_support_documents": 34,
        "heldout_documents": 34,
        "train_queries": 68,
        "train_pairs": 68,
    }
    assert len(pairs) == 68


def test_training_split_cli_writes_manifest_and_derived_pairs(
    tmp_path, changed_evidence
):
    records, documents = _corpus(changed_evidence)
    dataset_path = tmp_path / "dataset.json"
    episodes_path = tmp_path / "episodes.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    output_root = tmp_path / "split"
    dataset_path.write_text(json.dumps({"build_id": "test-build"}), encoding="utf-8")
    episodes_path.write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    documents_path.write_text(
        "".join(json.dumps(document) + "\n" for document in documents),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "dataset_artifact": dataset_path,
            "episodes": episodes_path,
            "documents": documents_path,
            "output_root": output_root,
            "support_documents_per_task": 2,
            "heldout_documents_per_task": 1,
            "query_episodes_per_task": None,
            "min_trainable_units": 1,
            "seed": 3,
        },
    )()

    assert _make_training_split(args) == 0

    manifest = load_training_split_manifest(output_root / "training-split.json")
    pairs = [
        json.loads(line)
        for line in (output_root / "train-pairs.jsonl").read_text().splitlines()
    ]
    assert manifest["totals"]["train_pairs"] == len(pairs) == 10
