from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from video_harness.eval_guidance import (
    build_eval_guidance_catalog,
    load_eval_guidance_catalog,
)
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document


def _document(
    episode_index: int,
    *,
    task_index: int = 3,
    instruction: str = "Put bread into the toaster.",
    build_id: str = "eval-build",
    changed_evidence: dict,
) -> dict:
    record = EpisodeRecord(
        episode_index=episode_index,
        task_index=task_index,
        task_instruction=instruction,
        task_kind="benchmark",
        length=51,
        dataset_from_index=episode_index * 51,
        dataset_to_index=(episode_index + 1) * 51,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=key,
                path=f"videos/{key}/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=51 / 25,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ),
    )
    document = plan_document(record, build_id=build_id)
    document["status"] = "partially-annotated"
    document["guidance_units"][0]["annotation"] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "record": copy.deepcopy(changed_evidence),
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
                "attempts": 1,
                "selected_attempt": 1,
            },
        },
    }
    return document


def _episode(episode_index: int, task_index: int = 3, instruction: str = "Put bread into the toaster.") -> dict:
    return {
        "episode_index": episode_index,
        "task_index": task_index,
        "task_instruction": instruction,
    }


def test_catalog_selects_dataset_first_episode_and_reuses_it(changed_evidence):
    documents = [
        _document(12, changed_evidence=changed_evidence),
        _document(10, changed_evidence=changed_evidence),
    ]
    catalog = build_eval_guidance_catalog(
        documents,
        episodes=[_episode(12), _episode(10), _episode(11)],
    )

    by_index = catalog.resolve(task_index=3)
    by_instruction = catalog.resolve(task_instruction="Put bread into the toaster.")

    assert by_index is by_instruction
    assert by_index.source_episode_index == 10
    assert by_index.document_id == "robodojo/episode-0000010"
    assert catalog.resolve(task_index=3) is by_index


def test_plan_skips_pending_units_and_deduplicates_frames(changed_evidence):
    catalog = build_eval_guidance_catalog([_document(10, changed_evidence=changed_evidence)])

    plan = catalog.build_plan(task_index=3)

    assert plan.support_episode_index == 10
    assert [frame.episode_frame_index for frame in plan.frames] == [0, 25]
    assert len(plan.units) == 1
    assert plan.units[0].before_slot == 0
    assert plan.units[0].after_slot == 1
    assert "Action:" in plan.units[0].transition_text


def test_catalog_rejects_document_that_is_not_dataset_first(changed_evidence):
    document = _document(11, changed_evidence=changed_evidence)

    with pytest.raises(ValueError, match="dataset-first"):
        build_eval_guidance_catalog(
            [document],
            episodes=[_episode(10), _episode(11)],
        )


def test_catalog_rejects_duplicate_and_build_mismatch(changed_evidence):
    document = _document(10, changed_evidence=changed_evidence)
    with pytest.raises(ValueError, match="Duplicate"):
        build_eval_guidance_catalog([document, copy.deepcopy(document)])

    other_build = _document(11, build_id="other-build", changed_evidence=changed_evidence)
    with pytest.raises(ValueError, match="build_id"):
        build_eval_guidance_catalog([document, other_build])


def test_plan_requires_at_least_one_trainable_unit(changed_evidence):
    document = _document(10, changed_evidence=changed_evidence)
    document["guidance_units"][0]["annotation"] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "pending",
        "record": None,
        "provenance": None,
    }
    document["status"] = "planned"
    catalog = build_eval_guidance_catalog([document])

    with pytest.raises(ValueError, match="no trainable"):
        catalog.build_plan(task_index=3)


def test_loader_accepts_single_json(tmp_path: Path, changed_evidence):
    path = tmp_path / "annotated-document.json"
    path.write_text(json.dumps(_document(10, changed_evidence=changed_evidence)), encoding="utf-8")

    catalog = load_eval_guidance_catalog(path)

    assert catalog.resolve(task_index=3).source_episode_index == 10
