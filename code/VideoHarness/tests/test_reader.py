from __future__ import annotations

import copy
import dataclasses
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from _support import annotate_boundaries, set_document_quality

from video_harness.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    compose_evidence_record,
    mock_evidence_record,
)
from video_harness.reader import (
    GuideArtifactBundle,
    GuidePlan,
    build_guide_plan,
    load_guide_artifact_bundle,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import BEHAVIOR_DOCUMENT_SCHEMA_VERSION, plan_document

_BUILD_ID = "build-m4a-test"
_TASK_INDEX = 3
_TASK_INSTRUCTION = "Put bread into the toaster."


def _changed_evidence() -> dict[str, Any]:
    views = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    return compose_evidence_record(
        {
            "motion_summary": "The gripper transports the bread slice and releases it into the toaster.",
            "before_boundary_observation": {
                view: "The bread slice is outside the toaster." for view in views
            },
            "after_boundary_observation": {
                view: "The bread slice is inside the toaster." for view in views
            },
            "boundary_conflicts": {"before": None, "after": None},
            "detail_observation": "The detail sheet supports the insertion motion.",
            "unit_interpretation": {
                "action_description": "The robot inserts the bread slice into the toaster.",
                "task_role": "This Evidence Unit loads one bread slice for the toasting task.",
            },
            "causal_validation": {
                "status": "pass",
                "reason": "The Boundary states and motion summary support the insertion.",
            },
        },
        quality_status="accepted",
    )


def _gripper_close_evidence() -> dict[str, Any]:
    record = _changed_evidence()
    record["motion_summary"] = "The gripper closes around the stationary bread slice."
    record["unit_interpretation"] = {
        "action_description": "The robot closes the gripper around the bread slice.",
        "task_role": "This Evidence Unit establishes a grasp before transport.",
    }
    return record


def _document(
    episode_index: int,
    *,
    build_id: str = _BUILD_ID,
    records_by_order: dict[int, dict[str, Any]] | None = None,
    statuses_by_order: dict[int, str] | None = None,
) -> dict[str, Any]:
    source = EpisodeRecord(
        episode_index=episode_index,
        task_index=_TASK_INDEX,
        task_instruction=_TASK_INSTRUCTION,
        task_kind="benchmark",
        length=101,
        dataset_from_index=episode_index * 101,
        dataset_to_index=(episode_index + 1) * 101,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=key,
                path=f"videos/{key}/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=101 / 25,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ),
    )
    document = plan_document(source, build_id=build_id)
    records_by_order = records_by_order or {}
    statuses_by_order = statuses_by_order or {}

    for order, unit in enumerate(document["evidence_units"]):
        status = statuses_by_order.get(order)
        record = records_by_order.get(order)
        if status is None:
            status = "complete" if record is not None else "pending"

        if status in {"pending", "failed"}:
            record = None
            provenance = None
        else:
            record = copy.deepcopy(record or _changed_evidence())
            provenance = {
                "call1": {
                    "provider": "test-provider",
                    "model": "test-motion",
                    "prompt_version": "test-inspection",
                },
                "call2": {
                    "provider": "test-provider",
                    "model": "test-evidence",
                    "prompt_version": "test-evidence",
                },
                "repair": None,
            }

        unit["annotation"] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": status,
            "record": record,
            "provenance": provenance,
        }

    unit_statuses = {
        unit["annotation"]["status"] for unit in document["evidence_units"]
    }
    if unit_statuses == {"pending"}:
        annotate_boundaries(document, status="pending")
    elif unit_statuses == {"mock"}:
        annotate_boundaries(document, status="mock")
    else:
        annotate_boundaries(document, status="complete")
    statuses = unit_statuses | {
        boundary["annotation"]["status"] for boundary in document["boundary_states"]
    }
    document["status"] = (
        "planned"
        if statuses == {"pending"}
        else "annotated"
        if statuses == {"complete"}
        else "mock-annotated"
        if statuses == {"mock"}
        else "partially-annotated"
    )
    set_document_quality(
        document,
        "accepted"
        if document["status"] == "annotated"
        else "quarantined"
        if document["status"] == "mock-annotated"
        else "pending",
    )
    return document


def _pair(
    query_episode_index: int = 1,
    support_episode_index: int = 10,
    *,
    build_id: str = _BUILD_ID,
    pair_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "video-harness.support-query-pair",
        "build_id": build_id,
        "pair_id": pair_id or f"pair-q{query_episode_index}-s{support_episode_index}",
        "task_index": _TASK_INDEX,
        "task_instruction": _TASK_INSTRUCTION,
        "query_episode_index": query_episode_index,
        "support_episode_index": support_episode_index,
        "support_rank": 0,
        "support_document_id": f"robodojo/episode-{support_episode_index:07d}",
        "guide_schema_version": BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
    }


def _dataset(
    *, build_id: str = _BUILD_ID, supports_per_query: int = 1
) -> dict[str, Any]:
    return {
        "schema_version": "video-harness.robodojo-source",
        "task_scope": "benchmark-34",
        "episodes": 2,
        "tasks": 1,
        "frames": 202,
        "fps": 25,
        "episode_counts_by_task_index": {str(_TASK_INDEX): 2},
        "build_id": build_id,
        "source_dataset": "RoboDojo-Benchmark/RoboDojo/data/RoboDojo_lerobot_v30_video",
        "source_revision": "main",
        "sample_hz": 1,
        "supports_per_query": supports_per_query,
        "document_camera": "observation.images.cam_high",
        "benchmark_source_episodes": 2,
        "selection": {"max_tasks": 1, "episodes_per_task": 2},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_bundle(
    tmp_path: Path,
    *,
    documents: list[dict[str, Any]] | None = None,
    pairs: list[dict[str, Any]] | None = None,
    dataset: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    dataset_path = tmp_path / "dataset.json"
    documents_path = tmp_path / "documents.openai.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"

    _write_json(dataset_path, dataset or _dataset())
    _write_jsonl(
        documents_path,
        documents or [_document(1), _document(10)],
    )
    _write_jsonl(pairs_path, pairs or [_pair()])
    return dataset_path, documents_path, pairs_path


def test_load_artifact_bundle_is_typed_and_read_only(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)

    bundle = load_guide_artifact_bundle(
        dataset_path=paths[0],
        documents_path=paths[1],
        pairs_path=paths[2],
    )

    assert isinstance(bundle, GuideArtifactBundle)
    assert bundle.build_id == _BUILD_ID
    assert bundle.supports_per_query == 1
    assert {source.document_id for source in bundle.documents} == {
        "robodojo/episode-0000001",
        "robodojo/episode-0000010",
    }
    assert len(bundle.support_bindings) == 1
    binding = bundle.support_bindings[0]
    assert binding.query_episode_index == 1
    assert binding.support_episode_index == 10
    assert binding.support_document_id == "robodojo/episode-0000010"

    with pytest.raises(FrozenInstanceError):
        bundle.build_id = "other-build"


def test_loader_rejects_missing_artifact_file(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)
    paths[1].unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(str(paths[1]))):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_reports_malformed_jsonl_with_path_and_line(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)
    paths[1].write_text(
        json.dumps(_document(10)) + "\n" + "{malformed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{re.escape(str(paths[1]))}.*line 2"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_duplicate_document_id(tmp_path: Path) -> None:
    document = _document(10)
    paths = _write_bundle(tmp_path, documents=[document, copy.deepcopy(document)])

    with pytest.raises(ValueError, match="duplicate document_id"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_duplicate_pair_id(tmp_path: Path) -> None:
    pair = _pair()
    paths = _write_bundle(tmp_path, pairs=[pair, copy.deepcopy(pair)])

    with pytest.raises(ValueError, match="duplicate pair_id"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


@pytest.mark.parametrize("mismatched_artifact", ["dataset", "document", "pair"])
def test_loader_rejects_build_id_mismatch(
    tmp_path: Path, mismatched_artifact: str
) -> None:
    documents = [_document(1), _document(10)]
    dataset = _dataset()
    pairs = [_pair()]
    if mismatched_artifact == "dataset":
        dataset["build_id"] = "other-build"
    elif mismatched_artifact == "document":
        documents[1]["build_id"] = "other-build"
    else:
        pairs[0]["build_id"] = "other-build"
    paths = _write_bundle(tmp_path, documents=documents, pairs=pairs, dataset=dataset)

    with pytest.raises(ValueError, match="build_id"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_pair_schema_mismatch_and_unknown_keys(tmp_path: Path) -> None:
    pair = _pair()
    pair["guide_schema_version"] = "wrong-guide-schema"
    paths = _write_bundle(tmp_path, pairs=[pair])

    with pytest.raises(ValueError, match="guide_schema_version"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )

    pair = _pair()
    pair["unexpected"] = True
    paths = _write_bundle(tmp_path, pairs=[pair])
    with pytest.raises(ValueError, match="exactly|unknown|schema"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_missing_support_document(tmp_path: Path) -> None:
    pair = _pair()
    pair["support_document_id"] = "robodojo/episode-9999999"
    paths = _write_bundle(tmp_path, pairs=[pair])

    with pytest.raises(ValueError, match="support document"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("support_episode_index", 11, "support episode"),
        ("task_index", 4, "task"),
        ("task_instruction", "different task", "task"),
    ],
)
def test_loader_rejects_support_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    pair = _pair()
    pair[field] = value
    paths = _write_bundle(tmp_path, pairs=[pair])

    with pytest.raises(ValueError, match=message):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_query_support_collision(tmp_path: Path) -> None:
    pair = _pair(query_episode_index=10, support_episode_index=10)
    paths = _write_bundle(tmp_path, pairs=[pair])

    with pytest.raises(ValueError, match="different|support=query|query"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_multiple_supports_for_one_query(tmp_path: Path) -> None:
    documents = [_document(1), _document(10), _document(11)]
    pairs = [_pair(support_episode_index=10), _pair(support_episode_index=11)]
    paths = _write_bundle(tmp_path, documents=documents, pairs=pairs)

    with pytest.raises(ValueError, match="one support|multiple|query"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def test_loader_rejects_non_single_support_configuration(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path, dataset=_dataset(supports_per_query=2))

    with pytest.raises(ValueError, match="supports_per_query"):
        load_guide_artifact_bundle(
            dataset_path=paths[0],
            documents_path=paths[1],
            pairs_path=paths[2],
        )


def _bundle_for_plan(
    tmp_path: Path,
    *,
    support_document: dict[str, Any],
    documents: list[dict[str, Any]] | None = None,
) -> tuple[GuideArtifactBundle, dict[str, Any]]:
    query_document = _document(1)
    documents = documents or [query_document, support_document]
    paths = _write_bundle(tmp_path, documents=documents)
    bundle = load_guide_artifact_bundle(
        dataset_path=paths[0],
        documents_path=paths[1],
        pairs_path=paths[2],
    )
    return bundle, support_document


def test_build_plan_selects_trainable_units_and_preserves_identity(
    tmp_path: Path,
) -> None:
    support_document = _document(
        10,
        records_by_order={
            0: _changed_evidence(),
            1: _gripper_close_evidence(),
            2: _changed_evidence(),
            3: _changed_evidence(),
        },
    )
    original = copy.deepcopy(support_document)
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    plan = build_guide_plan(
        bundle,
        query_episode_index=1,
        profile="actuator",
    )

    assert isinstance(plan, GuidePlan)
    assert plan.query_episode_index == 1
    assert plan.support_document_id == "robodojo/episode-0000010"
    assert [unit.unit_id for unit in plan.units] == [
        "u0000",
        "u0001",
        "u0002",
        "u0003",
    ]
    assert [unit.order for unit in plan.units] == [0, 1, 2, 3]
    assert [frame.episode_frame_index for frame in plan.frames] == [0, 25, 50, 75, 100]
    assert [(unit.before_slot, unit.after_slot) for unit in plan.units] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(unit.transition_text.startswith("Motion:") for unit in plan.units)
    assert all(
        unit.provenance["call1"]["provider"] == "test-provider" for unit in plan.units
    )
    assert support_document == original


def test_build_plan_reuses_shared_slots_for_adjacent_selected_units(
    tmp_path: Path,
) -> None:
    support_document = _document(
        10,
        records_by_order={order: _changed_evidence() for order in range(4)},
    )
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    plan = build_guide_plan(bundle, query_episode_index=1)

    assert [(unit.before_slot, unit.after_slot) for unit in plan.units] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert plan.units[0].after_slot == plan.units[1].before_slot


def test_build_plan_rejects_skipped_units(tmp_path: Path) -> None:
    support_document = _document(
        10,
        records_by_order={0: _changed_evidence(), 2: _changed_evidence()},
    )
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    with pytest.raises(ValueError, match="quality-accepted"):
        build_guide_plan(bundle, query_episode_index=1)


def test_build_plan_rejects_pending_failed_or_mock_units(tmp_path: Path) -> None:
    support_document = _document(
        10,
        records_by_order={
            0: _changed_evidence(),
            1: _gripper_close_evidence(),
            2: mock_evidence_record(),
        },
        statuses_by_order={2: "mock", 3: "pending"},
    )
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    with pytest.raises(ValueError, match="quality-accepted"):
        build_guide_plan(bundle, query_episode_index=1)


@pytest.mark.parametrize("quarantine_target", ["unit", "boundary", "document"])
def test_build_plan_fails_closed_instead_of_building_partial_guide(
    tmp_path: Path, quarantine_target: str
) -> None:
    support_document = _document(
        10,
        records_by_order={order: _changed_evidence() for order in range(4)},
    )
    if quarantine_target == "document":
        set_document_quality(support_document, "quarantined")
    if quarantine_target == "unit":
        support_document["evidence_units"][1]["annotation"]["record"][
            "quality_status"
        ] = "quarantined"
        support_document["evidence_units"][1]["annotation"]["record"][
            "causal_validation"
        ] = {
            "status": "retry",
            "reason": "Automatic repair was unresolved.",
        }
        set_document_quality(support_document, "quarantined")
    elif quarantine_target == "boundary":
        support_document["boundary_states"][2]["annotation"]["record"][
            "quality_status"
        ] = "quarantined"
        set_document_quality(support_document, "quarantined")
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    with pytest.raises(ValueError, match="episode-0000010"):
        build_guide_plan(bundle, query_episode_index=1)


def test_build_plan_fails_closed_when_no_unit_is_trainable(tmp_path: Path) -> None:
    support_document = _document(
        10,
        records_by_order={0: mock_evidence_record()},
    )
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    with pytest.raises(ValueError, match="robodojo/episode-0000010"):
        build_guide_plan(bundle, query_episode_index=1)


def test_guide_plan_is_token_neutral_and_has_no_image_payload(tmp_path: Path) -> None:
    support_document = _document(
        10,
        records_by_order={order: _changed_evidence() for order in range(4)},
    )
    bundle, _ = _bundle_for_plan(tmp_path, support_document=support_document)

    plan = build_guide_plan(bundle, query_episode_index=1)

    field_names = {field.name for field in dataclasses.fields(plan)}
    assert "images" not in field_names
    assert "tokens" not in field_names
    assert "observation" not in field_names
    assert "actions" not in field_names
    assert all(isinstance(frame.episode_frame_index, int) for frame in plan.frames)
    assert all(isinstance(unit.transition_text, str) for unit in plan.units)
    assert all(not isinstance(unit.transition_text, bytes) for unit in plan.units)
