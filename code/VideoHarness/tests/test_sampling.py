import copy

import pytest

from video_harness.evidence import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import (
    BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
    boundary_frames,
    media_timestamp,
    plan_document,
    validate_document,
)


def _record(length: int = 579) -> EpisodeRecord:
    videos = tuple(
        VideoSlice(
            key=key,
            path=f"videos/{key}/file-000.mp4",
            from_timestamp=23.16,
            to_timestamp=46.32,
        )
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    )
    return EpisodeRecord(
        episode_index=7,
        task_index=3,
        task_instruction="Do the task.",
        task_kind="benchmark",
        length=length,
        dataset_from_index=100,
        dataset_to_index=100 + length,
        data_path="data/chunk-000/file-000.parquet",
        videos=videos,
    )


def test_boundary_frames_include_first_and_last_without_duplicates() -> None:
    frames = boundary_frames(length=579, fps=25, sample_hz=1)
    assert frames[0] == 0
    assert frames[-1] == 578
    assert len(frames) == len(set(frames))
    assert frames[:4] == [0, 25, 50, 75]


def test_document_interleaves_shared_boundaries() -> None:
    document = plan_document(_record(), build_id="test-build")
    assert BEHAVIOR_DOCUMENT_SCHEMA_VERSION == "video-harness.behavior-document.v2"
    units = document["evidence_units"]
    boundaries = document["boundary_states"]
    assert units[0]["after_boundary_id"] == units[1]["before_boundary_id"]
    assert len(boundaries) == len(units) + 1
    assert (
        boundaries[0]["annotation"]["schema_version"] == BOUNDARY_STATE_SCHEMA_VERSION
    )
    assert units[0]["annotation"] == {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "pending",
        "record": None,
        "provenance": None,
    }
    assert document["sampling"]["kind"] == "uniform_evidence_unit"
    assert document["source"]["episode_length"] == 579
    assert "before" not in units[0] and "after" not in units[0]
    assert media_timestamp(document, 25) == 24.16


def test_legacy_document_and_boundary_schemas_are_rejected() -> None:
    document = plan_document(_record(length=51), build_id="test-build")
    document["schema_version"] = "video-harness.behavior-document"
    with pytest.raises(ValueError, match="unexpected behavior document schema"):
        validate_document(document)

    document = plan_document(_record(length=51), build_id="test-build")
    document["boundary_states"][0]["annotation"]["schema_version"] = (
        "video-harness.boundary-state"
    )
    with pytest.raises(ValueError, match="unexpected schema"):
        validate_document(document)


def test_document_rejects_missing_or_dangling_boundary_state() -> None:
    document = plan_document(_record(length=51), build_id="test-build")
    missing = copy.deepcopy(document)
    missing["boundary_states"].pop(1)
    with pytest.raises(ValueError, match="one more Boundary State"):
        validate_document(missing)

    dangling = copy.deepcopy(document)
    dangling["evidence_units"][0]["after_boundary_id"] = "b9999"
    with pytest.raises(ValueError, match="adjacent Boundary States"):
        validate_document(dangling)
