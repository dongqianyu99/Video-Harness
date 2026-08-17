from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.sampling import boundary_frames, media_timestamp, plan_document


def _record(length: int = 579) -> EpisodeRecord:
    videos = tuple(
        VideoSlice(key=key, path=f"videos/{key}/file-000.mp4", from_timestamp=23.16, to_timestamp=46.32)
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
    units = document["guidance_units"]
    assert units[0]["after"] == units[1]["before"]
    assert units[0]["annotation"] == {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "pending",
        "record": None,
        "provenance": None,
    }
    assert document["sampling"]["kind"] == "uniform_guidance_unit"
    assert document["source"]["episode_length"] == 579
    assert "before_media" not in units[0]
    assert media_timestamp(document, 25) == 24.16
