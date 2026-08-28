from pathlib import Path

import pyarrow as pa
from pyarrow import parquet

from video_harness.gripper_state import GripperStateReader, keyframe_indices
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document


def _document(dataset_root: Path) -> dict:
    path = dataset_root / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    states = []
    for frame in range(26):
        left = frame / 25
        right = 1 - frame / 25
        states.append([0.0] * 6 + [left] + [0.0] * 6 + [right])
    parquet.write_table(
        pa.table(
            {
                "episode_index": [7] * 26,
                "frame_index": list(range(26)),
                "observation.state": states,
            }
        ),
        path,
    )
    record = EpisodeRecord(
        episode_index=7,
        task_index=1,
        task_instruction="Move the object.",
        task_kind="benchmark",
        length=26,
        dataset_from_index=100,
        dataset_to_index=126,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=f"observation.images.{view}",
                path=f"videos/{view}.mp4",
                from_timestamp=0.0,
                to_timestamp=26 / 25,
            )
            for view in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ),
    )
    return plan_document(record, build_id="state-test")


def test_lerobot_gripper_state_matches_keyframe_sheet_indices(tmp_path: Path) -> None:
    document = _document(tmp_path)
    state = GripperStateReader(tmp_path).read_unit(
        document,
        episode_start_frame=0,
        episode_end_frame=25,
    )

    assert state.unit_frame_indices == (0, 5, 10, 15, 20, 25)
    assert state.left[0] == 0.0
    assert state.left[-1] == 1.0
    assert state.right[0] == 1.0
    assert state.right[-1] == 0.0
    assert "Left gripper" in state.prompt_text()


def test_short_unit_uses_same_evenly_spaced_indices_as_visual_sheet() -> None:
    assert keyframe_indices(4) == (0, 1, 2, 3)
