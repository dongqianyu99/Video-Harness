import copy

import pytest

from video_harness.robodojo import SourceContractError, validate_info


def _info() -> dict:
    cameras = {
        key: {
            "dtype": "video",
            "shape": [3, 480, 640],
            "info": {"video.fps": 25, "video.is_depth_map": False},
        }
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    }
    return {
        "codebase_version": "v3.0",
        "fps": 25,
        "total_episodes": 3500,
        "total_frames": 1_859_602,
        "total_tasks": 35,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "episode_index": {},
            "frame_index": {},
            "task_index": {},
            "timestamp": {},
            **cameras,
        },
    }


def test_public_joint_v3_contract_is_accepted() -> None:
    validate_info(_info())


def test_ee_variant_is_rejected() -> None:
    info = copy.deepcopy(_info())
    info["features"]["action"]["shape"] = [16]
    with pytest.raises(SourceContractError, match=r"float32\[14\]"):
        validate_info(info)
