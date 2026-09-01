from __future__ import annotations

import os
from pathlib import Path


def _workspace_root() -> Path:
    override = os.getenv("VIDEO_HARNESS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "code" / "VideoHarness").is_dir() and (
            parent / "code" / "RoboDojo"
        ).is_dir():
            return parent
    raise RuntimeError("Set VIDEO_HARNESS_ROOT to the Video-Harness checkout")


WORKSPACE_ROOT = _workspace_root()
DATA_ROOT = Path(
    os.getenv("VIDEO_HARNESS_DATA_ROOT", WORKSPACE_ROOT / "data")
).expanduser().resolve()
ROBODOJO_DATASET_ROOT = Path(
    os.getenv(
        "ROBODOJO_DATASET_ROOT",
        DATA_ROOT / "RoboDojo_lerobot_v30_video",
    )
).expanduser().resolve()
VIDEO_HARNESS_RUN_ROOT = Path(
    os.getenv(
        "VIDEO_HARNESS_RUN_ROOT",
        DATA_ROOT / "video-harness" / "default",
    )
).expanduser().resolve()
