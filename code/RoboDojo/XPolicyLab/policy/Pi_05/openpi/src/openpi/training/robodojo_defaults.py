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
os.environ.setdefault("HF_LEROBOT_HOME", str(DATA_ROOT))
ROBODOJO_REPO_ID = os.getenv(
    "ROBODOJO_REPO_ID", "RoboDojo_lerobot_v30_video"
)
ROBODOJO_DATASET_ROOT = Path(
    os.getenv("ROBODOJO_DATASET_ROOT", DATA_ROOT / ROBODOJO_REPO_ID)
).expanduser().resolve()
GUIDE_DOCUMENTS_ROOT = Path(
    os.getenv(
        "GUIDE_DOCUMENTS_ROOT",
        DATA_ROOT / "video-harness" / "default" / "documents-openai",
    )
).expanduser().resolve()
GUIDE_MATERIALIZATION_CACHE_ROOT = Path(
    os.getenv(
        "GUIDE_MATERIALIZATION_CACHE_ROOT",
        DATA_ROOT / "guide-cache" / "default",
    )
).expanduser().resolve()

# Exact 1 Hz structural maxima derived from the official 3,400 benchmark episodes.
MAX_BOUNDARIES = 63
MAX_UNITS = 62
MAX_BOUNDARY_TEXT_TOKENS = 128
MAX_TRANSITION_TEXT_TOKENS = 128
GUIDES_PER_BATCH = 1
QUERIES_PER_GUIDE = 64
GRADIENT_ACCUMULATION_STEPS = 4
