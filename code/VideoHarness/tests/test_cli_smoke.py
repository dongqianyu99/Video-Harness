from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
from pyarrow import parquet

from video_harness.cli import _annotate
from video_harness.evidence import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document


def test_mock_cli_runs_full_multiview_media_pipeline_without_debug(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    video = dataset_root / "videos" / "shared.mkv"
    video.parent.mkdir(parents=True)
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x480:rate=25",
            "-frames:v",
            "26",
            "-c:v",
            "ffv1",
            "-y",
            str(video),
        ],
        capture_output=True,
        check=False,
    )
    assert encoded.returncode == 0, encoded.stderr.decode(errors="replace")
    state_path = dataset_root / "data/chunk-000/file-000.parquet"
    state_path.parent.mkdir(parents=True)
    states = [[0.0] * 6 + [1.0] + [0.0] * 6 + [1.0] for _ in range(26)]
    parquet.write_table(
        pa.table(
            {
                "episode_index": [0] * 26,
                "frame_index": list(range(26)),
                "observation.state": states,
            }
        ),
        state_path,
    )

    record = EpisodeRecord(
        episode_index=0,
        task_index=0,
        task_instruction="Place the visible object on the target.",
        task_kind="benchmark",
        length=26,
        dataset_from_index=0,
        dataset_to_index=26,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=key,
                path="videos/shared.mkv",
                from_timestamp=0.0,
                to_timestamp=26 / 25,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ),
    )
    documents_path = tmp_path / "documents.jsonl"
    documents_path.write_text(
        json.dumps(plan_document(record, build_id="smoke-build")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "documents.mock.jsonl"
    args = SimpleNamespace(
        documents=documents_path,
        output=output,
        dataset_root=dataset_root,
        provider="mock",
        model=None,
        limit_documents=1,
        limit_units_per_document=1,
        debug=False,
        debug_root=None,
        inspection_retries=0,
    )

    assert _annotate(args) == 0
    annotated = json.loads(output.read_text())
    annotation = annotated["evidence_units"][0]["annotation"]
    assert annotation["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert annotation["status"] == "mock"
    assert "quality_status" not in annotation["record"]
    assert "causal_validation" not in annotation["record"]
    assert "gripper_state" not in annotation["record"]
    assert set(annotation["provenance"]) == {"call1", "call2", "repair"}
    assert set(annotation["provenance"]["call2"]) == {
        "provider",
        "model",
        "prompt_version",
    }
    assert len(annotated["boundary_states"]) == 2
    for boundary in annotated["boundary_states"]:
        boundary_annotation = boundary["annotation"]
        assert boundary_annotation["schema_version"] == BOUNDARY_STATE_SCHEMA_VERSION
        assert boundary_annotation["status"] == "mock"
        assert "quality_status" not in boundary_annotation["record"]
    assert (
        annotated["evidence_units"][0]["after_boundary_id"]
        == annotated["boundary_states"][1]["boundary_id"]
    )
    assert not list(tmp_path.glob("*.debug"))
    assert not (tmp_path / "debug").exists()
