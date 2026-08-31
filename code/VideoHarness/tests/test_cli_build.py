from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from video_harness import cli
from video_harness.robodojo import EpisodeRecord, VideoSlice


def _episode(episode_index: int = 0) -> EpisodeRecord:
    length = 26
    return EpisodeRecord(
        episode_index=episode_index,
        task_index=3,
        task_instruction="Put bread into the toaster.",
        task_kind="benchmark",
        length=length,
        dataset_from_index=0,
        dataset_to_index=length,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=key,
                path=f"videos/{key}/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=length / 25,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ),
    )


def test_build_emits_only_document_source_artifacts_and_allows_one_episode(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "read_episodes", lambda _root: [_episode()])
    output_root = tmp_path / "build"
    args = argparse.Namespace(
        dataset_root=tmp_path / "dataset",
        output_root=output_root,
        sample_hz=1.0,
        max_tasks=1,
        episodes_per_task=1,
    )

    assert cli._build(args) == 0

    assert (output_root / "dataset.json").is_file()
    assert (output_root / "episodes.jsonl").is_file()
    assert (output_root / "documents.jsonl").is_file()
    assert not (output_root / "pairs.jsonl").exists()
    dataset = json.loads((output_root / "dataset.json").read_text())
    assert "supports_per_query" not in dataset
    assert "document_camera" not in dataset
    assert dataset["document_views"] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert "supports-" not in dataset["build_id"]
    assert "seed-" not in dataset["build_id"]


def test_build_parser_has_no_pair_or_split_options() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "make-training-split" not in help_text
    assert "pairs" not in help_text.lower()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build",
                "--dataset-root",
                "/tmp/dataset",
                "--output-root",
                "/tmp/output",
                "--supports-per-query",
                "1",
            ]
        )
