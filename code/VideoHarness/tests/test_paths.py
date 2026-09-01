from video_harness import paths
from video_harness.cli import build_parser


def test_default_paths_share_repository_data_root():
    assert paths.ROBODOJO_DATASET_ROOT == (
        paths.DATA_ROOT / "RoboDojo_lerobot_v30_video"
    )
    assert paths.VIDEO_HARNESS_RUN_ROOT == (
        paths.DATA_ROOT / "video-harness"
    )


def test_cli_uses_standard_dataset_and_run_roots_by_default():
    build = build_parser().parse_args(["build"])
    annotate = build_parser().parse_args(["annotate", "--provider", "mock"])

    assert build.dataset_root == paths.ROBODOJO_DATASET_ROOT
    assert build.output_root == paths.VIDEO_HARNESS_RUN_ROOT
    assert annotate.dataset_root == paths.ROBODOJO_DATASET_ROOT
    assert annotate.documents == paths.VIDEO_HARNESS_RUN_ROOT / "documents.jsonl"
    assert annotate.output_mode == "json"
    assert annotate.thinking is True
