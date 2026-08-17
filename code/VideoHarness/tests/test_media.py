import pytest

from video_harness.media import FFmpegFrameLoader, FrameDecodeError


def _document() -> dict:
    return {
        "source": {
            "episode_length": 100,
            "fps": 25,
            "video_from_timestamp": 10.0,
            "video_path": "videos/cam/file-000.mp4",
        }
    }


def test_loader_rejects_drifting_redundant_timestamp(tmp_path) -> None:
    loader = FFmpegFrameLoader(tmp_path)
    with pytest.raises(FrameDecodeError, match="timestamp disagrees"):
        loader.load(
            _document(),
            {"episode_frame_index": 25, "timestamp_s": 1.25},
        )


def test_loader_rejects_frame_outside_episode(tmp_path) -> None:
    loader = FFmpegFrameLoader(tmp_path)
    with pytest.raises(FrameDecodeError, match="outside"):
        loader.load(
            _document(),
            {"episode_frame_index": 100, "timestamp_s": 4.0},
        )
