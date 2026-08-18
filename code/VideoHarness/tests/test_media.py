from types import SimpleNamespace

import numpy as np
import pytest

from video_harness import media as _media
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


def test_load_rgb_requests_raw_rgb24_and_restores_hwc_uint8(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "videos" / "cam" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video-placeholder")
    rgb = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    calls = []

    def fake_run(command, *, check, capture_output):
        calls.append((command, check, capture_output))
        return SimpleNamespace(returncode=0, stdout=rgb.tobytes(), stderr=b"")

    monkeypatch.setattr(_media.subprocess, "run", fake_run)
    loader = FFmpegFrameLoader(tmp_path, rgb_shape=(2, 4, 3))

    actual = loader.load_rgb(
        _document(),
        {"episode_frame_index": 25, "timestamp_s": 1.0},
    )

    np.testing.assert_array_equal(actual, rgb)
    assert actual.dtype == np.uint8
    assert actual.flags.owndata
    command = calls[0][0]
    assert command[command.index("-f") + 1] == "rawvideo"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"


def test_load_rgb_rejects_unexpected_frame_size(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "videos" / "cam" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video-placeholder")
    monkeypatch.setattr(
        _media.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"too-short",
            stderr=b"",
        ),
    )
    loader = FFmpegFrameLoader(tmp_path, rgb_shape=(2, 4, 3))

    with pytest.raises(FrameDecodeError, match=r"expected 24 bytes"):
        loader.load_rgb(
            _document(),
            {"episode_frame_index": 25, "timestamp_s": 1.0},
        )
