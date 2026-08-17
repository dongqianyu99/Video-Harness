from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any


class FrameDecodeError(RuntimeError):
    """Raised when a source frame cannot be decoded from RoboDojo video."""


class FFmpegFrameLoader:
    """Decode a referenced frame to JPEG bytes without materializing it on disk."""

    def __init__(self, dataset_root: Path, ffmpeg: str = "ffmpeg") -> None:
        self.dataset_root = dataset_root
        self.ffmpeg = ffmpeg

    def load(self, document: dict[str, Any], frame_ref: dict[str, Any]) -> bytes:
        source = document["source"]
        try:
            frame_index = int(frame_ref["episode_frame_index"])
            episode_length = int(source["episode_length"])
            fps = float(source["fps"])
            stored_timestamp = float(frame_ref["timestamp_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameDecodeError("Invalid frame reference or document source metadata") from exc
        if fps <= 0:
            raise FrameDecodeError(f"Invalid source FPS: {fps}")
        if not 0 <= frame_index < episode_length:
            raise FrameDecodeError(
                f"Episode-local frame {frame_index} is outside [0, {episode_length})"
            )
        expected_timestamp = frame_index / fps
        if not math.isclose(stored_timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-6):
            raise FrameDecodeError(
                "Frame reference timestamp disagrees with episode_frame_index/source FPS: "
                f"stored={stored_timestamp:.6f}, expected={expected_timestamp:.6f}"
            )
        relative_path = Path(source["video_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FrameDecodeError(f"Unsafe dataset-relative video path: {relative_path}")
        video_path = self.dataset_root / relative_path
        if not video_path.is_file():
            raise FrameDecodeError(f"Video shard does not exist: {video_path}")
        timestamp = float(source["video_from_timestamp"]) + expected_timestamp
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        process = subprocess.run(command, check=False, capture_output=True)
        if process.returncode != 0 or not process.stdout.startswith(b"\xff\xd8"):
            stderr = process.stderr.decode("utf-8", errors="replace").strip()
            raise FrameDecodeError(
                f"ffmpeg failed for {video_path} at {timestamp:.6f}s: {stderr or 'no JPEG output'}"
            )
        return process.stdout
