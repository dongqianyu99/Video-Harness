from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


class FrameDecodeError(RuntimeError):
    """Raised when a source frame cannot be decoded from RoboDojo video."""


class FFmpegFrameLoader:
    """Decode a referenced RoboDojo frame without materializing it on disk."""

    def __init__(
        self,
        dataset_root: Path,
        ffmpeg: str = "ffmpeg",
        *,
        rgb_shape: tuple[int, int, int] = (480, 640, 3),
    ) -> None:
        if (
            len(rgb_shape) != 3
            or rgb_shape[0] <= 0
            or rgb_shape[1] <= 0
            or rgb_shape[2] != 3
        ):
            raise ValueError(
                "rgb_shape must be a positive (height, width, 3) tuple"
            )
        self.dataset_root = dataset_root
        self.ffmpeg = ffmpeg
        self.rgb_shape = rgb_shape

    def _resolve_request(
        self,
        document: dict[str, Any],
        frame_ref: dict[str, Any],
    ) -> tuple[Path, float]:
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
        return video_path, timestamp

    def load(self, document: dict[str, Any], frame_ref: dict[str, Any]) -> bytes:
        """Return one JPEG frame for VLM provider requests."""

        video_path, timestamp = self._resolve_request(document, frame_ref)
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

    def load_rgb(
        self,
        document: dict[str, Any],
        frame_ref: dict[str, Any],
    ) -> np.ndarray:
        """Return one decoded uint8 RGB frame for an Actuator data path."""

        video_path, timestamp = self._resolve_request(document, frame_ref)
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
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        process = subprocess.run(command, check=False, capture_output=True)
        expected_size = int(np.prod(self.rgb_shape))
        if process.returncode != 0 or len(process.stdout) != expected_size:
            stderr = process.stderr.decode("utf-8", errors="replace").strip()
            raise FrameDecodeError(
                f"ffmpeg failed to decode RGB frame for {video_path} at "
                f"{timestamp:.6f}s: expected {expected_size} bytes, got "
                f"{len(process.stdout)}; {stderr or 'no ffmpeg error output'}"
            )
        return np.frombuffer(process.stdout, dtype=np.uint8).reshape(self.rgb_shape).copy()

    def load_rgb_many(
        self,
        document: dict[str, Any],
        frame_refs: Sequence[dict[str, Any]],
    ) -> tuple[np.ndarray, ...]:
        """Decode many episode-local frames with one FFmpeg process.

        Frames are decoded from the earliest requested timestamp and selected
        by their integral source-frame offsets.  Input order and duplicates are
        preserved in the returned tuple while FFmpeg only emits unique frames.
        """

        if not frame_refs:
            return ()

        resolved = [self._resolve_request(document, frame_ref) for frame_ref in frame_refs]
        video_paths = {video_path for video_path, _ in resolved}
        if len(video_paths) != 1:
            raise FrameDecodeError("One batch decode request must reference one video shard")
        video_path = next(iter(video_paths))

        source = document["source"]
        fps = float(source["fps"])
        frame_indices = [int(frame_ref["episode_frame_index"]) for frame_ref in frame_refs]
        first_index = min(frame_indices)
        first_timestamp = float(source["video_from_timestamp"]) + first_index / fps
        unique_indices = sorted(set(frame_indices))
        relative_indices = [index - first_index for index in unique_indices]
        select_expression = "+".join(
            f"eq(n\\,{relative_index})" for relative_index in relative_indices
        )

        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{first_timestamp:.6f}",
            "-i",
            str(video_path),
            "-vf",
            f"select={select_expression}",
            "-vsync",
            "0",
            "-frames:v",
            str(len(unique_indices)),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        process = subprocess.run(command, check=False, capture_output=True)
        frame_size = int(np.prod(self.rgb_shape))
        expected_size = frame_size * len(unique_indices)
        if process.returncode != 0 or len(process.stdout) != expected_size:
            stderr = process.stderr.decode("utf-8", errors="replace").strip()
            raise FrameDecodeError(
                f"ffmpeg batch decode failed for {video_path}: expected "
                f"{expected_size} bytes for {len(unique_indices)} frames, got "
                f"{len(process.stdout)}; {stderr or 'no ffmpeg error output'}"
            )

        decoded = np.frombuffer(process.stdout, dtype=np.uint8).reshape(
            (len(unique_indices), *self.rgb_shape)
        )
        by_index = {
            frame_index: np.array(decoded[position], copy=True)
            for position, frame_index in enumerate(unique_indices)
        }
        return tuple(np.array(by_index[frame_index], copy=True) for frame_index in frame_indices)
