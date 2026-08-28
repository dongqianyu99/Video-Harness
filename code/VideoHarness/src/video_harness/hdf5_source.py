from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HDF5_SOURCE_DATASET = "RoboDojo_hdf5_v1.0"
HDF5_VIEW_KEYS = {
    "cam_high": "vision/cam_head/colors",
    "cam_left_wrist": "vision/cam_left_wrist/colors",
    "cam_right_wrist": "vision/cam_right_wrist/colors",
}
HDF5_GRIPPER_KEYS = {
    "left": "state/left_ee_joint_states",
    "right": "state/right_ee_joint_states",
}
_EPISODE_NAME = re.compile(r"episode_(\d+)\.hdf5$")


class HDF5SourceError(ValueError):
    """Raised when a standalone RoboDojo HDF5 episode is unusable."""


@dataclass(frozen=True)
class HDF5Episode:
    path: Path
    episode_index: int
    task_name: str
    task_instruction: str
    length: int
    fps: int
    image_shape: tuple[int, int, int]


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("Reading RoboDojo HDF5 episodes requires h5py") from exc
    return h5py


def _text(value: Any, field: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str) or not value.strip():
        raise HDF5SourceError(f"HDF5 {field} must be a non-empty string")
    return value.strip()


def inspect_hdf5_episode(path: Path) -> HDF5Episode:
    path = Path(path)
    if not path.is_file():
        raise HDF5SourceError(f"HDF5 episode does not exist: {path}")
    match = _EPISODE_NAME.fullmatch(path.name)
    if match is None:
        raise HDF5SourceError("HDF5 episode name must match episode_<index>.hdf5")

    h5py = _h5py()
    try:
        with h5py.File(path, "r") as file:
            version = _text(file["data_format_version"][()], "data_format_version")
            instruction = _text(file["instruction"][()], "instruction")
            fps = int(file["additional_info/frequency"][()])
            if version != "v1.0":
                raise HDF5SourceError(f"unsupported RoboDojo HDF5 version {version!r}")
            if fps != 25:
                raise HDF5SourceError(f"Video Harness requires 25 Hz, got {fps}")

            lengths: set[int] = set()
            shapes: set[tuple[int, int, int]] = set()
            for key in HDF5_VIEW_KEYS.values():
                dataset = file.get(key)
                shape_dataset = file.get(key.rsplit("/", 1)[0] + "/shape")
                if dataset is None or shape_dataset is None:
                    raise HDF5SourceError(f"missing HDF5 camera dataset {key!r}")
                if len(dataset.shape) != 1:
                    raise HDF5SourceError(
                        f"{key} must contain one encoded image per frame"
                    )
                lengths.add(int(dataset.shape[0]))
                image_shape = tuple(int(value) for value in shape_dataset[()])
                if len(image_shape) != 3:
                    raise HDF5SourceError(f"{key} camera shape must be H,W,3")
                shapes.add(image_shape)
            if len(lengths) != 1 or next(iter(lengths)) < 2:
                raise HDF5SourceError(
                    "all HDF5 camera streams must share a length >= 2"
                )
            frame_count = next(iter(lengths))
            for key in HDF5_GRIPPER_KEYS.values():
                dataset = file.get(key)
                if dataset is None or dataset.shape not in {
                    (frame_count,),
                    (frame_count, 1),
                }:
                    raise HDF5SourceError(
                        f"{key} must contain one gripper value per frame"
                    )
    except (KeyError, OSError) as exc:
        raise HDF5SourceError(f"cannot read RoboDojo HDF5 episode: {exc}") from exc

    if shapes != {(480, 640, 3)}:
        raise HDF5SourceError(
            f"all HDF5 camera streams must be RGB 480x640, got {sorted(shapes)}"
        )
    task_name = path.parent.name
    if task_name == "data" and len(path.parents) >= 3:
        task_name = path.parents[2].name
    return HDF5Episode(
        path=path,
        episode_index=int(match.group(1)),
        task_name=task_name,
        task_instruction=instruction,
        length=next(iter(lengths)),
        fps=fps,
        image_shape=next(iter(shapes)),
    )


def hdf5_document_source(episode: HDF5Episode) -> dict[str, Any]:
    return {
        "dataset": HDF5_SOURCE_DATASET,
        "episode_index": episode.episode_index,
        "episode_length": episode.length,
        # Standalone HDF5 files do not carry the global RoboDojo task index.
        "task_index": 0,
        "hdf5_path": episode.path.name,
        "views": {
            view: {"dataset_key": dataset_key}
            for view, dataset_key in HDF5_VIEW_KEYS.items()
        },
        "fps": episode.fps,
    }


def _decode_camera_jpeg(payload: bytes, *, context: str) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as image:
            decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise HDF5SourceError(f"cannot decode {context}") from exc
    # RoboDojo HDF5 v1.0 encoded RGB arrays through OpenCV's BGR JPEG path.
    # Reverse the stored display channels to match the official LeRobot export.
    return decoded[..., ::-1].copy()


def decode_hdf5_frames(
    path: Path,
    views: dict[str, str],
    *,
    start: int,
    end: int,
    fps: int,
) -> dict[str, np.ndarray]:
    if not 0 <= start <= end:
        raise HDF5SourceError(f"invalid HDF5 frame range [{start}, {end}]")
    h5py = _h5py()
    decoded: dict[str, np.ndarray] = {}
    try:
        with h5py.File(path, "r") as file:
            stored_fps = int(file["additional_info/frequency"][()])
            if stored_fps != fps:
                raise HDF5SourceError(
                    f"HDF5 frequency {stored_fps} does not match document FPS {fps}"
                )
            for view, key in views.items():
                dataset = file.get(key)
                if (
                    dataset is None
                    or len(dataset.shape) != 1
                    or end >= dataset.shape[0]
                ):
                    raise HDF5SourceError(
                        f"HDF5 view {view!r} cannot provide frames [{start}, {end}]"
                    )
                frames: list[np.ndarray] = []
                for encoded in dataset[start : end + 1]:
                    payload = bytes(encoded).rstrip(b"\0")
                    frames.append(
                        _decode_camera_jpeg(
                            payload,
                            context=f"{view} frame {start + len(frames)}",
                        )
                    )
                decoded[view] = np.stack(frames, axis=0)
    except (KeyError, OSError) as exc:
        raise HDF5SourceError(f"cannot read RoboDojo HDF5 frames: {exc}") from exc
    return decoded


def load_hdf5_jpeg(path: Path, dataset_key: str, frame_index: int) -> bytes:
    h5py = _h5py()
    try:
        with h5py.File(path, "r") as file:
            dataset = file[dataset_key]
            if not 0 <= frame_index < dataset.shape[0]:
                raise HDF5SourceError(
                    f"HDF5 frame {frame_index} is outside [0, {dataset.shape[0]})"
                )
            payload = bytes(dataset[frame_index]).rstrip(b"\0")
    except (KeyError, OSError) as exc:
        raise HDF5SourceError(f"cannot read HDF5 frame: {exc}") from exc
    if not payload.startswith(b"\xff\xd8"):
        raise HDF5SourceError("HDF5 camera frame is not a JPEG image")
    frame = _decode_camera_jpeg(payload, context=f"frame {frame_index}")
    output = BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=95)
    return output.getvalue()


def load_hdf5_gripper_states(path: Path) -> np.ndarray:
    h5py = _h5py()
    try:
        with h5py.File(path, "r") as file:
            columns = [
                np.asarray(file[key][:], dtype=np.float32).reshape(-1)
                for key in HDF5_GRIPPER_KEYS.values()
            ]
    except (KeyError, OSError) as exc:
        raise HDF5SourceError(f"cannot read HDF5 gripper state: {exc}") from exc
    states = np.stack(columns, axis=1)
    if not np.isfinite(states).all() or np.any((states < 0) | (states > 1)):
        raise HDF5SourceError("HDF5 gripper state must be finite and within [0, 1]")
    return states
