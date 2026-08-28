from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hdf5_source import HDF5_SOURCE_DATASET, load_hdf5_gripper_states
from .robodojo import load_episode_gripper_states

STANDARD_KEYFRAME_INDICES = (0, 5, 10, 15, 20, 25)


def keyframe_indices(frame_count: int) -> tuple[int, ...]:
    if not 2 <= frame_count <= 26:
        raise ValueError("keyframe sampling requires 2..26 frames")
    if frame_count == 26:
        return STANDARD_KEYFRAME_INDICES
    return tuple(
        sorted({round(value) for value in np.linspace(0, frame_count - 1, 6)})
    )


@dataclass(frozen=True)
class GripperState:
    unit_frame_indices: tuple[int, ...]
    left: tuple[float, ...]
    right: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.unit_frame_indices)
        if size < 2 or len(self.left) != size or len(self.right) != size:
            raise ValueError("gripper samples must share at least two frame indices")
        if tuple(sorted(set(self.unit_frame_indices))) != self.unit_frame_indices:
            raise ValueError("gripper frame indices must be unique and ordered")
        values = (*self.left, *self.right)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("gripper samples must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, list[int] | list[float]]:
        return {
            "unit_frame_indices": list(self.unit_frame_indices),
            "left": [round(value, 4) for value in self.left],
            "right": [round(value, 4) for value in self.right],
        }

    def prompt_text(self) -> str:
        frames = ", ".join(str(value) for value in self.unit_frame_indices)
        left = ", ".join(f"{value:.2f}" for value in self.left)
        right = ", ".join(f"{value:.2f}" for value in self.right)
        return (
            "Measured gripper aperture synchronized with the 2x3 keyframe sheets. "
            "1.00 means fully open; smaller values mean a narrower aperture. "
            f"Unit frames: [{frames}]. Left gripper: [{left}]. "
            f"Right gripper: [{right}]."
        )


class GripperStateReader:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)
        self._episodes: dict[str, np.ndarray] = {}

    def _episode(self, document: Mapping[str, Any]) -> np.ndarray:
        source = document["source"]
        key = str(document["document_id"])
        cached = self._episodes.get(key)
        if cached is not None:
            return cached
        dataset = source["dataset"]
        if dataset == HDF5_SOURCE_DATASET:
            relative = Path(str(source["hdf5_path"]))
            states = load_hdf5_gripper_states(self.dataset_root / relative)
        elif dataset == "RoboDojo_lerobot_v30_video":
            states = load_episode_gripper_states(
                self.dataset_root,
                data_path=str(source["data_path"]),
                episode_index=int(source["episode_index"]),
                episode_length=int(source["episode_length"]),
            )
        else:
            raise ValueError(f"unsupported gripper state source {dataset!r}")
        expected = (int(source["episode_length"]), 2)
        if states.shape != expected:
            raise ValueError(f"gripper episode state must have shape {expected}")
        self._episodes[key] = states
        return states

    def read_unit(
        self,
        document: Mapping[str, Any],
        *,
        episode_start_frame: int,
        episode_end_frame: int,
    ) -> GripperState:
        frame_count = episode_end_frame - episode_start_frame + 1
        indices = keyframe_indices(frame_count)
        episode = self._episode(document)
        episode_indices = [episode_start_frame + index for index in indices]
        samples = episode[episode_indices]
        return GripperState(
            unit_frame_indices=indices,
            left=tuple(float(value) for value in samples[:, 0]),
            right=tuple(float(value) for value in samples[:, 1]),
        )
