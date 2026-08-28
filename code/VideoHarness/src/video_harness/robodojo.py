from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DLC_INSTRUCTION = 'Arrange the letters to spell "RoboDojo" in a row.'


class SourceContractError(ValueError):
    """Raised when a source cannot be treated as the Pi_05 RoboDojo dataset."""


@dataclass(frozen=True)
class VideoSlice:
    key: str
    path: str
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task_index: int
    task_instruction: str
    task_kind: str
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_path: str
    videos: tuple[VideoSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_pyarrow_parquet():
    try:
        from pyarrow import parquet
    except ImportError as exc:  # pragma: no cover - exercised in a minimal env
        raise RuntimeError(
            "Reading LeRobot metadata requires pyarrow. Install this package with "
            "`python -m pip install -e .`."
        ) from exc
    return parquet


def load_info(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "info.json"
    if not path.is_file():
        raise SourceContractError(f"Missing LeRobot metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_info(info: dict[str, Any]) -> None:
    expected_scalars = {
        "codebase_version": "v3.0",
        "fps": 25,
        "total_episodes": 3500,
        "total_frames": 1_859_602,
        "total_tasks": 35,
    }
    errors: list[str] = []
    for key, expected in expected_scalars.items():
        actual = info.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    features = info.get("features", {})
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict):
            errors.append(f"missing feature {key!r}")
            continue
        if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
            errors.append(
                f"{key}: expected float32[14], got "
                f"{feature.get('dtype')!r}{feature.get('shape')!r}"
            )

    for key in CAMERA_KEYS:
        feature = features.get(key)
        if not isinstance(feature, dict):
            errors.append(f"missing feature {key!r}")
            continue
        video_info = feature.get("info", {})
        if feature.get("dtype") != "video" or feature.get("shape") != [3, 480, 640]:
            errors.append(f"{key}: expected RGB video [3, 480, 640]")
        if (
            video_info.get("video.fps") != 25
            or video_info.get("video.is_depth_map") is not False
        ):
            errors.append(f"{key}: expected non-depth video at 25 FPS")

    for key in ("episode_index", "frame_index", "task_index", "timestamp"):
        if key not in features:
            errors.append(f"missing routing feature {key!r}")

    if errors:
        raise SourceContractError(
            "RoboDojo Pi_05 source contract failed:\n- " + "\n- ".join(errors)
        )


def read_tasks(dataset_root: Path) -> dict[str, int]:
    parquet = _load_pyarrow_parquet()
    path = dataset_root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise SourceContractError(f"Missing task metadata: {path}")
    table = parquet.read_table(path)
    columns = set(table.column_names)
    if "task_index" not in columns:
        raise SourceContractError("tasks.parquet has no task_index column")

    text_column = next(
        (
            candidate
            for candidate in ("task", "task_instruction", "__index_level_0__")
            if candidate in columns
        ),
        None,
    )
    if text_column is None:
        raise SourceContractError(
            "tasks.parquet has no supported task-text column; found "
            + ", ".join(sorted(columns))
        )

    mapping: dict[str, int] = {}
    for row in table.select(["task_index", text_column]).to_pylist():
        text = str(row[text_column]).strip()
        index = int(row["task_index"])
        if not text or text in mapping:
            raise SourceContractError(f"Invalid or duplicate task text: {text!r}")
        mapping[text] = index
    if len(mapping) != 35:
        raise SourceContractError(f"Expected 35 task labels, found {len(mapping)}")
    if DLC_INSTRUCTION not in mapping:
        raise SourceContractError("The expected auxiliary DLC task is absent")
    return mapping


def _format_route(template: str, **values: Any) -> str:
    try:
        return template.format(**values)
    except (KeyError, ValueError) as exc:
        raise SourceContractError(
            f"Cannot format LeRobot route {template!r}: {exc}"
        ) from exc


def read_episodes(dataset_root: Path) -> list[EpisodeRecord]:
    parquet = _load_pyarrow_parquet()
    info = load_info(dataset_root)
    validate_info(info)
    task_indices = read_tasks(dataset_root)

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise SourceContractError("No episode metadata parquet files were found")

    base_columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    video_columns = [
        f"videos/{key}/{suffix}"
        for key in CAMERA_KEYS
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    columns = base_columns + video_columns
    records: list[EpisodeRecord] = []

    for path in episode_files:
        table = parquet.read_table(path, columns=columns)
        for row in table.to_pylist():
            tasks = row["tasks"]
            if not isinstance(tasks, list) or len(tasks) != 1:
                raise SourceContractError(
                    f"Episode {row['episode_index']} must have exactly one task, got {tasks!r}"
                )
            instruction = str(tasks[0]).strip()
            if instruction not in task_indices:
                raise SourceContractError(
                    f"Episode {row['episode_index']} references an unknown task: {instruction!r}"
                )
            task_kind = "dlc" if instruction == DLC_INSTRUCTION else "benchmark"
            if task_kind == "dlc":
                continue

            data_path = _format_route(
                info["data_path"],
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            videos: list[VideoSlice] = []
            for key in CAMERA_KEYS:
                prefix = f"videos/{key}"
                video_path = _format_route(
                    info["video_path"],
                    video_key=key,
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                )
                videos.append(
                    VideoSlice(
                        key=key,
                        path=video_path,
                        from_timestamp=float(row[f"{prefix}/from_timestamp"]),
                        to_timestamp=float(row[f"{prefix}/to_timestamp"]),
                    )
                )

            length = int(row["length"])
            dataset_from_index = int(row["dataset_from_index"])
            dataset_to_index = int(row["dataset_to_index"])
            if dataset_to_index - dataset_from_index != length:
                raise SourceContractError(
                    f"Episode {row['episode_index']} has inconsistent dataset frame bounds"
                )
            for video in videos:
                duration = video.to_timestamp - video.from_timestamp
                # A small number of official shards expose a half-open video
                # range that is exactly one frame shorter than length / fps.
                # Reject larger routing errors but accept that release quirk.
                if abs(duration - length / info["fps"]) > 1 / info["fps"] + 1e-3:
                    raise SourceContractError(
                        f"Episode {row['episode_index']} camera {video.key} duration {duration} "
                        f"does not match {length}/{info['fps']}"
                    )

            records.append(
                EpisodeRecord(
                    episode_index=int(row["episode_index"]),
                    task_index=task_indices[instruction],
                    task_instruction=instruction,
                    task_kind=task_kind,
                    length=length,
                    dataset_from_index=dataset_from_index,
                    dataset_to_index=dataset_to_index,
                    data_path=data_path,
                    videos=tuple(videos),
                )
            )

    expected_episodes = 3400
    expected_tasks = 34
    actual_tasks = {record.task_index for record in records}
    if len(records) != expected_episodes or len(actual_tasks) != expected_tasks:
        raise SourceContractError(
            f"Benchmark-34 expected {expected_episodes} episodes/{expected_tasks} tasks, "
            f"found {len(records)} episodes/{len(actual_tasks)} tasks"
        )
    if len({record.episode_index for record in records}) != len(records):
        raise SourceContractError("Duplicate episode_index values were found")
    return sorted(records, key=lambda record: record.episode_index)


def summarize(records: Iterable[EpisodeRecord]) -> dict[str, Any]:
    records = list(records)
    counts: dict[int, int] = {}
    for record in records:
        counts[record.task_index] = counts.get(record.task_index, 0) + 1
    return {
        "schema_version": "video-harness.robodojo-source",
        "task_scope": "benchmark-34",
        "episodes": len(records),
        "tasks": len(counts),
        "frames": sum(record.length for record in records),
        "fps": 25,
        "episode_counts_by_task_index": {
            str(key): counts[key] for key in sorted(counts)
        },
    }


def load_episode_gripper_states(
    dataset_root: Path,
    *,
    data_path: str,
    episode_index: int,
    episode_length: int,
) -> np.ndarray:
    relative = Path(data_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SourceContractError(f"unsafe LeRobot data path: {relative}")
    path = Path(dataset_root) / relative
    parquet = _load_pyarrow_parquet()
    try:
        rows = parquet.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.state"],
            filters=[("episode_index", "=", episode_index)],
        ).to_pylist()
    except Exception as exc:  # noqa: BLE001 - normalized as a source error
        raise SourceContractError(
            f"cannot read LeRobot state parquet {path}: {exc}"
        ) from exc
    episode_rows = sorted(
        rows,
        key=lambda row: int(row["frame_index"]),
    )
    if len(episode_rows) != episode_length or [
        int(row["frame_index"]) for row in episode_rows
    ] != list(range(episode_length)):
        raise SourceContractError(
            f"episode {episode_index} state rows do not match length {episode_length}"
        )
    states = np.asarray(
        [row["observation.state"] for row in episode_rows],
        dtype=np.float32,
    )
    if states.shape != (episode_length, 14):
        raise SourceContractError(
            f"episode {episode_index} state must be [{episode_length},14], got {states.shape}"
        )
    grippers = states[:, (6, 13)]
    if not np.isfinite(grippers).all() or np.any((grippers < 0) | (grippers > 1)):
        raise SourceContractError(
            "LeRobot gripper state must be finite and within [0, 1]"
        )
    return grippers
