from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from .evidence import EVIDENCE_SCHEMA_VERSION, validate_evidence_record
from .robodojo import EpisodeRecord


BEHAVIOR_DOCUMENT_SCHEMA_VERSION = "video-harness.behavior-document.v0.2"


@dataclass(frozen=True)
class FrameRef:
    episode_frame_index: int
    timestamp_s: float


@dataclass(frozen=True)
class AnnotationSlot:
    schema_version: str
    status: str
    record: dict[str, Any] | None
    provenance: dict[str, str] | None


@dataclass(frozen=True)
class GuidanceUnit:
    unit_id: str
    order: int
    before: FrameRef
    annotation: AnnotationSlot
    after: FrameRef


def boundary_frames(length: int, fps: int, sample_hz: float) -> list[int]:
    if length < 2:
        raise ValueError("A document needs at least two source frames")
    if fps <= 0 or sample_hz <= 0:
        raise ValueError("fps and sample_hz must be positive")
    if sample_hz > fps:
        raise ValueError("sample_hz cannot exceed source fps")

    step = fps / sample_hz
    frames = [0]
    sample_index = 1
    while True:
        frame = min(length - 1, int(round(sample_index * step)))
        if frame >= length - 1:
            break
        if frame > frames[-1]:
            frames.append(frame)
        sample_index += 1
    if frames[-1] != length - 1:
        frames.append(length - 1)
    return frames


def plan_document(
    record: EpisodeRecord,
    *,
    build_id: str,
    fps: int = 25,
    sample_hz: float = 1.0,
) -> dict[str, Any]:
    boundaries = boundary_frames(record.length, fps, sample_hz)
    units = [
        GuidanceUnit(
            unit_id=f"u{order:04d}",
            order=order,
            before=FrameRef(episode_frame_index=start, timestamp_s=start / fps),
            annotation=AnnotationSlot(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                status="pending",
                record=None,
                provenance=None,
            ),
            after=FrameRef(episode_frame_index=end, timestamp_s=end / fps),
        )
        for order, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]
    head_video = next(video for video in record.videos if video.key == "observation.images.cam_high")
    return {
        "schema_version": BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
        "build_id": build_id,
        "status": "planned",
        "document_id": f"robodojo/episode-{record.episode_index:07d}",
        "source": {
            "dataset": "RoboDojo_lerobot_v30_video",
            "episode_index": record.episode_index,
            "episode_length": record.length,
            "task_index": record.task_index,
            "camera_key": head_video.key,
            "video_path": head_video.path,
            "video_from_timestamp": head_video.from_timestamp,
            "fps": fps,
        },
        "task_instruction": record.task_instruction,
        "sampling": {
            "kind": "uniform_guidance_unit",
            "sample_hz": sample_hz,
            "source_fps": fps,
        },
        "guidance_units": [asdict(unit) for unit in units],
    }


def _exact_keys(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{field} must have exactly {sorted(keys)}, got {actual}")
    return value


def _validate_frame_ref(value: Any, field: str, *, length: int, fps: int) -> int:
    frame = _exact_keys(value, field, {"episode_frame_index", "timestamp_s"})
    index = frame["episode_frame_index"]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < length:
        raise ValueError(f"{field}.episode_frame_index is outside [0, {length})")
    timestamp = frame["timestamp_s"]
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        raise ValueError(f"{field}.timestamp_s must be numeric")
    if abs(float(timestamp) - index / fps) > 1e-6:
        raise ValueError(f"{field}.timestamp_s does not match episode_frame_index / fps")
    return index


def validate_document(document: Any) -> dict[str, Any]:
    """Fail closed on source, sampling, frame, and annotation contract drift."""

    value = _exact_keys(
        document,
        "document",
        {
            "schema_version",
            "build_id",
            "status",
            "document_id",
            "source",
            "task_instruction",
            "sampling",
            "guidance_units",
        },
    )
    if value["schema_version"] != BEHAVIOR_DOCUMENT_SCHEMA_VERSION:
        raise ValueError(f"unexpected behavior document schema {value['schema_version']!r}")
    for field in ("build_id", "document_id", "task_instruction"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"document.{field} must be a non-empty string")

    source = _exact_keys(
        value["source"],
        "document.source",
        {
            "dataset",
            "episode_index",
            "episode_length",
            "task_index",
            "camera_key",
            "video_path",
            "video_from_timestamp",
            "fps",
        },
    )
    if source["dataset"] != "RoboDojo_lerobot_v30_video":
        raise ValueError(f"unsupported document source dataset {source['dataset']!r}")
    for field in ("episode_index", "task_index"):
        if not isinstance(source[field], int) or isinstance(source[field], bool) or source[field] < 0:
            raise ValueError(f"document.source.{field} must be a non-negative integer")
    length = source["episode_length"]
    fps = source["fps"]
    if not isinstance(length, int) or isinstance(length, bool) or length < 2:
        raise ValueError("document.source.episode_length must be an integer >= 2")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("document.source.fps must be a positive integer")
    if source["camera_key"] != "observation.images.cam_high":
        raise ValueError("behavior documents must use observation.images.cam_high")
    video_path = source["video_path"]
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("document.source.video_path must be a non-empty relative path")
    pure_video_path = PurePosixPath(video_path)
    if pure_video_path.is_absolute() or ".." in pure_video_path.parts:
        raise ValueError("document.source.video_path must stay within the dataset root")
    if not isinstance(source["video_from_timestamp"], (int, float)) or source[
        "video_from_timestamp"
    ] < 0:
        raise ValueError("document.source.video_from_timestamp must be non-negative")

    sampling = _exact_keys(
        value["sampling"],
        "document.sampling",
        {"kind", "sample_hz", "source_fps"},
    )
    if sampling["kind"] != "uniform_guidance_unit":
        raise ValueError(f"unsupported sampling kind {sampling['kind']!r}")
    sample_hz = sampling["sample_hz"]
    if (
        not isinstance(sample_hz, (int, float))
        or isinstance(sample_hz, bool)
        or not 0 < sample_hz <= fps
    ):
        raise ValueError("document.sampling.sample_hz must be within (0, source_fps]")
    if sampling["source_fps"] != fps:
        raise ValueError("document sampling/source FPS mismatch")

    units = value["guidance_units"]
    if not isinstance(units, list) or not units:
        raise ValueError("document.guidance_units must be a non-empty list")
    statuses: list[str] = []
    previous_after: int | None = None
    for order, raw_unit in enumerate(units):
        unit = _exact_keys(
            raw_unit,
            f"guidance_units[{order}]",
            {"unit_id", "order", "before", "annotation", "after"},
        )
        if unit["unit_id"] != f"u{order:04d}" or unit["order"] != order:
            raise ValueError(f"guidance_units[{order}] has inconsistent unit identity/order")
        before = _validate_frame_ref(unit["before"], f"guidance_units[{order}].before", length=length, fps=fps)
        after = _validate_frame_ref(unit["after"], f"guidance_units[{order}].after", length=length, fps=fps)
        if before >= after:
            raise ValueError(f"guidance_units[{order}] must advance in time")
        if order == 0 and before != 0:
            raise ValueError("the first guidance unit must start at episode frame 0")
        if previous_after is not None and before != previous_after:
            raise ValueError(f"guidance_units[{order}] does not share the previous boundary")
        previous_after = after

        annotation = _exact_keys(
            unit["annotation"],
            f"guidance_units[{order}].annotation",
            {"schema_version", "status", "record", "provenance"},
        )
        if annotation["schema_version"] != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"guidance_units[{order}] has unexpected evidence schema "
                f"{annotation['schema_version']!r}"
            )
        status = annotation["status"]
        if status not in {"pending", "complete", "mock", "failed"}:
            raise ValueError(f"guidance_units[{order}] has unsupported status {status!r}")
        statuses.append(status)
        if status in {"pending", "failed"}:
            if annotation["record"] is not None or annotation["provenance"] is not None:
                raise ValueError(f"guidance_units[{order}] {status} annotation must be empty")
            continue
        validate_evidence_record(annotation["record"])
        provenance = annotation["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "provider",
            "model",
            "prompt_version",
        }:
            raise ValueError(f"guidance_units[{order}] requires exact annotation provenance")
        if any(
            not isinstance(provenance[field], str) or not provenance[field].strip()
            for field in ("provider", "model", "prompt_version")
        ):
            raise ValueError(f"guidance_units[{order}] provenance fields must be non-empty strings")

    if previous_after != length - 1:
        raise ValueError("the final guidance unit must end at the final episode frame")
    status_set = set(statuses)
    expected_document_status = (
        "planned"
        if status_set == {"pending"}
        else "annotated"
        if status_set == {"complete"}
        else "mock-annotated"
        if status_set == {"mock"}
        else "partially-annotated"
    )
    if value["status"] != expected_document_status:
        raise ValueError(
            f"document status {value['status']!r} does not match unit statuses; "
            f"expected {expected_document_status!r}"
        )
    return value


def media_timestamp(document: dict[str, Any], source_frame: int) -> float:
    source = document["source"]
    return float(source["video_from_timestamp"]) + source_frame / int(source["fps"])
