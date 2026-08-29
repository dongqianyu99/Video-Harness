from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from .evidence import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    validate_boundary_state_record,
    validate_evidence_record,
)
from .hdf5_source import HDF5_SOURCE_DATASET, HDF5_VIEW_KEYS
from .robodojo import EpisodeRecord

BEHAVIOR_DOCUMENT_SCHEMA_VERSION = "video-harness.behavior-document"
MIN_ACCEPTED_UNIT_RATIO = 0.90

_VIEW_ALIASES = {
    "observation.images.cam_high": "cam_high",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}


@dataclass(frozen=True)
class FrameRef:
    episode_frame_index: int
    timestamp_s: float


@dataclass(frozen=True)
class AnnotationSlot:
    schema_version: str
    status: str
    record: dict[str, Any] | None
    provenance: dict[str, Any] | None


@dataclass(frozen=True)
class BoundaryState:
    boundary_id: str
    order: int
    frame: FrameRef
    annotation: AnnotationSlot


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    order: int
    before_boundary_id: str
    annotation: AnnotationSlot
    after_boundary_id: str


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
        frame = min(length - 1, round(sample_index * step))
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
    head_video = next(
        video for video in record.videos if video.key == "observation.images.cam_high"
    )
    videos = {video.key: video for video in record.videos}
    if set(videos) != set(_VIEW_ALIASES):
        raise ValueError(
            "RoboDojo document source requires exactly three camera videos"
        )
    view_sources = {
        alias: {
            "camera_key": camera_key,
            "video_path": videos[camera_key].path,
            "video_from_timestamp": videos[camera_key].from_timestamp,
        }
        for camera_key, alias in _VIEW_ALIASES.items()
    }
    return plan_document_from_source(
        build_id=build_id,
        document_id=f"robodojo/episode-{record.episode_index:07d}",
        source={
            "dataset": "RoboDojo_lerobot_v30_video",
            "episode_index": record.episode_index,
            "episode_length": record.length,
            "task_index": record.task_index,
            "camera_key": head_video.key,
            "video_path": head_video.path,
            "video_from_timestamp": head_video.from_timestamp,
            "data_path": record.data_path,
            "dataset_from_index": record.dataset_from_index,
            "dataset_to_index": record.dataset_to_index,
            "views": view_sources,
            "fps": fps,
        },
        task_instruction=record.task_instruction,
        sample_hz=sample_hz,
    )


def plan_document_from_source(
    *,
    build_id: str,
    document_id: str,
    source: dict[str, Any],
    task_instruction: str,
    sample_hz: float = 1.0,
) -> dict[str, Any]:
    length = int(source["episode_length"])
    fps = int(source["fps"])
    boundaries = boundary_frames(length, fps, sample_hz)
    boundary_states = [
        BoundaryState(
            boundary_id=f"b{order:04d}",
            order=order,
            frame=FrameRef(
                episode_frame_index=frame,
                timestamp_s=frame / fps,
            ),
            annotation=AnnotationSlot(
                schema_version=BOUNDARY_STATE_SCHEMA_VERSION,
                status="pending",
                record=None,
                provenance=None,
            ),
        )
        for order, frame in enumerate(boundaries)
    ]
    units = [
        EvidenceUnit(
            unit_id=f"u{order:04d}",
            order=order,
            before_boundary_id=f"b{order:04d}",
            annotation=AnnotationSlot(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                status="pending",
                record=None,
                provenance=None,
            ),
            after_boundary_id=f"b{order + 1:04d}",
        )
        for order in range(len(boundaries) - 1)
    ]
    return {
        "schema_version": BEHAVIOR_DOCUMENT_SCHEMA_VERSION,
        "build_id": build_id,
        "status": "planned",
        "quality_status": "pending",
        "quality_provenance": None,
        "document_id": document_id,
        "source": source,
        "task_instruction": task_instruction,
        "sampling": {
            "kind": "uniform_evidence_unit",
            "sample_hz": sample_hz,
            "source_fps": fps,
        },
        "boundary_states": [asdict(boundary) for boundary in boundary_states],
        "evidence_units": [asdict(unit) for unit in units],
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
        raise TypeError(f"{field}.timestamp_s must be numeric")
    if abs(float(timestamp) - index / fps) > 1e-6:
        raise ValueError(
            f"{field}.timestamp_s does not match episode_frame_index / fps"
        )
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
            "quality_status",
            "quality_provenance",
            "document_id",
            "source",
            "task_instruction",
            "sampling",
            "boundary_states",
            "evidence_units",
        },
    )
    if value["schema_version"] != BEHAVIOR_DOCUMENT_SCHEMA_VERSION:
        raise ValueError(
            f"unexpected behavior document schema {value['schema_version']!r}"
        )
    for field in ("build_id", "document_id", "task_instruction"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"document.{field} must be a non-empty string")

    raw_source = value["source"]
    if not isinstance(raw_source, dict):
        raise TypeError("document.source must be an object")
    dataset = raw_source.get("dataset")
    if dataset == "RoboDojo_lerobot_v30_video":
        source = _exact_keys(
            raw_source,
            "document.source",
            {
                "dataset",
                "episode_index",
                "episode_length",
                "task_index",
                "camera_key",
                "video_path",
                "video_from_timestamp",
                "data_path",
                "dataset_from_index",
                "dataset_to_index",
                "views",
                "fps",
            },
        )
    elif dataset == HDF5_SOURCE_DATASET:
        source = _exact_keys(
            raw_source,
            "document.source",
            {
                "dataset",
                "episode_index",
                "episode_length",
                "task_index",
                "hdf5_path",
                "views",
                "fps",
            },
        )
    else:
        raise ValueError(f"unsupported document source dataset {dataset!r}")

    for field in ("episode_index", "task_index"):
        if (
            not isinstance(source[field], int)
            or isinstance(source[field], bool)
            or source[field] < 0
        ):
            raise ValueError(f"document.source.{field} must be a non-negative integer")
    length = source["episode_length"]
    fps = source["fps"]
    if not isinstance(length, int) or isinstance(length, bool) or length < 2:
        raise ValueError("document.source.episode_length must be an integer >= 2")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("document.source.fps must be a positive integer")
    views = _exact_keys(
        source["views"], "document.source.views", set(_VIEW_ALIASES.values())
    )
    if dataset == HDF5_SOURCE_DATASET:
        hdf5_path = source["hdf5_path"]
        if not isinstance(hdf5_path, str) or not hdf5_path:
            raise ValueError("document.source.hdf5_path must be a relative path")
        pure_path = PurePosixPath(hdf5_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError(
                "document.source.hdf5_path must stay within the dataset root"
            )
        for alias, expected_key in HDF5_VIEW_KEYS.items():
            item = _exact_keys(
                views[alias], f"document.source.views.{alias}", {"dataset_key"}
            )
            if item["dataset_key"] != expected_key:
                raise ValueError(f"document source view {alias} dataset key mismatch")
    else:
        if source["camera_key"] != "observation.images.cam_high":
            raise ValueError("behavior documents must use observation.images.cam_high")
        data_path = source["data_path"]
        if not isinstance(data_path, str) or not data_path:
            raise ValueError("document.source.data_path must be a relative path")
        pure_data_path = PurePosixPath(data_path)
        if pure_data_path.is_absolute() or ".." in pure_data_path.parts:
            raise ValueError(
                "document.source.data_path must stay within the dataset root"
            )
        dataset_from = source["dataset_from_index"]
        dataset_to = source["dataset_to_index"]
        if (
            not isinstance(dataset_from, int)
            or isinstance(dataset_from, bool)
            or not isinstance(dataset_to, int)
            or isinstance(dataset_to, bool)
            or dataset_from < 0
            or dataset_to - dataset_from != length
        ):
            raise ValueError("document source dataset frame bounds are invalid")
        video_path = source["video_path"]
        if not isinstance(video_path, str) or not video_path:
            raise ValueError(
                "document.source.video_path must be a non-empty relative path"
            )
        pure_video_path = PurePosixPath(video_path)
        if pure_video_path.is_absolute() or ".." in pure_video_path.parts:
            raise ValueError(
                "document.source.video_path must stay within the dataset root"
            )
        if (
            not isinstance(source["video_from_timestamp"], (int, float))
            or source["video_from_timestamp"] < 0
        ):
            raise ValueError(
                "document.source.video_from_timestamp must be non-negative"
            )
        for alias, camera_key in ((alias, key) for key, alias in _VIEW_ALIASES.items()):
            item = _exact_keys(
                views[alias],
                f"document.source.views.{alias}",
                {"camera_key", "video_path", "video_from_timestamp"},
            )
            if item["camera_key"] != camera_key:
                raise ValueError(f"document source view {alias} camera key mismatch")
            path = item["video_path"]
            if not isinstance(path, str) or not path:
                raise ValueError(f"document source view {alias} video_path is invalid")
            pure_path = PurePosixPath(path)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                raise ValueError(f"document source view {alias} path is unsafe")
            if (
                not isinstance(item["video_from_timestamp"], (int, float))
                or isinstance(item["video_from_timestamp"], bool)
                or item["video_from_timestamp"] < 0
            ):
                raise ValueError(f"document source view {alias} timestamp is invalid")
        if views["cam_high"]["video_path"] != source["video_path"]:
            raise ValueError("primary cam_high path does not match multiview source")
        if views["cam_high"]["video_from_timestamp"] != source["video_from_timestamp"]:
            raise ValueError(
                "primary cam_high timestamp does not match multiview source"
            )

    sampling = _exact_keys(
        value["sampling"],
        "document.sampling",
        {"kind", "sample_hz", "source_fps"},
    )
    if sampling["kind"] != "uniform_evidence_unit":
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

    boundaries = value["boundary_states"]
    units = value["evidence_units"]
    if not isinstance(boundaries, list) or len(boundaries) < 2:
        raise ValueError("document.boundary_states must contain at least two entries")
    if not isinstance(units, list) or not units:
        raise ValueError("document.evidence_units must be a non-empty list")
    if len(boundaries) != len(units) + 1:
        raise ValueError(
            "document must contain exactly one more Boundary State than Evidence Units"
        )

    boundary_statuses: list[str] = []
    boundary_frames: list[int] = []
    for order, raw_boundary in enumerate(boundaries):
        boundary = _exact_keys(
            raw_boundary,
            f"boundary_states[{order}]",
            {"boundary_id", "order", "frame", "annotation"},
        )
        if boundary["boundary_id"] != f"b{order:04d}" or boundary["order"] != order:
            raise ValueError(
                f"boundary_states[{order}] has inconsistent boundary identity/order"
            )
        frame = _validate_frame_ref(
            boundary["frame"],
            f"boundary_states[{order}].frame",
            length=length,
            fps=fps,
        )
        if boundary_frames and frame <= boundary_frames[-1]:
            raise ValueError("Boundary State frames must advance strictly")
        boundary_frames.append(frame)

        annotation = _exact_keys(
            boundary["annotation"],
            f"boundary_states[{order}].annotation",
            {"schema_version", "status", "record", "provenance"},
        )
        if annotation["schema_version"] != BOUNDARY_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"boundary_states[{order}] has unexpected schema "
                f"{annotation['schema_version']!r}"
            )
        status = annotation["status"]
        if status not in {"pending", "complete", "mock", "failed"}:
            raise ValueError(
                f"boundary_states[{order}] has unsupported status {status!r}"
            )
        boundary_statuses.append(status)
        if status in {"pending", "failed"}:
            if annotation["record"] is not None or annotation["provenance"] is not None:
                raise ValueError(
                    f"boundary_states[{order}] {status} annotation must be empty"
                )
            continue
        validate_boundary_state_record(annotation["record"])
        provenance = annotation["provenance"]
        required = {
            "provider",
            "model",
            "prompt_version",
            "source_unit_id",
            "boundary_role",
        }
        if not isinstance(provenance, dict) or set(provenance) != required:
            raise ValueError(
                f"boundary_states[{order}] has invalid annotation provenance"
            )
        if provenance["boundary_role"] not in {"before", "after"}:
            raise ValueError(
                f"boundary_states[{order}] provenance boundary_role is invalid"
            )
        if (provenance["boundary_role"] == "before" and order >= len(units)) or (
            provenance["boundary_role"] == "after" and order == 0
        ):
            raise ValueError(
                f"boundary_states[{order}] provenance role cannot reference an Evidence Unit"
            )
        expected_source_order = (
            order if provenance["boundary_role"] == "before" else order - 1
        )
        if provenance["source_unit_id"] != f"u{expected_source_order:04d}":
            raise ValueError(
                f"boundary_states[{order}] provenance source unit is inconsistent"
            )
        if any(
            not isinstance(provenance[field], str) or not provenance[field].strip()
            for field in ("provider", "model", "prompt_version")
        ):
            raise ValueError(
                f"boundary_states[{order}] provenance fields must be non-empty strings"
            )

    if boundary_frames[0] != 0 or boundary_frames[-1] != length - 1:
        raise ValueError("Boundary States must cover the first and final episode frame")

    unit_statuses: list[str] = []
    for order, raw_unit in enumerate(units):
        unit = _exact_keys(
            raw_unit,
            f"evidence_units[{order}]",
            {
                "unit_id",
                "order",
                "before_boundary_id",
                "annotation",
                "after_boundary_id",
            },
        )
        if unit["unit_id"] != f"u{order:04d}" or unit["order"] != order:
            raise ValueError(
                f"evidence_units[{order}] has inconsistent unit identity/order"
            )
        if (
            unit["before_boundary_id"] != f"b{order:04d}"
            or unit["after_boundary_id"] != f"b{order + 1:04d}"
        ):
            raise ValueError(
                f"evidence_units[{order}] must reference its adjacent Boundary States"
            )

        annotation = _exact_keys(
            unit["annotation"],
            f"evidence_units[{order}].annotation",
            {"schema_version", "status", "record", "provenance"},
        )
        if annotation["schema_version"] != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"evidence_units[{order}] has unexpected evidence schema "
                f"{annotation['schema_version']!r}"
            )
        status = annotation["status"]
        if status not in {"pending", "complete", "mock", "failed"}:
            raise ValueError(
                f"evidence_units[{order}] has unsupported status {status!r}"
            )
        unit_statuses.append(status)
        if status in {"pending", "failed"}:
            if annotation["record"] is not None or annotation["provenance"] is not None:
                raise ValueError(
                    f"evidence_units[{order}] {status} annotation must be empty"
                )
            continue
        for boundary_index in (order, order + 1):
            allowed_boundary_statuses = (
                {"complete"} if status == "complete" else {"complete", "mock"}
            )
            if boundary_statuses[boundary_index] not in allowed_boundary_statuses:
                raise ValueError(
                    f"evidence_units[{order}] {status} annotation requires "
                    "usable adjacent Boundary States"
                )
        validate_evidence_record(annotation["record"])
        provenance = annotation["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "call1",
            "call2",
            "repair",
        }:
            raise ValueError(
                f"evidence_units[{order}] requires exact annotation provenance"
            )
        call1 = provenance["call1"]
        call2 = provenance["call2"]
        repair = provenance["repair"]
        if not isinstance(call1, dict) or set(call1) != {
            "provider",
            "model",
            "prompt_version",
        }:
            raise ValueError(f"evidence_units[{order}] has invalid Call 1 provenance")
        if not isinstance(call2, dict) or set(call2) != {
            "provider",
            "model",
            "prompt_version",
        }:
            raise ValueError(f"evidence_units[{order}] has invalid Call 2 provenance")
        if repair is not None:
            if not isinstance(repair, dict) or set(repair) != {
                "provider",
                "model",
                "prompt_version",
                "attempts",
                "reason",
            }:
                raise ValueError(
                    f"evidence_units[{order}] has invalid repair provenance"
                )
            if any(
                not isinstance(repair[field], str) or not repair[field].strip()
                for field in ("provider", "model", "prompt_version", "reason")
            ):
                raise ValueError(
                    f"evidence_units[{order}] repair provenance fields must be non-empty"
                )
            if (
                isinstance(repair["attempts"], bool)
                or not isinstance(repair["attempts"], int)
                or repair["attempts"] < 1
            ):
                raise ValueError(
                    f"evidence_units[{order}] repair attempts must be positive"
                )
        if any(
            not isinstance(scope[field], str) or not scope[field].strip()
            for scope in (call1, call2)
            for field in ("provider", "model", "prompt_version")
        ):
            raise ValueError(
                f"evidence_units[{order}] provenance fields must be non-empty strings"
            )
    status_set = set(boundary_statuses + unit_statuses)
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
            f"document status {value['status']!r} does not match boundary/unit statuses; "
            f"expected {expected_document_status!r}"
        )
    quality_status = value["quality_status"]
    if quality_status not in {"pending", "accepted", "quarantined"}:
        raise ValueError("document.quality_status is unsupported")
    quality_provenance = value["quality_provenance"]
    if quality_status == "pending":
        if quality_provenance is not None:
            raise ValueError("pending documents must not have quality provenance")
    else:
        required_quality_provenance = {
            "provider",
            "model",
            "prompt_version",
            "audit_attempts",
            "repair_rounds",
            "sequence_sha256",
            "issues",
        }
        if (
            not isinstance(quality_provenance, dict)
            or set(quality_provenance) != required_quality_provenance
        ):
            raise ValueError("final document quality provenance is invalid")
        for field in ("provider", "model", "prompt_version", "sequence_sha256"):
            if (
                not isinstance(quality_provenance[field], str)
                or not quality_provenance[field].strip()
            ):
                raise ValueError(f"quality provenance {field} must be non-empty")
        for field in ("audit_attempts", "repair_rounds"):
            if (
                isinstance(quality_provenance[field], bool)
                or not isinstance(quality_provenance[field], int)
                or quality_provenance[field] < 0
            ):
                raise ValueError(f"quality provenance {field} must be non-negative")
        if not isinstance(quality_provenance["issues"], list):
            raise ValueError("quality provenance issues must be a list")
    if value["status"] == "planned" and quality_status != "pending":
        raise ValueError("planned documents must have quality_status='pending'")
    if value["status"] == "partially-annotated" and quality_status not in {
        "pending",
        "quarantined",
    }:
        raise ValueError("partial documents must be pending or quarantined")
    if quality_status == "accepted":
        if value["status"] != "annotated":
            raise ValueError("only fully annotated real documents can be accepted")
        if any(
            boundary["annotation"]["record"]["quality_status"] != "accepted"
            for boundary in boundaries
        ):
            raise ValueError("accepted documents require accepted Boundary States")
        accepted_units = sum(
            unit["annotation"]["record"]["quality_status"] == "accepted"
            for unit in units
        )
        if accepted_units / len(units) < MIN_ACCEPTED_UNIT_RATIO:
            raise ValueError(
                "accepted documents require at least 90% accepted Evidence Units"
            )
    if quality_status == "quarantined" and value["status"] not in {
        "annotated",
        "mock-annotated",
        "partially-annotated",
    }:
        raise ValueError("only complete documents can be quarantined")
    return value


def boundary_state_by_id(
    document: Mapping[str, Any],
    boundary_id: str,
) -> Mapping[str, Any]:
    matches = [
        boundary
        for boundary in document.get("boundary_states", [])
        if isinstance(boundary, Mapping) and boundary.get("boundary_id") == boundary_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"document must contain exactly one Boundary State {boundary_id!r}"
        )
    return matches[0]


def unit_boundary_states(
    document: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return (
        boundary_state_by_id(document, unit["before_boundary_id"]),
        boundary_state_by_id(document, unit["after_boundary_id"]),
    )


def media_timestamp(document: dict[str, Any], source_frame: int) -> float:
    source = document["source"]
    return float(source.get("video_from_timestamp", 0.0)) + source_frame / int(
        source["fps"]
    )
