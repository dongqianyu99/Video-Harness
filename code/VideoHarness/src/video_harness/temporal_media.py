from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .camera_contract import CAMERA_VIEWS, image_label
from .hdf5_source import HDF5_SOURCE_DATASET, decode_hdf5_frames
from .media import FrameDecodeError
from .protocol import ImagePayload
from .sampling import unit_boundary_states

VIEWS = CAMERA_VIEWS
VIEW_TO_DATASET_KEY = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
KEYFRAME_INDICES = (0, 5, 10, 15, 20, 25)
EVIDENCE_UNIT_FRAME_COUNT = 26
DEFAULT_FFMPEG_TIMEOUT_S = 120.0


class TemporalMediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoSource:
    path: Path
    from_timestamp: float

    def __post_init__(self) -> None:
        if self.from_timestamp < 0:
            raise ValueError("video source timestamp must be non-negative")


@dataclass(frozen=True)
class EvidenceUnitFrames:
    frames: Mapping[str, np.ndarray]
    fps: int
    episode_start_frame: int
    episode_end_frame: int

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        frame_count = self.episode_end_frame - self.episode_start_frame + 1
        if not 2 <= frame_count <= EVIDENCE_UNIT_FRAME_COUNT:
            raise ValueError(
                "Evidence Unit must contain 2..26 consecutive episode frames"
            )
        if set(self.frames) != set(VIEWS):
            raise ValueError(f"frames must contain exactly {VIEWS}")
        shapes: set[tuple[int, ...]] = set()
        for view, array in self.frames.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{view} frames must be a numpy array")
            if array.ndim != 4 or array.shape[0] != frame_count or array.shape[-1] != 3:
                raise ValueError(
                    f"{view} must have shape [{frame_count},H,W,3], got {array.shape}"
                )
            if array.dtype != np.uint8:
                raise ValueError(f"{view} frames must be uint8 RGB")
            shapes.add(tuple(array.shape[1:]))
        if len(shapes) != 1:
            raise ValueError("all three views must share one H,W,3 shape")

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(self.frames["cam_high"].shape[1:])  # type: ignore[return-value]

    @property
    def frame_count(self) -> int:
        return self.episode_end_frame - self.episode_start_frame + 1


@dataclass(frozen=True)
class DetailRequest:
    roi: tuple[float, float, float, float]
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        x_min, y_min, x_max, y_max = self.roi
        if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
            raise ValueError("ROI must be normalized and non-empty")
        if not 0 <= self.start_frame <= self.end_frame < EVIDENCE_UNIT_FRAME_COUNT:
            raise ValueError("detail frame range must be within [0,25]")

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start_frame, self.end_frame + 1))


@dataclass(frozen=True)
class BaseMedia:
    unit_frames: EvidenceUnitFrames
    overviews: tuple[ImagePayload, ...]
    keyframe_sheets: tuple[ImagePayload, ...]
    boundary_images: tuple[ImagePayload, ...]


Runner = Callable[..., Any]


def _run_process(
    command: list[str],
    *,
    runner: Runner,
    input_bytes: bytes | None = None,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> bytes:
    kwargs: dict[str, Any] = {"check": False, "capture_output": True}
    if input_bytes is not None:
        kwargs["input"] = input_bytes
    try:
        process = runner(command, timeout=timeout_s, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise TemporalMediaError(f"FFmpeg timed out after {timeout_s:g}s") from exc
    payload = getattr(process, "stdout", b"")
    if getattr(process, "returncode", 1) != 0:
        stderr = getattr(process, "stderr", b"")
        message = bytes(stderr).decode("utf-8", errors="replace").strip()
        raise TemporalMediaError(message or "FFmpeg command failed")
    return bytes(payload)


def decode_unit_frames(
    sources: Mapping[str, VideoSource],
    *,
    episode_start_frame: int,
    episode_end_frame: int,
    fps: int = 25,
    image_shape: tuple[int, int, int] = (480, 640, 3),
    ffmpeg: str = "ffmpeg",
    runner: Runner = subprocess.run,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> EvidenceUnitFrames:
    if set(sources) != set(VIEWS):
        raise ValueError(f"sources must contain exactly {VIEWS}")
    frame_count = episode_end_frame - episode_start_frame + 1
    if not 2 <= frame_count <= EVIDENCE_UNIT_FRAME_COUNT:
        raise ValueError("Evidence Unit must contain 2..26 consecutive episode frames")
    height, width, channels = image_shape
    if min(height, width) <= 0 or channels != 3:
        raise ValueError("image_shape must be positive H,W,3")
    frame_bytes = height * width * channels
    expected_bytes = frame_count * frame_bytes
    decoded: dict[str, np.ndarray] = {}
    for view in VIEWS:
        source = sources[view]
        source_start_float = source.from_timestamp * fps
        source_start = round(source_start_float)
        if abs(source_start_float - source_start) > 1e-4:
            raise ValueError(
                f"{view} video_from_timestamp is not aligned to the {fps} Hz frame grid"
            )
        first_video_frame = source_start + episode_start_frame
        last_video_frame = first_video_frame + frame_count - 1
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source.path),
            "-vf",
            f"select=between(n\\,{first_video_frame}\\,{last_video_frame})",
            "-vsync",
            "0",
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        payload = _run_process(command, runner=runner, timeout_s=timeout_s)
        if len(payload) != expected_bytes:
            raise FrameDecodeError(
                f"{view} decoded {len(payload)} bytes for exact video frames "
                f"{first_video_frame}..{last_video_frame}; expected {expected_bytes}"
            )
        array = np.frombuffer(payload, dtype=np.uint8).reshape(
            frame_count,
            height,
            width,
            channels,
        )
        decoded[view] = np.array(array, copy=True)
    return EvidenceUnitFrames(
        frames=decoded,
        fps=fps,
        episode_start_frame=episode_start_frame,
        episode_end_frame=episode_end_frame,
    )


def _ffmpeg_tile(
    frames: Sequence[np.ndarray],
    *,
    columns: int,
    cell_size: tuple[int, int],
    ffmpeg: str,
    runner: Runner,
    timeout_s: float,
) -> bytes:
    if not frames or columns <= 0:
        raise ValueError("tile requires frames and positive columns")
    source_shape = frames[0].shape
    if any(frame.shape != source_shape or frame.dtype != np.uint8 for frame in frames):
        raise ValueError("tile frames must share uint8 H,W,3 shape")
    rows = (len(frames) + columns - 1) // columns
    padded = list(frames)
    black = np.zeros(source_shape, dtype=np.uint8)
    padded.extend(black for _ in range(columns * rows - len(padded)))
    source_height, source_width, _ = source_shape
    cell_width, cell_height = cell_size
    filter_graph = (
        f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
        f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"tile={columns}x{rows}:padding=2:margin=2:color=black"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{source_width}x{source_height}",
        "-framerate",
        "25",
        "-i",
        "pipe:0",
        "-vf",
        filter_graph,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    payload = _run_process(
        command,
        runner=runner,
        input_bytes=np.stack(padded).tobytes(),
        timeout_s=timeout_s,
    )
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise TemporalMediaError("FFmpeg tile renderer did not return PNG")
    return payload


def _label_tiled_png(
    png: bytes,
    *,
    labels: Sequence[str],
    columns: int,
    cell_size: tuple[int, int],
) -> bytes:
    image = Image.open(BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    cell_width, cell_height = cell_size
    margin = 2
    padding = 2
    for index, label in enumerate(labels):
        x = margin + (index % columns) * (cell_width + padding)
        y = margin + (index // columns) * (cell_height + padding)
        draw.rectangle((x, y, x + cell_width, y + 22), fill="black")
        draw.text((x + 4, y + 4), label, fill="white")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def render_sheet(
    frames: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    columns: int,
    cell_size: tuple[int, int],
    ffmpeg: str = "ffmpeg",
    runner: Runner = subprocess.run,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> bytes:
    if len(frames) != len(labels):
        raise ValueError("sheet labels must match frame count")
    tiled = _ffmpeg_tile(
        frames,
        columns=columns,
        cell_size=cell_size,
        ffmpeg=ffmpeg,
        runner=runner,
        timeout_s=timeout_s,
    )
    return _label_tiled_png(
        tiled,
        labels=labels,
        columns=columns,
        cell_size=cell_size,
    )


def _frame_label(unit: EvidenceUnitFrames, view: str, unit_frame: int) -> str:
    episode_frame = unit.episode_start_frame + unit_frame
    return (
        f"{view} | u={unit_frame:02d} | e={episode_frame} | "
        f"t=+{unit_frame / unit.fps:.2f}s"
    )


def overview_payload(
    unit: EvidenceUnitFrames,
    view: str,
    *,
    ffmpeg: str = "ffmpeg",
    runner: Runner = subprocess.run,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> ImagePayload:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    indices = tuple(range(min(25, max(1, unit.frame_count - 1))))
    data = render_sheet(
        [unit.frames[view][index] for index in indices],
        labels=[_frame_label(unit, view, index) for index in indices],
        columns=5,
        cell_size=(320, 240),
        ffmpeg=ffmpeg,
        runner=runner,
        timeout_s=timeout_s,
    )
    return ImagePayload(
        label=image_label(
            evidence_role="OVERVIEW",
            view=view,
            metadata=(
                f"UNIT_FRAMES={indices[0]}..{indices[-1]} | "
                "ORDER=CHRONOLOGICAL_ROW_MAJOR"
            ),
        ),
        data=data,
        media_type="image/png",
    )


def keyframe_sheet_payload(
    unit: EvidenceUnitFrames,
    view: str,
    *,
    ffmpeg: str = "ffmpeg",
    runner: Runner = subprocess.run,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> ImagePayload:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    if unit.frame_count == EVIDENCE_UNIT_FRAME_COUNT:
        indices = KEYFRAME_INDICES
    else:
        indices = tuple(
            sorted({round(value) for value in np.linspace(0, unit.frame_count - 1, 6)})
        )
    data = render_sheet(
        [unit.frames[view][index] for index in indices],
        labels=[_frame_label(unit, view, index) for index in indices],
        columns=3,
        cell_size=(512, 384),
        ffmpeg=ffmpeg,
        runner=runner,
        timeout_s=timeout_s,
    )
    return ImagePayload(
        label=image_label(
            evidence_role="KEYFRAME_SHEET",
            view=view,
            metadata=(
                "UNIT_FRAMES="
                + ",".join(str(index) for index in indices)
                + " | ORDER=CHRONOLOGICAL_ROW_MAJOR"
            ),
        ),
        data=data,
        media_type="image/png",
    )


def _jpeg(frame: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=95)
    return output.getvalue()


def boundary_image_payloads(unit: EvidenceUnitFrames) -> tuple[ImagePayload, ...]:
    payloads: list[ImagePayload] = []
    for role, unit_frame in (("BEFORE", 0), ("AFTER", unit.frame_count - 1)):
        episode_frame = unit.episode_start_frame + unit_frame
        for view in VIEWS:
            payloads.append(
                ImagePayload(
                    label=image_label(
                        evidence_role=f"BOUNDARY_{role}",
                        view=view,
                        metadata=(
                            f"UNIT_FRAME={unit_frame} | EPISODE_FRAME={episode_frame}"
                        ),
                    ),
                    data=_jpeg(unit.frames[view][unit_frame]),
                    media_type="image/jpeg",
                )
            )
    return tuple(payloads)


def validate_detail_request(
    inspection: Mapping[str, Any],
    *,
    max_frame: int = 25,
    context_margin: float = 0.15,
    min_area: float = 0.02,
    max_area: float = 0.60,
) -> DetailRequest | None:
    if not inspection.get("needs_detail"):
        return None
    detail = inspection.get("detail_request")
    window = inspection.get("interaction_window")
    if not isinstance(detail, Mapping) or not isinstance(window, Mapping):
        raise TypeError("inspection detail request/window is missing")
    roi = tuple(float(detail[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    x_min, y_min, x_max, y_max = roi
    area = (x_max - x_min) * (y_max - y_min)
    if not min_area <= area <= max_area:
        raise ValueError(f"detail ROI area {area:.4f} is outside accepted bounds")
    width = x_max - x_min
    height = y_max - y_min
    expanded = (
        max(0.0, x_min - width * context_margin),
        max(0.0, y_min - height * context_margin),
        min(1.0, x_max + width * context_margin),
        min(1.0, y_max + height * context_margin),
    )
    start_frame = int(window["start_frame"])
    end_frame = int(window["end_frame"])
    if not 0 <= start_frame <= end_frame <= max_frame:
        raise ValueError(
            f"detail frame range must be within [0,{max_frame}], got "
            f"[{start_frame},{end_frame}]"
        )
    return DetailRequest(
        roi=expanded,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def detail_payload(
    unit: EvidenceUnitFrames,
    request: DetailRequest,
    *,
    ffmpeg: str = "ffmpeg",
    runner: Runner = subprocess.run,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> ImagePayload:
    if request.end_frame >= unit.frame_count:
        raise ValueError("detail request exceeds decoded Evidence Unit frame count")
    height, width, _ = unit.image_shape
    x_min, y_min, x_max, y_max = request.roi
    left = max(0, min(width - 1, round(x_min * width)))
    top = max(0, min(height - 1, round(y_min * height)))
    right = max(left + 1, min(width, round(x_max * width)))
    bottom = max(top + 1, min(height, round(y_max * height)))
    frames = [
        unit.frames["cam_high"][index][top:bottom, left:right]
        for index in request.indices
    ]
    data = render_sheet(
        frames,
        labels=[_frame_label(unit, "cam_high", index) for index in request.indices],
        columns=5,
        cell_size=(384, 384),
        ffmpeg=ffmpeg,
        runner=runner,
        timeout_s=timeout_s,
    )
    return ImagePayload(
        label=image_label(
            evidence_role="DETAIL",
            view="cam_high",
            metadata=(
                f"UNIT_FRAMES={request.start_frame}..{request.end_frame} | "
                "ROI=FIXED | ORDER=CHRONOLOGICAL_ROW_MAJOR"
            ),
        ),
        data=data,
        media_type="image/png",
    )


def encode_debug_unit_video(
    frames: np.ndarray,
    *,
    fps: int,
    ffmpeg: str = "ffmpeg",
    runner: Runner = subprocess.run,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> bytes:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("debug video frames must be uint8 [N,H,W,3]")
    _, height, width, _ = frames.shape
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "pipe:1",
    ]
    payload = _run_process(
        command,
        runner=runner,
        input_bytes=frames.tobytes(),
        timeout_s=timeout_s,
    )
    if not payload:
        raise TemporalMediaError("FFmpeg debug video encoder returned no bytes")
    return payload


class TemporalMediaBuilder:
    def __init__(
        self,
        dataset_root: Path,
        *,
        image_shape: tuple[int, int, int] = (480, 640, 3),
        ffmpeg: str = "ffmpeg",
        runner: Runner = subprocess.run,
        timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.dataset_root = Path(dataset_root)
        self.image_shape = image_shape
        self.ffmpeg = ffmpeg
        self.runner = runner
        self.timeout_s = float(timeout_s)

    def _sources(self, document: Mapping[str, Any]) -> dict[str, VideoSource]:
        source = document["source"]
        views = source.get("views")
        if not isinstance(views, Mapping):
            raise TypeError("behavior document source has no multiview source map")
        result: dict[str, VideoSource] = {}
        for view in VIEWS:
            item = views.get(view)
            if not isinstance(item, Mapping):
                raise TypeError(f"behavior document is missing source view {view}")
            relative = Path(str(item["video_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source path for {view}: {relative}")
            result[view] = VideoSource(
                path=self.dataset_root / relative,
                from_timestamp=float(item["video_from_timestamp"]),
            )
        return result

    def build_base(
        self,
        document: Mapping[str, Any],
        unit: Mapping[str, Any],
    ) -> BaseMedia:
        before_boundary, after_boundary = unit_boundary_states(document, unit)
        start = int(before_boundary["frame"]["episode_frame_index"])
        end = int(after_boundary["frame"]["episode_frame_index"])
        source = document["source"]
        fps = int(source["fps"])
        if source["dataset"] == HDF5_SOURCE_DATASET:
            relative = Path(str(source["hdf5_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe HDF5 source path: {relative}")
            arrays = decode_hdf5_frames(
                self.dataset_root / relative,
                {view: str(source["views"][view]["dataset_key"]) for view in VIEWS},
                start=start,
                end=end,
                fps=fps,
            )
            frames = EvidenceUnitFrames(
                frames=arrays,
                fps=fps,
                episode_start_frame=start,
                episode_end_frame=end,
            )
        else:
            frames = decode_unit_frames(
                self._sources(document),
                episode_start_frame=start,
                episode_end_frame=end,
                fps=fps,
                image_shape=self.image_shape,
                ffmpeg=self.ffmpeg,
                runner=self.runner,
                timeout_s=self.timeout_s,
            )
        overviews = tuple(
            overview_payload(
                frames,
                view,
                ffmpeg=self.ffmpeg,
                runner=self.runner,
                timeout_s=self.timeout_s,
            )
            for view in VIEWS
        )
        keyframe_sheets = tuple(
            keyframe_sheet_payload(
                frames,
                view,
                ffmpeg=self.ffmpeg,
                runner=self.runner,
                timeout_s=self.timeout_s,
            )
            for view in VIEWS
        )
        return BaseMedia(
            unit_frames=frames,
            overviews=overviews,
            keyframe_sheets=keyframe_sheets,
            boundary_images=boundary_image_payloads(frames),
        )

    def build_detail(self, base: BaseMedia, request: DetailRequest) -> ImagePayload:
        return detail_payload(
            base.unit_frames,
            request,
            ffmpeg=self.ffmpeg,
            runner=self.runner,
            timeout_s=self.timeout_s,
        )

    def debug_media(
        self,
        base: BaseMedia,
        detail: ImagePayload | None,
    ) -> dict[str, bytes]:
        artifacts: dict[str, bytes] = {}
        for view in VIEWS:
            view_frames = base.unit_frames.frames[view]
            artifacts[f"videos/{view}.mp4"] = encode_debug_unit_video(
                view_frames,
                fps=base.unit_frames.fps,
                ffmpeg=self.ffmpeg,
                runner=self.runner,
                timeout_s=self.timeout_s,
            )
            for index, frame in enumerate(view_frames):
                artifacts[f"frames/{view}/frame-{index:02d}.jpg"] = _jpeg(frame)
        for payload in base.overviews:
            view = next(view for view in VIEWS if f"VIEW={view}" in payload.label)
            artifacts[f"sheets/{view}-overview.png"] = payload.data
        for payload in base.keyframe_sheets:
            view = next(view for view in VIEWS if f"VIEW={view}" in payload.label)
            artifacts[f"sheets/{view}-keyframes.png"] = payload.data
        for payload in base.boundary_images:
            role = "before" if "EVIDENCE=BOUNDARY_BEFORE" in payload.label else "after"
            view = next(view for view in VIEWS if f"VIEW={view}" in payload.label)
            slug = f"{role}-{view}"
            artifacts[f"boundaries/{slug}.jpg"] = payload.data
        if detail is not None:
            artifacts["sheets/cam_high-detail.png"] = detail.data
        return artifacts
