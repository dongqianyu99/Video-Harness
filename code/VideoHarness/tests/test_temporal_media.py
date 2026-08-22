from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from video_harness.media import FrameDecodeError
from video_harness.temporal_media import (
    STAGE_INDICES,
    UNIT_FRAME_COUNT,
    VIEWS,
    DetailRequest,
    TemporalMediaBuilder,
    UnitFrames,
    VideoSource,
    decode_unit_frames,
    detail_payload,
    endpoint_payloads,
    overview_payload,
    stage_payload,
    validate_detail_request,
)


def _unit(*, frame_count: int = UNIT_FRAME_COUNT) -> UnitFrames:
    frames: dict[str, np.ndarray] = {}
    for view_index, view in enumerate(VIEWS):
        values = np.empty((frame_count, 24, 32, 3), dtype=np.uint8)
        for frame_index in range(frame_count):
            values[frame_index, :, :, 0] = view_index * 60
            values[frame_index, :, :, 1] = frame_index
            values[frame_index, :, :, 2] = 255 - frame_index
        frames[view] = values
    return UnitFrames(
        frames=frames,
        fps=25,
        episode_start_frame=100,
        episode_end_frame=100 + frame_count - 1,
    )


def _open(payload: bytes) -> Image.Image:
    return Image.open(BytesIO(payload)).convert("RGB")


def test_decode_unit_frames_uses_each_view_offset_and_returns_owned_rgb() -> None:
    calls: list[list[str]] = []
    payloads = {
        view: np.full((26, 2, 3, 3), index + 1, dtype=np.uint8).tobytes()
        for index, view in enumerate(VIEWS)
    }

    def runner(command, **_kwargs):
        calls.append(command)
        path = Path(command[command.index("-i") + 1]).stem
        return SimpleNamespace(returncode=0, stdout=payloads[path], stderr=b"")

    sources = {
        view: VideoSource(Path(f"{view}.mp4"), from_timestamp=index * 10.0)
        for index, view in enumerate(VIEWS)
    }
    unit = decode_unit_frames(
        sources,
        episode_start_frame=25,
        episode_end_frame=50,
        fps=25,
        image_shape=(2, 3, 3),
        runner=runner,
    )

    assert len(calls) == 3
    assert [call[call.index("-vf") + 1] for call in calls] == [
        "select=between(n\\,25\\,50)",
        "select=between(n\\,275\\,300)",
        "select=between(n\\,525\\,550)",
    ]
    assert all(call[call.index("-frames:v") + 1] == "26" for call in calls)
    assert all(unit.frames[view].flags.owndata for view in VIEWS)
    assert [int(unit.frames[view][0, 0, 0, 0]) for view in VIEWS] == [1, 2, 3]


def test_decode_unit_frames_fails_closed_on_short_decode() -> None:
    def runner(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=b"too-short", stderr=b"")

    sources = {view: VideoSource(Path(f"{view}.mp4"), 0.0) for view in VIEWS}
    with pytest.raises(FrameDecodeError, match="expected"):
        decode_unit_frames(
            sources,
            episode_start_frame=0,
            episode_end_frame=25,
            image_shape=(2, 3, 3),
            runner=runner,
        )


def test_exact_decode_selects_source_frame_indices_with_real_ffmpeg(tmp_path: Path) -> None:
    frames = np.zeros((60, 24, 32, 3), dtype=np.uint8)
    for index in range(60):
        frames[index, :, :, 1] = index
    video = tmp_path / "source.mkv"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            "32x24",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-y",
            str(video),
        ],
        input=frames.tobytes(),
        capture_output=True,
        check=False,
    )
    assert encoded.returncode == 0, encoded.stderr.decode(errors="replace")

    sources = {view: VideoSource(video, from_timestamp=1.0) for view in VIEWS}
    unit = decode_unit_frames(
        sources,
        episode_start_frame=5,
        episode_end_frame=30,
        fps=25,
        image_shape=(24, 32, 3),
    )

    assert unit.frame_count == 26
    assert unit.frames["cam_high"][:, 0, 0, 1].tolist() == list(range(30, 56))


def test_all_views_get_overview_and_stage_sheets() -> None:
    unit = _unit()
    overviews = [overview_payload(unit, view) for view in VIEWS]
    stages = [stage_payload(unit, view) for view in VIEWS]

    assert [f"EVIDENCE=OVERVIEW | VIEW={view}" in payload.label for view, payload in zip(VIEWS, overviews, strict=True)] == [True, True, True]
    assert "CAMERA_ROLE=FIXED_GLOBAL" in overviews[0].label
    assert "CAMERA_ROLE=MOVING_LOCAL_LEFT_WRIST" in overviews[1].label
    assert "CAMERA_ROLE=MOVING_LOCAL_RIGHT_WRIST" in overviews[2].label
    assert all(payload.media_type == "image/png" for payload in overviews)
    assert all(_open(payload.data).width > 1500 for payload in overviews)
    assert all(
        f"EVIDENCE=STAGE | VIEW={view}" in payload.label
        for view, payload in zip(VIEWS, stages, strict=True)
    )
    assert all(",".join(str(index) for index in STAGE_INDICES) in payload.label for payload in stages)
    assert all(_open(payload.data).width > 1500 for payload in stages)


def test_endpoints_are_six_labeled_jpegs_in_before_then_after_order() -> None:
    endpoints = endpoint_payloads(_unit())
    assert [payload.label.split(" | ", 1)[0] for payload in endpoints] == [
        "EVIDENCE=ENDPOINT_BEFORE",
        "EVIDENCE=ENDPOINT_BEFORE",
        "EVIDENCE=ENDPOINT_BEFORE",
        "EVIDENCE=ENDPOINT_AFTER",
        "EVIDENCE=ENDPOINT_AFTER",
        "EVIDENCE=ENDPOINT_AFTER",
    ]
    assert all(payload.media_type == "image/jpeg" for payload in endpoints)
    assert all(_open(payload.data).size == (32, 24) for payload in endpoints)


def test_detail_request_expands_valid_roi_and_uses_cam_high_only() -> None:
    inspection = {
        "needs_detail": True,
        "interaction_window": {"start_frame": 3, "end_frame": 9},
        "detail_request": {
            "x_min": 0.25,
            "y_min": 0.25,
            "x_max": 0.75,
            "y_max": 0.75,
            "reason": "gripper_object",
        },
    }
    request = validate_detail_request(inspection)
    assert request is not None
    assert request.start_frame == 3
    assert request.end_frame == 9
    assert request.roi == pytest.approx((0.175, 0.175, 0.825, 0.825))

    payload = detail_payload(_unit(), request)
    assert payload.label.startswith("EVIDENCE=DETAIL | VIEW=cam_high")
    assert payload.media_type == "image/png"
    assert _open(payload.data).width > 1900


def test_detail_request_is_optional_and_invalid_requests_fail_closed() -> None:
    assert validate_detail_request({"needs_detail": False}) is None
    with pytest.raises(ValueError, match="outside accepted bounds"):
        validate_detail_request(
            {
                "needs_detail": True,
                "interaction_window": {"start_frame": 0, "end_frame": 2},
                "detail_request": {
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 1.0,
                    "y_max": 1.0,
                },
            }
        )
    with pytest.raises(ValueError, match="within"):
        validate_detail_request(
            {
                "needs_detail": True,
                "interaction_window": {"start_frame": 0, "end_frame": 26},
                "detail_request": {
                    "x_min": 0.1,
                    "y_min": 0.1,
                    "x_max": 0.4,
                    "y_max": 0.4,
                },
            }
        )
    with pytest.raises(ValueError, match="normalized"):
        DetailRequest((0.5, 0.5, 0.4, 0.7), 0, 2)


def test_partial_final_unit_still_produces_chronological_sheets() -> None:
    unit = _unit(frame_count=8)
    overview = overview_payload(unit, "cam_high")
    stage = stage_payload(unit, "cam_high")
    assert "UNIT_FRAMES=0..6" in overview.label
    assert stage.label.startswith("EVIDENCE=STAGE | VIEW=cam_high")
    assert _open(overview.data).width > 1500
    assert _open(stage.data).width > 1500


def test_unit_contract_requires_uint8_arrays_for_exact_three_views() -> None:
    unit = _unit()
    with pytest.raises(ValueError, match="exactly"):
        UnitFrames(
            frames={"cam_high": unit.frames["cam_high"]},
            fps=25,
            episode_start_frame=0,
            episode_end_frame=25,
        )
    wrong = dict(unit.frames)
    wrong["cam_high"] = wrong["cam_high"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        UnitFrames(
            frames=wrong,
            fps=25,
            episode_start_frame=0,
            episode_end_frame=25,
        )


def test_debug_media_contains_only_expected_multiview_artifacts() -> None:
    unit = _unit()
    base = SimpleNamespace(
        unit_frames=unit,
        overviews=tuple(overview_payload(unit, view) for view in VIEWS),
        stages=tuple(stage_payload(unit, view) for view in VIEWS),
        endpoints=endpoint_payloads(unit),
    )
    builder = TemporalMediaBuilder(Path("/unused"), image_shape=(24, 32, 3))
    detail = detail_payload(unit, DetailRequest((0.1, 0.1, 0.5, 0.5), 2, 5))
    artifacts = builder.debug_media(base, detail)

    assert {f"videos/{view}.mp4" for view in VIEWS} <= set(artifacts)
    assert {f"sheets/{view}-overview.png" for view in VIEWS} <= set(artifacts)
    assert {f"sheets/{view}-stage.png" for view in VIEWS} <= set(artifacts)
    assert "sheets/cam_high-detail.png" in artifacts
    assert len([name for name in artifacts if name.startswith("frames/")]) == 78
    assert len([name for name in artifacts if name.startswith("endpoints/")]) == 6
    assert not any("wrist-detail" in name for name in artifacts)
