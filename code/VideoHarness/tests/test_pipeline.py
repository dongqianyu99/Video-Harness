from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from video_harness.annotations import AnnotationError, EvidenceResult, InspectionResult
from video_harness.camera_contract import image_label
from video_harness.config import HarnessConfig
from video_harness.evidence import mock_call2_record
from video_harness.pipeline import UnitPipeline
from video_harness.protocol import ImagePayload
from video_harness.temporal_media import BaseMedia, DetailRequest, UnitFrames, VIEWS


def _payload(label: str, media_type: str = "image/png") -> ImagePayload:
    return ImagePayload(label=label, data=b"payload", media_type=media_type)


def _camera_payload(
    role: str, view: str, media_type: str = "image/png"
) -> ImagePayload:
    return _payload(
        image_label(evidence_role=role, view=view, metadata="TEST_FIXTURE=true"),
        media_type,
    )


def _base() -> BaseMedia:
    frames = {view: np.zeros((26, 4, 6, 3), dtype=np.uint8) for view in VIEWS}
    unit = UnitFrames(
        frames=frames,
        fps=25,
        episode_start_frame=0,
        episode_end_frame=25,
    )
    return BaseMedia(
        unit_frames=unit,
        overviews=tuple(_camera_payload("OVERVIEW", view) for view in VIEWS),
        stages=tuple(_camera_payload("STAGE", view) for view in VIEWS),
        endpoints=tuple(
            _camera_payload(f"ENDPOINT_{role}", view, "image/jpeg")
            for role in ("BEFORE", "AFTER")
            for view in VIEWS
        ),
    )


def _call2(
    status: str,
    *,
    action: str = "The robot approaches the object.",
    detail: bool = False,
) -> dict:
    record = mock_call2_record()
    record["detail_observation"] = (
        "The detail sheet shows the gripper closing near the object."
        if detail
        else None
    )
    record["unit_interpretation"] = {
        "action_description": action,
        "task_role": "This Unit positions the gripper for the next task step.",
    }
    record["causal_validation"] = {
        "status": status,
        "reason": (
            "The interpretation is physically plausible."
            if status == "pass"
            else "The claimed object motion lacks a compatible interaction."
        ),
    }
    return record


def _document_and_unit() -> tuple[dict, dict]:
    unit = {
        "unit_id": "u0000",
        "before": {"episode_frame_index": 0, "timestamp_s": 0.0},
        "after": {"episode_frame_index": 25, "timestamp_s": 1.0},
    }
    document = {
        "document_id": "robodojo/episode-0000000",
        "task_instruction": "Put bread into the toaster.",
    }
    return document, unit


class FakeMediaBuilder:
    def __init__(self) -> None:
        self.base = _base()
        self.base_calls = 0
        self.detail_requests: list[DetailRequest] = []
        self.debug_calls = 0

    def build_base(self, _document, _unit) -> BaseMedia:
        self.base_calls += 1
        return self.base

    def build_detail(self, _base, request: DetailRequest) -> ImagePayload:
        self.detail_requests.append(request)
        return _camera_payload("DETAIL", "cam_high")

    def debug_media(self, _base, detail) -> dict[str, bytes]:
        self.debug_calls += 1
        artifacts = {
            "videos/cam_high.mp4": b"mp4",
            "frames/cam_high/frame-00.jpg": b"jpg",
            "sheets/cam_high-overview.png": b"png",
            "sheets/cam_high-stage.png": b"stage",
            "sheets/cam_left_wrist-stage.png": b"stage",
            "sheets/cam_right_wrist-stage.png": b"stage",
        }
        if detail is not None:
            artifacts["sheets/cam_high-detail.png"] = b"detail"
        return artifacts


class FakeInspectionBackend:
    provider = "test"
    model = "inspection"

    def __init__(self, *, needs_detail: bool, failures: int = 0) -> None:
        self.needs_detail = needs_detail
        self.remaining_failures = failures
        self.calls = 0

    def inspect(self, request) -> InspectionResult:
        self.calls += 1
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise AnnotationError("transient inspection failure")
        assert len(request.overviews) == len(request.stages) == 3
        detail = (
            {
                "x_min": 0.2,
                "y_min": 0.2,
                "x_max": 0.5,
                "y_max": 0.5,
                "reason": "gripper_object",
            }
            if self.needs_detail
            else None
        )
        return InspectionResult(
            inspection={
                "motion_summary": "The right gripper approaches a visible object.",
                "interaction_window": {"start_frame": 4, "end_frame": 10},
                "needs_detail": self.needs_detail,
                "detail_request": detail,
            },
            provider=self.provider,
            requested_model=self.model,
        )


class FakeEvidenceBackend:
    provider = "test"
    model = "evidence"

    def __init__(self, outcomes: list[dict | Exception] | None = None) -> None:
        self.outcomes = list(outcomes or [_call2("pass")])
        self.requests = []

    def annotate(self, request) -> EvidenceResult:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return EvidenceResult(
            evidence=copy.deepcopy(outcome),
            provider=self.provider,
            requested_model=self.model,
        )


def test_pipeline_passes_call1_motion_endpoints_and_optional_detail_to_call2() -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    evidence = FakeEvidenceBackend([_call2("pass", detail=True)])
    result = UnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=True),
        evidence_backend=evidence,
        media_builder=media,
        config=HarnessConfig(),
    ).run(document, unit)

    request = evidence.requests[0]
    assert result.detail_status == "requested"
    assert len(media.detail_requests) == 1
    assert request.detail is not None
    assert len(request.endpoints) == 6
    assert request.motion_summary == result.inspection.inspection["motion_summary"]
    assert not hasattr(request, "overviews")
    assert not hasattr(request, "stages")
    assert result.review_status == "accepted"
    assert result.evidence.evidence["motion_summary"] == request.motion_summary


def test_call2_retry_then_pass_reuses_call1_media_and_supplies_previous_output() -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    inspection = FakeInspectionBackend(needs_detail=False)
    first = _call2("retry", action="The object moves without visible interaction.")
    evidence = FakeEvidenceBackend([first, _call2("pass")])
    result = UnitPipeline(
        inspection_backend=inspection,
        evidence_backend=evidence,
        media_builder=media,
        config=HarnessConfig(call2_max_attempts=3),
    ).run(document, unit)

    assert result.call2_attempts == 2
    assert result.review_status == "accepted"
    assert inspection.calls == media.base_calls == 1
    assert len(evidence.requests) == 2
    assert evidence.requests[0].previous_attempt is None
    assert evidence.requests[1].previous_attempt == first


def test_call2_retries_when_detail_description_does_not_match_supplied_media() -> None:
    document, unit = _document_and_unit()
    evidence = FakeEvidenceBackend([_call2("pass"), _call2("pass", detail=True)])
    result = UnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=True),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(call2_max_attempts=3),
    ).run(document, unit)

    assert result.call2_attempts == 2
    assert result.review_status == "accepted"


def test_three_retry_results_keep_last_and_mark_needs_review() -> None:
    document, unit = _document_and_unit()
    evidence = FakeEvidenceBackend(
        [_call2("retry", action=f"Attempt {index} remains inconsistent.") for index in range(3)]
    )
    result = UnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(call2_max_attempts=3),
    ).run(document, unit)

    assert result.call2_attempts == 3
    assert result.review_status == "needs_review"
    assert result.evidence.evidence["review_status"] == "needs_review"
    assert result.evidence.evidence["unit_interpretation"]["action_description"].startswith("Attempt 2")


def test_debug_mode_saves_each_call2_attempt_and_final_selection(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    result = UnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=True),
        evidence_backend=FakeEvidenceBackend(
            [_call2("retry", detail=True), _call2("pass", detail=True)]
        ),
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(debug=True, debug_root=tmp_path),
    ).run(document, unit)

    root = Path(result.debug_root or "")
    assert (root / "call1.json").is_file()
    assert (root / "call2-attempt-01.json").is_file()
    assert (root / "call2-attempt-02.json").is_file()
    assert (root / "final.json").is_file()
    assert not (root / "call2.json").exists()
    final = json.loads((root / "final.json").read_text())
    assert final["selected_call2_attempt"] == 2
    assert final["review_status"] == "accepted"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["call2_attempts"] == 2
    assert manifest["selected_call2_attempt"] == 2


def test_all_call2_provider_errors_fail_with_attempt_scoped_debug(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    errors = [AnnotationError(f"provider failure {index}") for index in range(3)]
    with pytest.raises(AnnotationError, match="provider failure 2"):
        UnitPipeline(
            inspection_backend=FakeInspectionBackend(needs_detail=False),
            evidence_backend=FakeEvidenceBackend(errors),
            media_builder=FakeMediaBuilder(),
            config=HarnessConfig(
                debug=True,
                debug_root=tmp_path,
                call2_max_attempts=3,
            ),
        ).run(document, unit)

    root = next(tmp_path.rglob("u0000"))
    assert all(
        (root / f"call2-attempt-{attempt:02d}-error.json").is_file()
        for attempt in range(1, 4)
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["call2_attempts"] == 3


def test_inspection_retry_and_fallback_do_not_repeat_media_decode(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    result = UnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=True, failures=2),
        evidence_backend=FakeEvidenceBackend(),
        media_builder=media,
        config=HarnessConfig(
            debug=True,
            debug_root=tmp_path,
            inspection_retries=1,
        ),
    ).run(document, unit)

    assert media.base_calls == 1
    assert result.detail_status == "inspection-failed-omitted"
    assert result.inspection.provider == "harness-fallback"
    root = Path(result.debug_root or "")
    assert (root / "call1-error.json").is_file()
    assert not (root / "sheets/cam_high-detail.png").exists()
