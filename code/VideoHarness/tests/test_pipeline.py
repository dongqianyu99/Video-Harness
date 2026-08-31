from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from _support import boundary_observation

from video_harness.annotations import (
    AnnotationError,
    EvidenceResult,
    InspectionResult,
    RepairResult,
)
from video_harness.camera_contract import image_label
from video_harness.config import HarnessConfig
from video_harness.evidence import compose_evidence_record, mock_call2_record
from video_harness.gripper_state import GripperState
from video_harness.pipeline import EvidenceUnitPipeline
from video_harness.protocol import ImagePayload
from video_harness.temporal_media import (
    VIEWS,
    BaseMedia,
    DetailRequest,
    EvidenceUnitFrames,
    TemporalMediaError,
)


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
    unit = EvidenceUnitFrames(
        frames=frames,
        fps=25,
        episode_start_frame=0,
        episode_end_frame=25,
    )
    return BaseMedia(
        unit_frames=unit,
        overviews=tuple(_camera_payload("OVERVIEW", view) for view in VIEWS),
        keyframe_sheets=tuple(
            _camera_payload("KEYFRAME_SHEET", view) for view in VIEWS
        ),
        boundary_images=tuple(
            _camera_payload(f"BOUNDARY_{role}", view, "image/jpeg")
            for role in ("BEFORE", "AFTER")
            for view in VIEWS
        ),
        gripper_state=GripperState(
            unit_frame_indices=(0, 5, 10, 15, 20, 25),
            left=(1.0,) * 6,
            right=(1.0, 0.8, 0.4, 0.4, 0.8, 1.0),
        ),
    )


def _call2(
    *,
    action: str = "The robot approaches the object.",
    include_before_boundary: bool = True,
    include_after_boundary: bool = True,
) -> dict:
    record = mock_call2_record(
        include_before_boundary=include_before_boundary,
        include_after_boundary=include_after_boundary,
    )
    record["motion_summary"] = "The revised motion is grounded in Call 2 evidence."
    record["detail_observation"] = (
        "The detail sheet shows the gripper closing near the object."
    )
    record["unit_interpretation"] = {
        "action_description": action,
        "task_role": "This Evidence Unit prepares the next task interaction.",
    }
    return record


def _document_and_unit() -> tuple[dict, dict]:
    unit = {
        "unit_id": "u0000",
        "order": 0,
        "before_boundary_id": "b0000",
        "after_boundary_id": "b0001",
    }
    document = {
        "document_id": "robodojo/episode-0000000",
        "task_instruction": "Put bread into the toaster.",
        "boundary_states": [
            {
                "boundary_id": "b0000",
                "order": 0,
                "frame": {"episode_frame_index": 0, "timestamp_s": 0.0},
                "annotation": {"status": "pending", "record": None},
            },
            {
                "boundary_id": "b0001",
                "order": 1,
                "frame": {"episode_frame_index": 25, "timestamp_s": 1.0},
                "annotation": {"status": "pending", "record": None},
            },
        ],
        "evidence_units": [unit],
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
            "sheets/cam_high-keyframes.png": b"keyframes",
            "sheets/cam_left_wrist-keyframes.png": b"keyframes",
            "sheets/cam_right_wrist-keyframes.png": b"keyframes",
        }
        if detail is not None:
            artifacts["sheets/cam_high-detail.png"] = b"detail"
        return artifacts


class FakeInspectionBackend:
    provider = "test"
    model = "inspection"

    def __init__(self, *, failures: int = 0) -> None:
        self.remaining_failures = failures
        self.calls = 0
        self.requests = []

    def inspect(self, request) -> InspectionResult:
        self.calls += 1
        self.requests.append(request)
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise AnnotationError("transient inspection failure")
        assert len(request.overviews) == len(request.keyframe_sheets) == 3
        return InspectionResult(
            inspection={
                "motion_summary": "The right gripper approaches a visible object.",
                "interaction_window": {"start_frame": 4, "end_frame": 10},
                "detail_request": {
                    "x_min": 0.2,
                    "y_min": 0.2,
                    "x_max": 0.5,
                    "y_max": 0.5,
                },
            },
            provider=self.provider,
            requested_model=self.model,
        )


class FakeEvidenceBackend:
    provider = "test"
    model = "evidence"

    def __init__(self, outcomes: list[dict | Exception] | None = None) -> None:
        self.outcomes = list(outcomes or [_call2()])
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


class FakeRepairBackend:
    provider = "test-repair"
    model = "repair"

    def __init__(self, resolved_call2: dict) -> None:
        self.resolved_call2 = resolved_call2

    def repair(self, request) -> RepairResult:
        return RepairResult(
            repair={
                "evidence_sufficient": True,
                "reason": request.issue_reason,
                "resolved_call2": copy.deepcopy(self.resolved_call2),
            },
            provider=self.provider,
            requested_model=self.model,
        )

    def audit_sequence(self, request):  # pragma: no cover
        raise AssertionError(request)


def test_pipeline_passes_call1_motion_boundaries_and_detail_to_call2() -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    revised_call2 = _call2()
    evidence = FakeEvidenceBackend([revised_call2])
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=evidence,
        media_builder=media,
        config=HarnessConfig(),
    ).run(document, unit)

    request = evidence.requests[0]
    assert result.detail_status == "requested"
    assert len(media.detail_requests) == 1
    assert request.detail is not None
    assert len(request.boundary_images) == 6
    assert request.motion_summary == result.inspection.inspection["motion_summary"]
    assert not hasattr(request, "overviews")
    assert not hasattr(request, "keyframe_sheets")
    assert result.evidence.evidence["motion_summary"] == revised_call2["motion_summary"]
    assert result.before_boundary_record is not None
    assert result.after_boundary_record is not None


def test_pipeline_reuses_shared_boundary_without_describing_it_again() -> None:
    document, unit = _document_and_unit()
    document["boundary_states"][0]["annotation"] = {
        "status": "complete",
        "record": {
            "observation": boundary_observation("Shared before"),
        },
    }
    evidence = FakeEvidenceBackend([_call2(include_before_boundary=False)])
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, unit)

    assert evidence.requests[0].before_boundary_observation == boundary_observation(
        "Shared before"
    )
    assert result.before_boundary_record is None
    assert result.after_boundary_record is not None


def test_pipeline_resume_reuses_both_boundaries() -> None:
    document, unit = _document_and_unit()
    for order, boundary in enumerate(document["boundary_states"]):
        boundary["annotation"] = {
            "status": "complete",
            "record": {
                "observation": boundary_observation(f"Boundary {order}"),
            },
        }
    evidence = FakeEvidenceBackend(
        [
            _call2(
                include_before_boundary=False,
                include_after_boundary=False,
            )
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, unit)

    request = evidence.requests[0]
    assert request.before_boundary_observation == boundary_observation("Boundary 0")
    assert request.after_boundary_observation == boundary_observation("Boundary 1")
    assert result.before_boundary_record is None
    assert result.after_boundary_record is None


def test_pipeline_passes_previous_compiled_task_blind_motion_context() -> None:
    previous_summary = (
        "The right gripper remains closed around a visible entity at the final frame."
    )
    previous = {
        "unit_id": "u0000",
        "order": 0,
        "annotation": {
            "status": "complete",
            "record": {
                "motion_summary": previous_summary,
            },
            "provenance": {"call1": {"provider": "test"}},
        },
    }
    current = {
        "unit_id": "u0001",
        "order": 1,
        "before_boundary_id": "b0001",
        "after_boundary_id": "b0002",
    }
    document = {
        "document_id": "robodojo/episode-0000000",
        "task_instruction": "Perform the task.",
        "evidence_units": [previous, current],
        "boundary_states": [
            {
                "boundary_id": f"b{order:04d}",
                "frame": {
                    "episode_frame_index": order * 25,
                    "timestamp_s": float(order),
                },
                "annotation": {"status": "pending", "record": None},
            }
            for order in range(3)
        ],
    }
    inspection = FakeInspectionBackend()
    EvidenceUnitPipeline(
        inspection_backend=inspection,
        evidence_backend=FakeEvidenceBackend(),
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, current)

    assert inspection.requests[0].previous_motion_summary == previous_summary


def test_debug_mode_saves_compilation_outputs(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=FakeEvidenceBackend([_call2()]),
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(debug=True, debug_root=tmp_path),
    ).run(document, unit)

    root = Path(result.debug_root or "")
    assert (root / "call1.json").is_file()
    assert (root / "call2.json").is_file()
    assert (root / "final.json").is_file()
    final = json.loads((root / "final.json").read_text())
    assert "quality_status" not in final
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"


def test_call2_provider_error_fails_with_scoped_debug(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    with pytest.raises(AnnotationError, match="provider failure"):
        EvidenceUnitPipeline(
            inspection_backend=FakeInspectionBackend(),
            evidence_backend=FakeEvidenceBackend(
                [AnnotationError("provider failure") for _ in range(3)]
            ),
            media_builder=FakeMediaBuilder(),
            config=HarnessConfig(debug=True, debug_root=tmp_path),
        ).run(document, unit)

    root = next(tmp_path.rglob("u0000"))
    assert (root / "call2-attempt-01-error.json").is_file()
    assert (root / "call2-attempt-02-error.json").is_file()
    assert (root / "call2-attempt-03-error.json").is_file()
    assert (root / "call2-error.json").is_file()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_call2_retries_malformed_output_without_rebuilding_media(
    tmp_path: Path,
) -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    evidence = FakeEvidenceBackend(
        [AnnotationError("malformed evidence arguments"), _call2()]
    )

    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=evidence,
        media_builder=media,
        config=HarnessConfig(debug=True, debug_root=tmp_path),
    ).run(document, unit)

    root = Path(result.debug_root or "")
    assert len(evidence.requests) == 2
    assert media.base_calls == 1
    assert (root / "call2-attempt-01-error.json").is_file()
    assert (root / "call2.json").is_file()


def test_media_failure_retries_before_compiling_unit() -> None:
    class FlakyMedia(FakeMediaBuilder):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 2

        def build_base(self, document, unit):
            self.base_calls += 1
            if self.failures:
                self.failures -= 1
                raise TemporalMediaError("transient decode failure")
            return self.base

    document, unit = _document_and_unit()
    media = FlakyMedia()
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=FakeEvidenceBackend(),
        media_builder=media,
        config=HarnessConfig(media_retries=2),
    ).run(document, unit)

    assert media.base_calls == 3
    assert result.evidence.evidence["motion_summary"]


def test_inspection_failure_after_retries_is_technical(
    tmp_path: Path,
) -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    with pytest.raises(AnnotationError, match="Call 1 failed after retries"):
        EvidenceUnitPipeline(
            inspection_backend=FakeInspectionBackend(failures=2),
            evidence_backend=FakeEvidenceBackend(),
            media_builder=media,
            config=HarnessConfig(
                debug=True,
                debug_root=tmp_path,
                inspection_retries=1,
            ),
        ).run(document, unit)

    assert media.base_calls == 1


def _compiled_repair_target() -> tuple[dict, dict]:
    document, unit = _document_and_unit()
    unit["annotation"] = {
        "status": "complete",
        "record": compose_evidence_record(_call2()),
    }
    for order, boundary in enumerate(document["boundary_states"]):
        boundary["annotation"] = {
            "status": "complete",
            "record": {
                "observation": boundary_observation(f"Boundary {order}"),
            },
        }
    return document, unit


def test_targeted_repair_requires_declared_boundary_replacement() -> None:
    document, unit = _compiled_repair_target()
    pipeline = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=FakeEvidenceBackend(),
        repair_backend=FakeRepairBackend(
            _call2(include_before_boundary=False, include_after_boundary=False)
        ),
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(repair_max_attempts=1),
    )

    outcome = pipeline.repair_target(
        document,
        unit,
        issue_reason="Replace the final Boundary.",
        allowed_boundary_replacements=frozenset({"after"}),
        required_boundary_replacements=frozenset({"after"}),
    )

    assert outcome.canonical_evidence is None


def test_targeted_repair_returns_only_authorized_boundary() -> None:
    document, unit = _compiled_repair_target()
    pipeline = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(),
        evidence_backend=FakeEvidenceBackend(),
        repair_backend=FakeRepairBackend(
            _call2(include_before_boundary=False, include_after_boundary=True)
        ),
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(repair_max_attempts=1),
    )

    outcome = pipeline.repair_target(
        document,
        unit,
        issue_reason="Replace the final Boundary.",
        allowed_boundary_replacements=frozenset({"after"}),
        required_boundary_replacements=frozenset({"after"}),
    )

    assert outcome.canonical_evidence is not None
    assert outcome.before_boundary_record is None
    assert outcome.after_boundary_record is not None
