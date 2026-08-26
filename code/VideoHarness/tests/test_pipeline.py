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
from video_harness.evidence import mock_call2_record
from video_harness.pipeline import EvidenceUnitPipeline
from video_harness.protocol import ImagePayload
from video_harness.temporal_media import (
    VIEWS,
    BaseMedia,
    DetailRequest,
    EvidenceUnitFrames,
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
    )


def _call2(
    status: str,
    *,
    action: str = "The robot approaches the object.",
    detail: bool = False,
    include_before_boundary: bool = True,
    include_after_boundary: bool = True,
    boundary_conflicts: dict[str, str | None] | None = None,
) -> dict:
    record = mock_call2_record(
        include_before_boundary=include_before_boundary,
        include_after_boundary=include_after_boundary,
    )
    record["motion_summary"] = "The revised motion is grounded in Call 2 evidence."
    record["detail_observation"] = (
        "The detail sheet shows the gripper closing near the object."
        if detail
        else None
    )
    record["unit_interpretation"] = {
        "action_description": action,
        "task_role": "This Evidence Unit prepares the next task interaction.",
    }
    record["causal_validation"] = {
        "status": status,
        "reason": (
            "The interpretation is physically plausible."
            if status == "pass"
            else "The claimed object motion lacks a compatible interaction."
        ),
    }
    record["boundary_conflicts"] = boundary_conflicts or {
        "before": None,
        "after": None,
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

    def __init__(self, *, needs_detail: bool, failures: int = 0) -> None:
        self.needs_detail = needs_detail
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
        detail = (
            {
                "x_min": 0.2,
                "y_min": 0.2,
                "x_max": 0.5,
                "y_max": 0.5,
                "reason": "fine_spatial_detail",
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


class FakeRepairBackend:
    provider = "test-repair"
    model = "repair"

    def __init__(self, repairs: list[dict]) -> None:
        self.repairs = list(repairs)
        self.requests = []

    def repair(self, request) -> RepairResult:
        self.requests.append(request)
        return RepairResult(
            repair=copy.deepcopy(self.repairs.pop(0)),
            provider=self.provider,
            requested_model=self.model,
        )

    def audit_sequence(self, request):  # pragma: no cover - pipeline does not audit
        raise AssertionError(request)


def test_pipeline_passes_call1_motion_boundaries_and_optional_detail_to_call2() -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    revised_call2 = _call2("pass", detail=True)
    evidence = FakeEvidenceBackend([revised_call2])
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=True),
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
    assert result.quality_status == "accepted"
    assert result.evidence.evidence["motion_summary"] == revised_call2["motion_summary"]
    assert result.before_boundary_record is not None
    assert result.after_boundary_record is not None


def test_pipeline_reuses_accepted_shared_boundary_without_describing_it_again() -> None:
    document, unit = _document_and_unit()
    document["boundary_states"][0]["annotation"] = {
        "status": "complete",
        "record": {
            "observation": boundary_observation("Shared before"),
            "quality_status": "accepted",
        },
    }
    evidence = FakeEvidenceBackend([_call2("pass", include_before_boundary=False)])
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, unit)

    assert evidence.requests[0].before_boundary_observation == boundary_observation(
        "Shared before"
    )
    assert result.before_boundary_record is None
    assert result.after_boundary_record is not None


def test_pipeline_resume_reuses_both_accepted_boundaries() -> None:
    document, unit = _document_and_unit()
    for order, boundary in enumerate(document["boundary_states"]):
        boundary["annotation"] = {
            "status": "complete",
            "record": {
                "observation": boundary_observation(f"Boundary {order}"),
                "quality_status": "accepted",
            },
        }
    evidence = FakeEvidenceBackend(
        [
            _call2(
                "pass",
                include_before_boundary=False,
                include_after_boundary=False,
            )
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, unit)

    request = evidence.requests[0]
    assert request.before_boundary_observation == boundary_observation("Boundary 0")
    assert request.after_boundary_observation == boundary_observation("Boundary 1")
    assert result.before_boundary_record is None
    assert result.after_boundary_record is None


def test_pipeline_flags_conflicting_shared_boundary_without_duplicating_it() -> None:
    document, unit = _document_and_unit()
    document["boundary_states"][0]["annotation"] = {
        "status": "complete",
        "record": {
            "observation": boundary_observation("Shared before"),
            "quality_status": "accepted",
        },
    }
    conflict = {
        "before": "The accepted description materially disagrees with the image.",
        "after": None,
    }
    evidence = FakeEvidenceBackend(
        [
            _call2(
                "retry",
                include_before_boundary=False,
                boundary_conflicts=conflict,
            )
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False),
        evidence_backend=evidence,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, unit)

    assert result.quality_status == "quarantined"
    assert result.before_boundary_record is None
    assert result.conflicted_boundary_roles == ("before",)


def test_pipeline_passes_only_accepted_previous_task_blind_motion_context() -> None:
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
                "quality_status": "accepted",
            },
            "provenance": {
                "call1": {"provider": "test"},
            },
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
    inspection = FakeInspectionBackend(needs_detail=False)
    EvidenceUnitPipeline(
        inspection_backend=inspection,
        evidence_backend=FakeEvidenceBackend(),
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(),
    ).run(document, current)

    assert inspection.requests[0].previous_motion_summary == previous_summary


def test_retry_enters_targeted_repair_and_commits_resolved_transition() -> None:
    document, unit = _document_and_unit()
    unresolved = _call2(
        "retry",
        action="The initial transition is inconsistent.",
    )
    resolved = _call2(
        "pass",
        action="The repaired transition is supported by the temporal evidence.",
    )
    resolved["motion_summary"] = "The corrected qualitative motion."
    repair = FakeRepairBackend(
        [
            {
                "evidence_sufficient": True,
                "reason": "The full temporal sheets resolve the inconsistency.",
                "resolved_call2": resolved,
            }
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False),
        evidence_backend=FakeEvidenceBackend([unresolved]),
        repair_backend=repair,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(repair_max_attempts=2),
    ).run(document, unit)

    assert result.quality_status == "accepted"
    assert result.repair_attempts == 1
    assert result.repair is not None
    assert (
        result.evidence.evidence["motion_summary"]
        == "The corrected qualitative motion."
    )
    assert result.evidence.evidence["unit_interpretation"][
        "action_description"
    ].startswith("The repaired transition")
    assert len(repair.requests[0].overviews) == 3
    assert len(repair.requests[0].keyframe_sheets) == 3
    assert len(repair.requests[0].boundary_images) == 6


def test_call2_rejects_detail_description_without_supplied_media() -> None:
    document, unit = _document_and_unit()
    with pytest.raises(AnnotationError, match="detail observation"):
        EvidenceUnitPipeline(
            inspection_backend=FakeInspectionBackend(needs_detail=True),
            evidence_backend=FakeEvidenceBackend([_call2("pass")]),
            media_builder=FakeMediaBuilder(),
            config=HarnessConfig(),
        ).run(document, unit)


def test_unresolved_targeted_repairs_quarantine_transition() -> None:
    document, unit = _document_and_unit()
    unresolved = _call2("retry", action="The transition remains inconsistent.")
    repair = FakeRepairBackend(
        [
            {
                "evidence_sufficient": False,
                "reason": "Evidence remains insufficient.",
                "resolved_call2": None,
            }
            for _ in range(2)
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False),
        evidence_backend=FakeEvidenceBackend([unresolved]),
        repair_backend=repair,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(repair_max_attempts=2),
    ).run(document, unit)

    assert result.quality_status == "quarantined"
    assert result.repair_attempts == 2
    assert result.evidence.evidence["quality_status"] == "quarantined"
    assert result.before_boundary_record["quality_status"] == "accepted"
    assert result.after_boundary_record["quality_status"] == "accepted"


def test_debug_mode_saves_call2_repair_and_final_selection(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    resolved = _call2("pass", detail=True)
    repair = FakeRepairBackend(
        [
            {
                "evidence_sufficient": True,
                "reason": "The temporal evidence resolves the issue.",
                "resolved_call2": resolved,
            }
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=True),
        evidence_backend=FakeEvidenceBackend([_call2("retry", detail=True)]),
        repair_backend=repair,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(debug=True, debug_root=tmp_path),
    ).run(document, unit)

    root = Path(result.debug_root or "")
    assert (root / "call1.json").is_file()
    assert (root / "call2.json").is_file()
    assert (root / "repair-attempt-01.json").is_file()
    assert (root / "final.json").is_file()
    final = json.loads((root / "final.json").read_text())
    assert final["quality_status"] == "accepted"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["repair_attempts"] == 1


def test_call2_provider_error_fails_with_scoped_debug(tmp_path: Path) -> None:
    document, unit = _document_and_unit()
    with pytest.raises(AnnotationError, match="provider failure"):
        EvidenceUnitPipeline(
            inspection_backend=FakeInspectionBackend(needs_detail=False),
            evidence_backend=FakeEvidenceBackend([AnnotationError("provider failure")]),
            media_builder=FakeMediaBuilder(),
            config=HarnessConfig(debug=True, debug_root=tmp_path),
        ).run(document, unit)

    root = next(tmp_path.rglob("u0000"))
    assert (root / "call2-error.json").is_file()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_inspection_retry_and_fallback_do_not_repeat_media_decode(
    tmp_path: Path,
) -> None:
    document, unit = _document_and_unit()
    media = FakeMediaBuilder()
    result = EvidenceUnitPipeline(
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
    assert result.quality_status == "quarantined"
    assert result.evidence.evidence["causal_validation"]["status"] == "retry"
    root = Path(result.debug_root or "")
    assert (root / "call1-error.json").is_file()
    assert not (root / "sheets/cam_high-detail.png").exists()


def test_inspection_failure_can_be_recovered_from_full_temporal_evidence() -> None:
    document, unit = _document_and_unit()
    resolved = _call2(
        "pass",
        action="The full temporal evidence resolves the demonstrated action.",
    )
    resolved["motion_summary"] = "Recovered task-conditioned motion evidence."
    repair = FakeRepairBackend(
        [
            {
                "evidence_sufficient": True,
                "reason": "The temporal sheets recover the missing motion evidence.",
                "resolved_call2": resolved,
            }
        ]
    )
    result = EvidenceUnitPipeline(
        inspection_backend=FakeInspectionBackend(needs_detail=False, failures=2),
        evidence_backend=FakeEvidenceBackend([_call2("pass")]),
        repair_backend=repair,
        media_builder=FakeMediaBuilder(),
        config=HarnessConfig(inspection_retries=1),
    ).run(document, unit)

    assert result.quality_status == "accepted"
    assert result.repair_attempts == 1
    assert result.evidence.evidence["motion_summary"] == (
        "Recovered task-conditioned motion evidence."
    )
