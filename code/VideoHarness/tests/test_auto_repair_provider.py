from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from video_harness.annotations import (
    AnnotationError,
    MockRepairBackend,
    OpenAIBackend,
    RepairRequest,
    SequenceAuditRequest,
)
from video_harness.camera_contract import image_label
from video_harness.gripper_state import GripperState
from video_harness.protocol import ImagePayload


def _image(role: str, view: str) -> ImagePayload:
    return ImagePayload(
        image_label(evidence_role=role, view=view, metadata="TEST=true"),
        b"image",
        "image/jpeg",
    )


def _repair_request(call2: dict) -> RepairRequest:
    views = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    return RepairRequest(
        document_id="doc",
        unit_id="u0001",
        task_instruction="Perform the task.",
        issue_reason="The transition is inconsistent.",
        call1_motion_summary="The object moves with the gripper.",
        call2=call2,
        boundary_context='{"previous_transition": null}',
        overviews=tuple(_image("OVERVIEW", view) for view in views),
        keyframe_sheets=tuple(_image("KEYFRAME_SHEET", view) for view in views),
        boundary_images=tuple(
            _image(f"BOUNDARY_{role}", view)
            for role in ("BEFORE", "AFTER")
            for view in views
        ),
        gripper_state=GripperState(
            unit_frame_indices=(0, 5, 10, 15, 20, 25),
            left=(1.0,) * 6,
            right=(1.0, 0.8, 0.4, 0.4, 0.8, 1.0),
        ),
        detail=_image("DETAIL", "cam_high"),
    )


def test_openai_targeted_repair_uses_full_temporal_evidence(call2_record) -> None:
    payload = {
        "evidence_sufficient": True,
        "reason": "The full temporal evidence resolves the issue.",
        "resolved_call2": call2_record,
    }

    class Responses:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="resolve_transition_repair",
                        arguments=json.dumps(payload),
                    )
                ]
            )

    responses = Responses()
    result = OpenAIBackend(
        "repair-model",
        client=SimpleNamespace(responses=responses),
    ).repair(_repair_request(call2_record))

    content = responses.kwargs["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 13
    assert result.repair["evidence_sufficient"] is True
    assert result.repair["resolved_call2"]["causal_validation"]["status"] == "pass"


def test_repair_rejects_sufficient_claim_without_resolved_output(call2_record) -> None:
    invalid = {
        "evidence_sufficient": True,
        "reason": "claimed sufficient",
        "resolved_call2": None,
    }
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="resolve_transition_repair",
                        arguments=json.dumps(invalid),
                    )
                ]
            )
        )
    )
    with pytest.raises(AnnotationError, match="requires resolved"):
        OpenAIBackend("repair-model", client=client).repair(
            _repair_request(call2_record)
        )


def test_sequence_audit_deduplicates_unit_issues() -> None:
    audit = {
        "issues": [
            {"target_type": "unit", "target_id": "u0001", "reason": "first"},
            {
                "target_type": "unit",
                "target_id": "u0001",
                "reason": "duplicate",
            },
        ]
    }

    class Responses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="audit_sequence_consistency",
                        arguments=json.dumps(audit),
                    )
                ]
            )

    result = OpenAIBackend(
        "audit-model",
        client=SimpleNamespace(responses=Responses()),
    ).audit_sequence(
        SequenceAuditRequest(
            canonical_sequence='{"evidence_units": []}',
            task_instruction="Perform the task.",
        )
    )
    assert result.audit == {
        "issues": [
            {"target_type": "unit", "target_id": "u0001", "reason": "first"}
        ]
    }


def test_mock_repair_is_automatically_insufficient(call2_record) -> None:
    result = MockRepairBackend().repair(_repair_request(call2_record))
    assert result.repair["evidence_sufficient"] is False
    assert result.repair["resolved_call2"] is None
