import json
from types import SimpleNamespace

import pytest

from video_harness.annotations import (
    AnnotationError,
    EvidenceRequest,
    ImagePayload,
    InspectionRequest,
    MockEvidenceBackend,
    MockInspectionBackend,
    OpenAIBackend,
)
from video_harness.camera_contract import image_label
from video_harness.prompts import (
    EVIDENCE_SCHEMA,
    INSPECTION_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_NAME,
)


def _image(label: str) -> ImagePayload:
    return ImagePayload(label=label, data=b"jpeg", media_type="image/jpeg")


def _camera_image(role: str, view: str) -> ImagePayload:
    return _image(
        image_label(evidence_role=role, view=view, metadata="TEST_FIXTURE=true")
    )


def _inspection_request() -> InspectionRequest:
    views = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    return InspectionRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=100,
        episode_end_frame=125,
        overviews=tuple(_camera_image("OVERVIEW", view) for view in views),
        stages=tuple(_camera_image("STAGE", view) for view in views),
    )


def _evidence_request(*, detail: bool = False, previous_attempt=None) -> EvidenceRequest:
    return EvidenceRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=100,
        episode_end_frame=125,
        motion_summary="The right gripper approaches the bread slice and closes around it.",
        task_instruction="Put bread into the toaster.",
        detail=_camera_image("DETAIL", "cam_high") if detail else None,
        endpoints=tuple(
            _camera_image(f"ENDPOINT_{role}", view)
            for role in ("BEFORE", "AFTER")
            for view in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ),
        previous_attempt=previous_attempt,
    )


def test_mock_backends_are_explicit_and_schema_valid() -> None:
    inspection = MockInspectionBackend().inspect(_inspection_request())
    evidence = MockEvidenceBackend().annotate(_evidence_request())
    assert inspection.inspection["needs_detail"] is False
    assert "active_end_effector" not in inspection.inspection
    assert evidence.evidence["causal_validation"]["status"] == "retry"
    assert evidence.prompt_version == PROMPT_VERSION


def test_prompt_and_schema_define_progressive_calls() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).lower()
    assert "task-blind motion_summary from call 1" in normalized
    assert "static state visible in every camera endpoint" in normalized
    assert "status=retry only for a clear violation" in normalized
    assert "active_end_effector" not in INSPECTION_SCHEMA["properties"]
    before = EVIDENCE_SCHEMA["properties"]["endpoint_observation"]["properties"]["before"]
    assert set(before["required"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }


def test_request_requires_three_overviews_three_stages_and_six_endpoints() -> None:
    inspection = _inspection_request()
    with pytest.raises(ValueError, match="three stage"):
        InspectionRequest(
            document_id="doc",
            unit_id="u0000",
            episode_start_frame=0,
            episode_end_frame=25,
            overviews=inspection.overviews,
            stages=inspection.stages[:2],
        )
    request = _evidence_request()
    with pytest.raises(ValueError, match="six endpoint"):
        EvidenceRequest(
            document_id=request.document_id,
            unit_id=request.unit_id,
            episode_start_frame=0,
            episode_end_frame=25,
            motion_summary=request.motion_summary,
            task_instruction=request.task_instruction,
            detail=None,
            endpoints=request.endpoints[:5],
        )


def test_openai_inspection_uses_six_images_without_task_text() -> None:
    inspection = {
        "motion_summary": "The left gripper approaches one yellow object.",
        "interaction_window": {"start_frame": 5, "end_frame": 18},
        "needs_detail": False,
        "detail_request": None,
    }

    class Responses:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                id="req-inspect",
                model="response-model",
                usage={"total_tokens": 10},
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="locate_temporal_detail",
                        arguments=json.dumps(inspection),
                    )
                ],
            )

    responses = Responses()
    result = OpenAIBackend(
        "requested-model", client=SimpleNamespace(responses=responses)
    ).inspect(_inspection_request())
    body = responses.kwargs
    assert body["tools"][0]["strict"] is True
    assert sum(
        item["type"] == "input_image" for item in body["input"][0]["content"]
    ) == 6
    assert "Put bread" not in json.dumps(body)
    assert result.trace.request_id == "req-inspect"


def test_openai_call2_uses_endpoints_detail_motion_and_task_last(call2_record) -> None:
    class Responses:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                id="req-evidence",
                model="response-model",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name=TOOL_NAME,
                        arguments=json.dumps(call2_record),
                    )
                ],
            )

    responses = Responses()
    result = OpenAIBackend(
        "requested-model", client=SimpleNamespace(responses=responses)
    ).annotate(_evidence_request(detail=True))
    content = responses.kwargs["input"][0]["content"]
    assert "motion summary" in content[0]["text"].lower()
    assert content[-1]["type"] == "input_text"
    assert "Task instruction" in content[-1]["text"]
    assert sum(item["type"] == "input_image" for item in content) == 7
    assert result.evidence["unit_interpretation"]["action_description"]


def test_retry_context_is_passed_to_call2(call2_record) -> None:
    class Responses:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name=TOOL_NAME,
                        arguments=json.dumps(call2_record),
                    )
                ]
            )

    responses = Responses()
    OpenAIBackend(
        "requested-model", client=SimpleNamespace(responses=responses)
    ).annotate(_evidence_request(previous_attempt=call2_record))
    assert "Previous Call 2 interpretation was marked retry" in responses.kwargs["input"][0]["content"][-1]["text"]


def test_openai_rejects_malformed_tool_arguments() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name=TOOL_NAME,
                        arguments="{",
                    )
                ]
            )
        )
    )
    with pytest.raises(AnnotationError, match="malformed"):
        OpenAIBackend("test-model", client=client).annotate(_evidence_request())
