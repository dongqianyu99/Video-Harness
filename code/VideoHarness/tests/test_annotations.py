import json
import sys
from types import SimpleNamespace

import pytest

from video_harness.annotations import (
    AnnotationError,
    AnthropicBackend,
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


def _inspection_request(
    *, previous_motion_summary: str | None = None
) -> InspectionRequest:
    views = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    return InspectionRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=100,
        episode_end_frame=125,
        overviews=tuple(_camera_image("OVERVIEW", view) for view in views),
        keyframe_sheets=tuple(_camera_image("KEYFRAME_SHEET", view) for view in views),
        previous_motion_summary=previous_motion_summary,
    )


def _evidence_request(*, detail: bool = False) -> EvidenceRequest:
    return EvidenceRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=100,
        episode_end_frame=125,
        motion_summary="The right gripper approaches the bread slice and closes around it.",
        before_boundary_observation=None,
        after_boundary_observation=None,
        task_instruction="Put bread into the toaster.",
        detail=_camera_image("DETAIL", "cam_high") if detail else None,
        boundary_images=tuple(
            _camera_image(f"BOUNDARY_{role}", view)
            for role in ("BEFORE", "AFTER")
            for view in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ),
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
    assert "describe each boundary state once" in normalized
    assert "status=retry only for a clear violation" in normalized
    assert "active_end_effector" not in INSPECTION_SCHEMA["properties"]
    before = EVIDENCE_SCHEMA["properties"]["before_boundary_observation"]["anyOf"][0]
    assert set(before["required"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }


def test_request_requires_three_overviews_three_keyframe_sheets_and_six_boundary_images() -> (
    None
):
    inspection = _inspection_request()
    with pytest.raises(ValueError, match="three keyframe-sheet"):
        InspectionRequest(
            document_id="doc",
            unit_id="u0000",
            episode_start_frame=0,
            episode_end_frame=25,
            overviews=inspection.overviews,
            keyframe_sheets=inspection.keyframe_sheets[:2],
        )
    request = _evidence_request()
    with pytest.raises(ValueError, match="six Boundary"):
        EvidenceRequest(
            document_id=request.document_id,
            unit_id=request.unit_id,
            episode_start_frame=0,
            episode_end_frame=25,
            motion_summary=request.motion_summary,
            before_boundary_observation=None,
            after_boundary_observation=None,
            task_instruction=request.task_instruction,
            detail=None,
            boundary_images=request.boundary_images[:5],
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
    assert (
        sum(item["type"] == "input_image" for item in body["input"][0]["content"]) == 6
    )
    assert "Put bread" not in json.dumps(body)
    assert result.trace.request_id == "req-inspect"


def test_openai_inspection_passes_only_one_step_task_blind_context() -> None:
    inspection = {
        "motion_summary": "The interaction continues across the shared boundary.",
        "interaction_window": {"start_frame": 0, "end_frame": 8},
        "needs_detail": False,
        "detail_request": None,
    }

    class Responses:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="locate_temporal_detail",
                        arguments=json.dumps(inspection),
                    )
                ]
            )

    responses = Responses()
    OpenAIBackend(
        "requested-model", client=SimpleNamespace(responses=responses)
    ).inspect(
        _inspection_request(
            previous_motion_summary=(
                "The gripper remains in a visually supported interaction at the final frame."
            )
        )
    )
    user_text = responses.kwargs["input"][0]["content"][0]["text"]
    assert "Previous Evidence Unit context" in user_text
    assert "task-blind and fallible" in user_text
    assert "remains in a visually supported interaction" in user_text


def test_openai_call2_uses_boundaries_detail_motion_and_task_last(call2_record) -> None:
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


def test_provider_clients_receive_explicit_timeout_and_retry_config(
    monkeypatch,
) -> None:
    created = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def openai_factory(**kwargs):
        created["openai"] = kwargs
        return SimpleNamespace()

    def anthropic_factory(**kwargs):
        created["anthropic"] = kwargs
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_factory))
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=anthropic_factory),
    )
    OpenAIBackend("model", timeout_s=45, max_retries=3)
    AnthropicBackend("model", timeout_s=60, max_retries=4)
    assert created == {
        "openai": {"api_key": None, "timeout": 45, "max_retries": 3},
        "anthropic": {"api_key": None, "timeout": 60, "max_retries": 4},
    }
