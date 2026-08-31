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
from video_harness.gripper_state import GripperState
from video_harness.prompts import (
    EVIDENCE_SCHEMA,
    INSPECTION_SCHEMA,
    INSPECTION_SYSTEM_PROMPT,
    PROMPT_VERSION,
    REPAIR_SYSTEM_PROMPT,
    SEQUENCE_AUDIT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOOL_NAME,
)


def _image(label: str) -> ImagePayload:
    return ImagePayload(label=label, data=b"jpeg", media_type="image/jpeg")


def _camera_image(role: str, view: str) -> ImagePayload:
    return _image(
        image_label(evidence_role=role, view=view, metadata="TEST_FIXTURE=true")
    )


def _gripper_state() -> GripperState:
    return GripperState(
        unit_frame_indices=(0, 5, 10, 15, 20, 25),
        left=(1.0,) * 6,
        right=(1.0, 0.8, 0.4, 0.4, 0.8, 1.0),
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
        gripper_state=_gripper_state(),
        previous_motion_summary=previous_motion_summary,
    )


def _evidence_request() -> EvidenceRequest:
    return EvidenceRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=100,
        episode_end_frame=125,
        motion_summary="The right gripper approaches the bread slice and closes around it.",
        before_boundary_observation=None,
        after_boundary_observation=None,
        task_instruction="Put bread into the toaster.",
        detail=_camera_image("DETAIL", "cam_high"),
        boundary_images=tuple(
            _camera_image(f"BOUNDARY_{role}", view)
            for role in ("BEFORE", "AFTER")
            for view in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ),
        gripper_state=_gripper_state(),
    )


def test_mock_backends_are_explicit_and_schema_valid() -> None:
    inspection = MockInspectionBackend().inspect(_inspection_request())
    evidence = MockEvidenceBackend().annotate(_evidence_request())
    assert inspection.inspection["detail_request"] is not None
    assert "active_end_effector" not in inspection.inspection
    assert "causal_validation" not in evidence.evidence
    assert "boundary_conflicts" not in evidence.evidence
    assert evidence.prompt_version == PROMPT_VERSION


def test_prompt_and_schema_define_progressive_calls() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).lower()
    assert "task-blind motion_summary from call 1" in normalized
    assert "describe each boundary state once" in normalized
    assert "exactly one concise sentence per camera view" in normalized
    assert "return motion_summary as exactly one concise sentence" in normalized
    assert "action_description as exactly one concise, task-grounded restatement" in (
        normalized
    )
    assert "without adding physical claims beyond it" in normalized
    assert "task_role as exactly one concise sentence" in normalized
    assert "grasp, hold, release, or contact require direct supporting" in normalized
    assert "decompose motion_summary and action_description" in normalized
    assert "measured gripper aperture" in normalized
    assert "do not speculate about, quote, or reproduce" in normalized
    assert "does not alone prove object attachment" not in normalized
    assert "one physically coherent account of persistent entities" in normalized
    assert "causal_validation" not in EVIDENCE_SCHEMA["properties"]
    assert "boundary_conflicts" not in EVIDENCE_SCHEMA["properties"]
    assert "active_end_effector" not in INSPECTION_SCHEMA["properties"]
    inspection_prompt = " ".join(INSPECTION_SYSTEM_PROMPT.split()).lower()
    assert "motion_summary as exactly one concise sentence" in inspection_prompt
    assert "do not inventory or repeat the static scene" in inspection_prompt
    assert "measured gripper aperture" in inspection_prompt
    assert "do not speculate about, quote, or reproduce" in inspection_prompt
    assert "one physically coherent account of persistent entities" in (
        inspection_prompt
    )
    assert "always select one cam_high roi" in inspection_prompt
    assert "needs_detail" not in INSPECTION_SCHEMA["properties"]
    assert set(INSPECTION_SCHEMA["properties"]["detail_request"]["required"]) == {
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    repair_prompt = " ".join(REPAIR_SYSTEM_PROMPT.split()).lower()
    assert "one physically coherent account of persistent entities" in repair_prompt
    audit_prompt = " ".join(SEQUENCE_AUDIT_SYSTEM_PROMPT.split()).lower()
    assert (
        "persistent entities and their relations to the end effectors" in audit_prompt
    )
    assert "action_description is entailed by its motion_summary" in audit_prompt
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
            gripper_state=_gripper_state(),
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
            detail=request.detail,
            boundary_images=request.boundary_images[:5],
            gripper_state=_gripper_state(),
        )
    with pytest.raises(ValueError, match="detail image"):
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
            boundary_images=request.boundary_images,
            gripper_state=_gripper_state(),
        )


def test_openai_inspection_uses_six_images_without_task_text() -> None:
    inspection = {
        "motion_summary": "The left gripper approaches one yellow object.",
        "interaction_window": {"start_frame": 5, "end_frame": 18},
        "detail_request": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.7, "y_max": 0.7},
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
        "detail_request": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.7, "y_max": 0.7},
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
    ).annotate(_evidence_request())
    content = responses.kwargs["input"][0]["content"]
    assert content[0]["text"].startswith("EVIDENCE=BOUNDARY_BEFORE")
    image_positions = [
        index for index, item in enumerate(content) if item["type"] == "input_image"
    ]
    motion_position = next(
        index
        for index, item in enumerate(content)
        if item["type"] == "input_text" and "motion summary" in item["text"].lower()
    )
    assert max(image_positions) < motion_position < len(content) - 1
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
