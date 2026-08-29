import json

import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from video_harness.annotations import (
    AnthropicBackend,
    EvidenceRequest,
    ImagePayload,
    InspectionRequest,
    OpenAIBackend,
)
from video_harness.camera_contract import image_label
from video_harness.gripper_state import GripperState


def _image(label: str) -> ImagePayload:
    return ImagePayload(label, b"jpeg", "image/jpeg")


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


def _inspection_request() -> InspectionRequest:
    views = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    return InspectionRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=0,
        episode_end_frame=25,
        overviews=tuple(_camera_image("OVERVIEW", view) for view in views),
        keyframe_sheets=tuple(_camera_image("KEYFRAME_SHEET", view) for view in views),
        gripper_state=_gripper_state(),
    )


def _evidence_request() -> EvidenceRequest:
    return EvidenceRequest(
        document_id="doc",
        unit_id="u0000",
        episode_start_frame=0,
        episode_end_frame=25,
        motion_summary="The gripper approaches the bread and closes around it.",
        before_boundary_observation=None,
        after_boundary_observation=None,
        task_instruction="Put bread into the toaster.",
        detail=_camera_image("DETAIL", "cam_high"),
        boundary_images=(
            _camera_image("BOUNDARY_BEFORE", "cam_high"),
            _camera_image("BOUNDARY_BEFORE", "cam_left_wrist"),
            _camera_image("BOUNDARY_BEFORE", "cam_right_wrist"),
            _camera_image("BOUNDARY_AFTER", "cam_high"),
            _camera_image("BOUNDARY_AFTER", "cam_left_wrist"),
            _camera_image("BOUNDARY_AFTER", "cam_right_wrist"),
        ),
        gripper_state=_gripper_state(),
    )


def test_openai_sdk_serializes_inspection_images_without_task() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "intentional local mock",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(openai.BadRequestError):
        OpenAIBackend("test-model", client=client).inspect(_inspection_request())
    body = bodies[0]
    assert body["tools"][0]["strict"] is True
    assert body["max_output_tokens"] == 768
    assert (
        sum(item["type"] == "input_image" for item in body["input"][0]["content"]) == 6
    )
    assert "Put bread" not in json.dumps(body)
    assert "Measured gripper aperture" in json.dumps(body)


def test_openai_sdk_serializes_evidence_images_and_task_last() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "intentional local mock",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(openai.BadRequestError):
        OpenAIBackend("test-model", client=client).annotate(_evidence_request())
    assert bodies[0]["max_output_tokens"] == 1200
    content = bodies[0]["input"][0]["content"]
    image_positions = [
        index for index, item in enumerate(content) if item["type"] == "input_image"
    ]
    motion_position = next(
        index
        for index, item in enumerate(content)
        if item["type"] == "input_text" and "motion summary" in item["text"].lower()
    )
    assert len(image_positions) == 7
    assert content[0]["text"].startswith("EVIDENCE=BOUNDARY_BEFORE")
    assert max(image_positions) < motion_position < len(content) - 1
    assert "Task instruction" in content[-1]["text"]
    assert "Measured gripper aperture" in content[motion_position]["text"]


def test_deepseek_responses_disable_thinking_for_forced_tool_choice() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "intentional local mock",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://api.deepseek.com",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(openai.BadRequestError):
        OpenAIBackend("deepseek-v4-flash-vision-exp", client=client).inspect(
            _inspection_request()
        )

    assert bodies[0]["reasoning"] == {"effort": "none"}
    assert bodies[0]["max_output_tokens"] == 768
    assert bodies[0]["tool_choice"]["name"] == "locate_temporal_detail"


def test_anthropic_sdk_serializes_multiview_evidence() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "intentional local mock",
                },
            },
        )

    client = anthropic.Anthropic(
        api_key="local-test-key",
        base_url="https://local.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(anthropic.BadRequestError):
        AnthropicBackend("test-model", client=client).annotate(_evidence_request())
    body = bodies[0]
    assert body["tools"][0]["strict"] is True
    assert body["max_tokens"] == 1200
    content = body["messages"][0]["content"]
    image_positions = [
        index for index, item in enumerate(content) if item["type"] == "image"
    ]
    motion_position = next(
        index
        for index, item in enumerate(content)
        if item["type"] == "text" and "motion summary" in item["text"].lower()
    )
    assert len(image_positions) == 7
    assert content[0]["text"].startswith("EVIDENCE=BOUNDARY_BEFORE")
    assert max(image_positions) < motion_position < len(content) - 1
    assert "Task instruction" in content[-1]["text"]
