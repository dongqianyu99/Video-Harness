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


def _image(label: str) -> ImagePayload:
    return ImagePayload(label, b"jpeg", "image/jpeg")


def _camera_image(role: str, view: str) -> ImagePayload:
    return _image(
        image_label(evidence_role=role, view=view, metadata="TEST_FIXTURE=true")
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
        detail=None,
        boundary_images=(
            _camera_image("BOUNDARY_BEFORE", "cam_high"),
            _camera_image("BOUNDARY_BEFORE", "cam_left_wrist"),
            _camera_image("BOUNDARY_BEFORE", "cam_right_wrist"),
            _camera_image("BOUNDARY_AFTER", "cam_high"),
            _camera_image("BOUNDARY_AFTER", "cam_left_wrist"),
            _camera_image("BOUNDARY_AFTER", "cam_right_wrist"),
        ),
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
    assert (
        sum(item["type"] == "input_image" for item in body["input"][0]["content"]) == 6
    )
    assert "Put bread" not in json.dumps(body)


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
    content = bodies[0]["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 6
    assert "motion summary" in content[0]["text"].lower()
    assert "Task instruction" in content[-1]["text"]


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
    assert sum(item["type"] == "image" for item in body["messages"][0]["content"]) == 6
    assert "motion summary" in body["messages"][0]["content"][0]["text"].lower()
    assert "Task instruction" in body["messages"][0]["content"][-1]["text"]
