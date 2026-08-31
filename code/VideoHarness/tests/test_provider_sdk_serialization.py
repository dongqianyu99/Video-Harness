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
    StructuredOutputError,
)
from video_harness.camera_contract import image_label
from video_harness.gripper_state import GripperState
from video_harness.prompts import (
    EVIDENCE_SCHEMA,
    INSPECTION_SCHEMA,
    REPAIR_SCHEMA,
    SEQUENCE_AUDIT_SCHEMA,
)


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
    assert "locate_temporal_detail tool call" in body["instructions"]
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


def test_json_output_serializes_thinking_without_tools() -> None:
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
        OpenAIBackend(
            "vision-model",
            client=client,
            output_mode="json",
            thinking=True,
            reasoning_effort="max",
        ).inspect(_inspection_request())

    body = bodies[0]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "locate_temporal_detail",
            "description": "Locate the required cam_high detail region.",
            "strict": True,
            "schema": INSPECTION_SCHEMA,
        },
    }
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "max"
    assert body["max_tokens"] == 16768
    assert "tools" not in body and "tool_choice" not in body
    content = body["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 6
    system = body["messages"][0]["content"]
    assert "JSON Schema" not in system
    assert "tool call" not in system
    assert "structured JSON object" in system


@pytest.mark.parametrize(
    ("name", "schema"),
    [
        ("locate_temporal_detail", INSPECTION_SCHEMA),
        ("record_transition_evidence", EVIDENCE_SCHEMA),
        ("resolve_transition_repair", REPAIR_SCHEMA),
        ("audit_sequence_consistency", SEQUENCE_AUDIT_SCHEMA),
    ],
)
def test_json_output_serializes_each_role_schema(name, schema) -> None:
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
    backend = OpenAIBackend("vision-model", client=client, output_mode="json")
    with pytest.raises(openai.BadRequestError):
        backend._call(
            instructions="semantic instructions",
            content=[],
            tool_name=name,
            description="structured output",
            schema=schema,
            role="test",
            max_output_tokens=128,
        )

    assert bodies[0]["response_format"]["json_schema"] == {
        "name": name,
        "description": "structured output",
        "strict": True,
        "schema": schema,
    }


def test_json_output_is_parsed_and_validated() -> None:
    output = {
        "motion_summary": "The left gripper approaches the object.",
        "interaction_window": {"start_frame": 5, "end_frame": 25},
        "detail_request": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.8},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "vision-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(output),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = OpenAIBackend(
        "vision-model",
        client=client,
        output_mode="json",
    ).inspect(_inspection_request())
    assert result.inspection == output


@pytest.mark.parametrize(
    ("content", "finish_reason", "refusal", "kind"),
    [
        (None, "stop", None, "empty_content"),
        ("", "stop", None, "empty_content"),
        ('{"motion_summary":', "length", None, "truncated"),
        ("{}", "content_filter", None, "incomplete_finish"),
        ("```json\n{}\n```", "stop", None, "malformed_json"),
        ("[]", "stop", None, "non_object"),
        (None, "stop", "cannot comply", "refusal"),
    ],
)
def test_json_output_failures_are_classified(
    content, finish_reason, refusal, kind
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-failure",
                "object": "chat.completion",
                "created": 0,
                "model": "vision-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "refusal": refusal,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(StructuredOutputError) as caught:
        OpenAIBackend(
            "vision-model", client=client, output_mode="json"
        ).inspect(_inspection_request())

    assert caught.value.kind == kind
    assert caught.value.diagnostic(include_raw=True)["raw_final_content"] == (
        refusal if refusal else content
    )


def test_json_output_still_passes_local_schema_validation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-invalid-schema",
                "object": "chat.completion",
                "created": 0,
                "model": "vision-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"motion_summary": "incomplete"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(StructuredOutputError) as caught:
        OpenAIBackend(
            "vision-model", client=client, output_mode="json"
        ).inspect(_inspection_request())

    assert caught.value.kind == "schema_validation"


def test_json_output_requires_a_completion_choice() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-empty",
                "object": "chat.completion",
                "created": 0,
                "model": "vision-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                    "total_tokens": 1,
                },
            },
        )

    client = openai.OpenAI(
        api_key="local-test-key",
        base_url="https://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(StructuredOutputError) as caught:
        OpenAIBackend(
            "vision-model", client=client, output_mode="json"
        ).inspect(_inspection_request())

    assert caught.value.kind == "missing_choice"


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
    assert "record_transition_evidence tool call" in body["system"]
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
