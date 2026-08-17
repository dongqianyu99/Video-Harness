import json

import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from video_harness.annotations import (  # noqa: E402
    AnthropicEvidenceBackend,
    EvidenceRequest,
    OpenAIEvidenceBackend,
)


def _request() -> EvidenceRequest:
    return EvidenceRequest(
        document_id="doc",
        unit_id="u0000",
        task_instruction="Put the bread into the toaster.",
        camera_key="observation.images.cam_high",
        before_frame=0,
        after_frame=25,
        before_image=b"before-jpeg",
        after_image=b"after-jpeg",
    )


def test_openai_sdk_serializes_images_and_strict_tool_without_network() -> None:
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
        OpenAIEvidenceBackend("test-model", client=client).annotate(_request())
    assert bodies[0]["tools"][0]["strict"] is True
    assert "entities" in bodies[0]["tools"][0]["parameters"]["properties"]
    assert [item["type"] for item in bodies[0]["input"][0]["content"]] == [
        "input_text",
        "input_image",
        "input_text",
        "input_image",
    ]


def test_anthropic_sdk_serializes_images_and_strict_tool_without_network() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "intentional local mock"},
            },
        )

    client = anthropic.Anthropic(
        api_key="local-test-key",
        base_url="https://local.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(anthropic.BadRequestError):
        AnthropicEvidenceBackend("test-model", client=client).annotate(_request())
    assert bodies[0]["tools"][0]["strict"] is True
    assert "entities" in bodies[0]["tools"][0]["input_schema"]["properties"]
    assert [item["type"] for item in bodies[0]["messages"][0]["content"]] == [
        "text",
        "image",
        "text",
        "image",
    ]
