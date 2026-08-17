import json
from types import SimpleNamespace

import pytest

from video_harness.annotations import (
    AnnotationError,
    AnthropicEvidenceBackend,
    EvidenceRequest,
    MockEvidenceBackend,
    OpenAIEvidenceBackend,
)
from video_harness.prompts import EVIDENCE_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, TOOL_NAME


def test_mock_is_explicit_and_schema_valid() -> None:
    backend = MockEvidenceBackend()
    result = backend.annotate(
        EvidenceRequest(
            document_id="doc",
            unit_id="u0000",
            task_instruction="Put bread into the toaster.",
            camera_key="observation.images.cam_high",
            before_frame=0,
            after_frame=25,
        )
    )
    assert result.evidence["change_status"] == "insufficient_visual_evidence"
    assert result.prompt_version == PROMPT_VERSION
    assert backend.requires_images is False


def test_prompt_defines_evidence_hierarchy() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).lower()
    assert "images are authoritative evidence" in normalized
    assert "operation_hint is an explicitly bounded hypothesis" in normalized
    assert "must never create a visual fact" in normalized
    assert "do not claim task success" in normalized


def test_provider_schema_uses_only_portable_strict_keywords() -> None:
    forbidden = {"maxItems", "minItems", "uniqueItems", "maxLength", "minLength"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(EVIDENCE_SCHEMA)


def _image_request() -> EvidenceRequest:
    return EvidenceRequest(
        document_id="doc",
        unit_id="u0000",
        task_instruction="Put bread into the toaster.",
        camera_key="observation.images.cam_high",
        before_frame=0,
        after_frame=25,
        before_image=b"before-jpeg",
        after_image=b"after-jpeg",
    )


def test_openai_adapter_requires_shared_strict_schema(changed_evidence) -> None:
    class Responses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name=TOOL_NAME,
                        arguments=json.dumps(changed_evidence),
                    )
                ]
            )

    responses = Responses()
    result = OpenAIEvidenceBackend(
        "test-model", client=SimpleNamespace(responses=responses)
    ).annotate(_image_request())
    assert result.evidence["entities"][0]["role"] == "manipulated_object"
    assert responses.kwargs["tool_choice"]["name"] == TOOL_NAME
    assert responses.kwargs["tools"][0]["strict"] is True


def test_anthropic_adapter_requires_shared_strict_schema(changed_evidence) -> None:
    class Messages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=changed_evidence)]
            )

    messages = Messages()
    result = AnthropicEvidenceBackend(
        "test-model", client=SimpleNamespace(messages=messages)
    ).annotate(_image_request())
    assert result.evidence["operation_hint"]["label"] == "insert"
    assert messages.kwargs["tool_choice"]["name"] == TOOL_NAME
    assert messages.kwargs["tools"][0]["strict"] is True
    assert messages.kwargs["max_tokens"] == 1024


def test_openai_adapter_normalizes_malformed_tool_arguments() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                output=[SimpleNamespace(type="function_call", name=TOOL_NAME, arguments="{")]
            )
        )
    )
    with pytest.raises(AnnotationError, match="malformed"):
        OpenAIEvidenceBackend("test-model", client=client).annotate(_image_request())


def test_anthropic_adapter_rejects_extra_root_field(changed_evidence) -> None:
    changed_evidence["extra"] = True
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=changed_evidence)]
            )
        )
    )
    with pytest.raises(AnnotationError, match="invalid transition evidence"):
        AnthropicEvidenceBackend("test-model", client=client).annotate(_image_request())
