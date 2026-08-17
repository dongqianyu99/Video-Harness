from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

from .evidence import EvidenceValidationError, mock_evidence_record, validate_evidence_record
from .prompts import EVIDENCE_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, TOOL_NAME, user_prompt


class AnnotationError(RuntimeError):
    """Raised when a provider response cannot become trusted canonical evidence."""


@dataclass(frozen=True)
class EvidenceRequest:
    document_id: str
    unit_id: str
    task_instruction: str
    camera_key: str
    before_frame: int
    after_frame: int
    before_image: bytes | None = None
    after_image: bytes | None = None


@dataclass(frozen=True)
class EvidenceResult:
    evidence: dict[str, Any]
    provider: str
    model: str
    prompt_version: str = PROMPT_VERSION


class EvidenceBackend(Protocol):
    provider: str
    model: str
    requires_images: bool

    def annotate(self, request: EvidenceRequest) -> EvidenceResult: ...


class MockEvidenceBackend:
    provider = "mock"
    model = "deterministic-insufficient-evidence"
    requires_images = False

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        del request
        return EvidenceResult(
            evidence=mock_evidence_record(),
            provider=self.provider,
            model=self.model,
        )


def _require_images(request: EvidenceRequest) -> tuple[bytes, bytes]:
    if request.before_image is None or request.after_image is None:
        raise AnnotationError("This provider requires BEFORE and AFTER JPEG bytes")
    return request.before_image, request.after_image


def _data_url(image: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")


def _request_text(request: EvidenceRequest) -> str:
    return user_prompt(
        task_instruction=request.task_instruction,
        camera_key=request.camera_key,
        unit_id=request.unit_id,
        before_frame=request.before_frame,
        after_frame=request.after_frame,
    )


def _validated_provider_payload(payload: Any, provider: str) -> dict[str, Any]:
    try:
        return validate_evidence_record(payload)
    except EvidenceValidationError as exc:
        raise AnnotationError(f"{provider} returned invalid transition evidence: {exc}") from exc


class OpenAIEvidenceBackend:
    provider = "openai"
    requires_images = True

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install the providers extra to use OpenAI") from exc
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = client
        self.model = model

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        before, after = _require_images(request)
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _request_text(request) + "\nBEFORE image:"},
                        {"type": "input_image", "image_url": _data_url(before)},
                        {"type": "input_text", "text": "AFTER image:"},
                        {"type": "input_image", "image_url": _data_url(after)},
                    ],
                }
            ],
            tools=[
                {
                    "type": "function",
                    "name": TOOL_NAME,
                    "description": (
                        "Compile one strict, evidence-grounded behavior record from the labeled "
                        "BEFORE and AFTER endpoints. Separate direct visual state, entity roles, "
                        "bounded operation hypotheses, task relevance, and visibility limits."
                    ),
                    "parameters": EVIDENCE_SCHEMA,
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": TOOL_NAME},
        )
        for item in response.output:
            if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == TOOL_NAME:
                try:
                    payload = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise AnnotationError("OpenAI returned malformed evidence tool arguments") from exc
                evidence = _validated_provider_payload(payload, "OpenAI")
                return EvidenceResult(evidence, self.provider, self.model)
        raise AnnotationError("OpenAI response did not contain the required evidence function call")


class AnthropicEvidenceBackend:
    provider = "anthropic"
    requires_images = True

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install the providers extra to use Anthropic") from exc
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        before, after = _require_images(request)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": (
                        "Compile one strict evidence record for a robot behavior document. "
                        "Direct visual state must remain separate from task-conditioned entity "
                        "roles and operation hypotheses. Two sampled endpoints do not reveal "
                        "the hidden path, force, or precise pose."
                    ),
                    "input_schema": EVIDENCE_SCHEMA,
                    "strict": True,
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _request_text(request) + "\nBEFORE image:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(before).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": "AFTER image:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(after).decode("ascii"),
                            },
                        },
                    ],
                }
            ],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
                evidence = _validated_provider_payload(getattr(block, "input", None), "Anthropic")
                return EvidenceResult(evidence, self.provider, self.model)
        raise AnnotationError("Anthropic response did not contain the required evidence tool call")


def make_backend(provider: str, model: str | None) -> EvidenceBackend:
    if provider == "mock":
        return MockEvidenceBackend()
    if not model:
        raise ValueError(f"--model is required for provider {provider!r}")
    if provider == "openai":
        return OpenAIEvidenceBackend(model)
    if provider == "anthropic":
        return AnthropicEvidenceBackend(model)
    raise ValueError(f"Unknown provider: {provider}")
