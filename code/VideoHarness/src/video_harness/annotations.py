from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, is_dataclass
import json
import os
from typing import Any, Protocol

from .camera_contract import CAMERA_VIEWS, validate_image_label
from .evidence import (
    EvidenceValidationError,
    mock_call2_record,
    mock_inspection_record,
    validate_call2_record,
    validate_inspection_record,
)
from .prompts import (
    EVIDENCE_SCHEMA,
    INSPECTION_PROMPT_VERSION,
    INSPECTION_SCHEMA,
    INSPECTION_SYSTEM_PROMPT,
    INSPECTION_TOOL_NAME,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_NAME,
    call2_context_prompt,
    evidence_user_prompt,
    inspection_user_prompt,
)
from .protocol import ImagePayload


class AnnotationError(RuntimeError):
    """Raised when a provider response cannot become trusted structured output."""


@dataclass(frozen=True)
class InspectionRequest:
    document_id: str
    unit_id: str
    episode_start_frame: int
    episode_end_frame: int
    overviews: tuple[ImagePayload, ...]
    stages: tuple[ImagePayload, ...]

    def __post_init__(self) -> None:
        if len(self.overviews) != 3:
            raise ValueError("InspectionRequest requires three overview images")
        for view, payload in zip(CAMERA_VIEWS, self.overviews, strict=True):
            validate_image_label(payload.label, evidence_role="OVERVIEW", view=view)
        if len(self.stages) != 3:
            raise ValueError("InspectionRequest requires three stage images")
        for view, payload in zip(CAMERA_VIEWS, self.stages, strict=True):
            validate_image_label(payload.label, evidence_role="STAGE", view=view)


@dataclass(frozen=True)
class EvidenceRequest:
    document_id: str
    unit_id: str
    episode_start_frame: int
    episode_end_frame: int
    motion_summary: str
    task_instruction: str
    detail: ImagePayload | None
    endpoints: tuple[ImagePayload, ...]
    previous_attempt: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.motion_summary.strip():
            raise ValueError("EvidenceRequest requires a non-empty motion_summary")
        if len(self.endpoints) != 6:
            raise ValueError("EvidenceRequest requires six endpoint images")
        expected = tuple(
            (f"ENDPOINT_{role}", view)
            for role in ("BEFORE", "AFTER")
            for view in CAMERA_VIEWS
        )
        for (role, view), payload in zip(expected, self.endpoints, strict=True):
            validate_image_label(payload.label, evidence_role=role, view=view)
        if self.detail is not None:
            validate_image_label(
                self.detail.label, evidence_role="DETAIL", view="cam_high"
            )


@dataclass(frozen=True)
class ProviderTrace:
    request_id: str | None = None
    response_model: str | None = None
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class InspectionResult:
    inspection: dict[str, Any]
    provider: str
    requested_model: str
    prompt_version: str = INSPECTION_PROMPT_VERSION
    trace: ProviderTrace = ProviderTrace()


@dataclass(frozen=True)
class EvidenceResult:
    evidence: dict[str, Any]
    provider: str
    requested_model: str
    prompt_version: str = PROMPT_VERSION
    trace: ProviderTrace = ProviderTrace()


class InspectionBackend(Protocol):
    provider: str
    model: str

    def inspect(self, request: InspectionRequest) -> InspectionResult: ...


class EvidenceBackend(Protocol):
    provider: str
    model: str

    def annotate(self, request: EvidenceRequest) -> EvidenceResult: ...


class MockInspectionBackend:
    provider = "mock"
    model = "deterministic-no-detail"

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        del request
        return InspectionResult(
            inspection=mock_inspection_record(),
            provider=self.provider,
            requested_model=self.model,
        )


class MockEvidenceBackend:
    provider = "mock"
    model = "deterministic-insufficient-evidence"

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        del request
        return EvidenceResult(
            evidence=mock_call2_record(),
            provider=self.provider,
            requested_model=self.model,
        )


def _data_url(image: ImagePayload) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.media_type};base64,{encoded}"


def _usage_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if is_dataclass(value):
        return asdict(value)
    return {"value": str(value)}


def _trace(response: Any, requested_model: str) -> ProviderTrace:
    return ProviderTrace(
        request_id=getattr(response, "id", None),
        response_model=getattr(response, "model", None) or requested_model,
        usage=_usage_dict(getattr(response, "usage", None)),
    )


def _openai_image_content(images: tuple[ImagePayload, ...]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "input_text", "text": image.label})
        content.append({"type": "input_image", "image_url": _data_url(image)})
    return content


def _anthropic_image_content(images: tuple[ImagePayload, ...]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "text", "text": image.label})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": base64.b64encode(image.data).decode("ascii"),
                },
            }
        )
    return content


def _inspection_images(request: InspectionRequest) -> tuple[ImagePayload, ...]:
    return (*request.overviews, *request.stages)


def _evidence_images(request: EvidenceRequest) -> tuple[ImagePayload, ...]:
    detail = () if request.detail is None else (request.detail,)
    return (*request.endpoints, *detail)


def _json_arguments(value: Any, provider: str, role: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise AnnotationError(f"{provider} returned non-JSON {role} arguments")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{provider} returned malformed {role} arguments") from exc
    if not isinstance(parsed, dict):
        raise AnnotationError(f"{provider} returned non-object {role} arguments")
    return parsed


class OpenAIBackend:
    provider = "openai"

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the providers extra to use OpenAI") from exc
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = client
        self.model = model

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        content = [
            {
                "type": "input_text",
                "text": inspection_user_prompt(
                    document_id=request.document_id,
                    unit_id=request.unit_id,
                    episode_start_frame=request.episode_start_frame,
                    episode_end_frame=request.episode_end_frame,
                ),
            },
            *_openai_image_content(_inspection_images(request)),
        ]
        response = self.client.responses.create(
            model=self.model,
            instructions=INSPECTION_SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
            tools=[
                {
                    "type": "function",
                    "name": INSPECTION_TOOL_NAME,
                    "description": "Locate one optional cam_high detail region from ordered multiview temporal evidence.",
                    "parameters": INSPECTION_SCHEMA,
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": INSPECTION_TOOL_NAME},
        )
        for item in response.output:
            if (
                getattr(item, "type", None) == "function_call"
                and getattr(item, "name", None) == INSPECTION_TOOL_NAME
            ):
                raw = _json_arguments(item.arguments, "OpenAI", "inspection")
                try:
                    inspection = validate_inspection_record(raw)
                except EvidenceValidationError as exc:
                    raise AnnotationError(
                        f"OpenAI returned invalid inspection output: {exc}"
                    ) from exc
                return InspectionResult(
                    inspection=inspection,
                    provider=self.provider,
                    requested_model=self.model,
                    trace=_trace(response, self.model),
                )
        raise AnnotationError("OpenAI response did not contain the inspection tool call")

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        images = _evidence_images(request)
        content = [
            {
                "type": "input_text",
                "text": call2_context_prompt(motion_summary=request.motion_summary),
            },
            *_openai_image_content(images),
        ]
        content.append(
            {
                "type": "input_text",
                "text": evidence_user_prompt(
                    document_id=request.document_id,
                    unit_id=request.unit_id,
                    episode_start_frame=request.episode_start_frame,
                    episode_end_frame=request.episode_end_frame,
                    task_instruction=request.task_instruction,
                    detail_supplied=request.detail is not None,
                    previous_attempt=request.previous_attempt,
                ),
            }
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
            tools=[
                {
                    "type": "function",
                    "name": TOOL_NAME,
                    "description": "Compile one strict evidence record from ordered temporal views and synchronized endpoints.",
                    "parameters": EVIDENCE_SCHEMA,
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": TOOL_NAME},
        )
        for item in response.output:
            if (
                getattr(item, "type", None) == "function_call"
                and getattr(item, "name", None) == TOOL_NAME
            ):
                raw = _json_arguments(item.arguments, "OpenAI", "evidence")
                try:
                    evidence = validate_call2_record(raw)
                except EvidenceValidationError as exc:
                    raise AnnotationError(
                        f"OpenAI returned invalid Call 2 evidence: {exc}"
                    ) from exc
                return EvidenceResult(
                    evidence=evidence,
                    provider=self.provider,
                    requested_model=self.model,
                    trace=_trace(response, self.model),
                )
        raise AnnotationError("OpenAI response did not contain the evidence tool call")


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the providers extra to use Anthropic") from exc
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": inspection_user_prompt(
                    document_id=request.document_id,
                    unit_id=request.unit_id,
                    episode_start_frame=request.episode_start_frame,
                    episode_end_frame=request.episode_end_frame,
                ),
            },
            *_anthropic_image_content(_inspection_images(request)),
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=INSPECTION_SYSTEM_PROMPT,
            tools=[
                {
                    "name": INSPECTION_TOOL_NAME,
                    "description": "Locate one optional cam_high detail region from ordered multiview temporal evidence.",
                    "input_schema": INSPECTION_SCHEMA,
                    "strict": True,
                }
            ],
            tool_choice={"type": "tool", "name": INSPECTION_TOOL_NAME},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == INSPECTION_TOOL_NAME
            ):
                raw = _json_arguments(block.input, "Anthropic", "inspection")
                try:
                    inspection = validate_inspection_record(raw)
                except EvidenceValidationError as exc:
                    raise AnnotationError(
                        f"Anthropic returned invalid inspection output: {exc}"
                    ) from exc
                return InspectionResult(
                    inspection=inspection,
                    provider=self.provider,
                    requested_model=self.model,
                    trace=_trace(response, self.model),
                )
        raise AnnotationError("Anthropic response did not contain the inspection tool call")

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        images = _evidence_images(request)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": call2_context_prompt(motion_summary=request.motion_summary),
            },
            *_anthropic_image_content(images),
        ]
        content.append(
            {
                "type": "text",
                "text": evidence_user_prompt(
                    document_id=request.document_id,
                    unit_id=request.unit_id,
                    episode_start_frame=request.episode_start_frame,
                    episode_end_frame=request.episode_end_frame,
                    task_instruction=request.task_instruction,
                    detail_supplied=request.detail is not None,
                    previous_attempt=request.previous_attempt,
                ),
            }
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1536,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Compile one strict evidence record from ordered temporal views and synchronized endpoints.",
                    "input_schema": EVIDENCE_SCHEMA,
                    "strict": True,
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == TOOL_NAME
            ):
                raw = _json_arguments(block.input, "Anthropic", "evidence")
                try:
                    evidence = validate_call2_record(raw)
                except EvidenceValidationError as exc:
                    raise AnnotationError(
                        f"Anthropic returned invalid Call 2 evidence: {exc}"
                    ) from exc
                return EvidenceResult(
                    evidence=evidence,
                    provider=self.provider,
                    requested_model=self.model,
                    trace=_trace(response, self.model),
                )
        raise AnnotationError("Anthropic response did not contain the evidence tool call")


def make_backends(
    provider: str,
    model: str | None,
) -> tuple[InspectionBackend, EvidenceBackend]:
    if provider == "mock":
        return MockInspectionBackend(), MockEvidenceBackend()
    if not model:
        raise ValueError(f"--model is required for provider {provider!r}")
    if provider == "openai":
        backend = OpenAIBackend(model)
        return backend, backend
    if provider == "anthropic":
        backend = AnthropicBackend(model)
        return backend, backend
    raise ValueError(f"Unknown provider: {provider}")
