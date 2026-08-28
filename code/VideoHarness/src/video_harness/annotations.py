from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

from .camera_contract import CAMERA_VIEWS, validate_image_label
from .evidence import (
    EvidenceValidationError,
    compose_boundary_state_record,
    mock_call2_record,
    mock_inspection_record,
    validate_call2_record,
    validate_inspection_record,
)
from .gripper_state import GripperState
from .prompts import (
    EVIDENCE_SCHEMA,
    INSPECTION_PROMPT_VERSION,
    INSPECTION_SCHEMA,
    INSPECTION_SYSTEM_PROMPT,
    INSPECTION_TOOL_NAME,
    PROMPT_VERSION,
    REPAIR_PROMPT_VERSION,
    REPAIR_SCHEMA,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_TOOL_NAME,
    SEQUENCE_AUDIT_PROMPT_VERSION,
    SEQUENCE_AUDIT_SCHEMA,
    SEQUENCE_AUDIT_SYSTEM_PROMPT,
    SEQUENCE_AUDIT_TOOL_NAME,
    SYSTEM_PROMPT,
    TOOL_NAME,
    call2_context_prompt,
    evidence_user_prompt,
    inspection_user_prompt,
    repair_user_prompt,
    sequence_audit_user_prompt,
)
from .protocol import ImagePayload

INSPECTION_MAX_OUTPUT_TOKENS = 768
EVIDENCE_MAX_OUTPUT_TOKENS = 1200
REPAIR_MAX_OUTPUT_TOKENS = 2048
SEQUENCE_AUDIT_MAX_OUTPUT_TOKENS = 1024


class AnnotationError(RuntimeError):
    """Raised when a provider response cannot become trusted structured output."""


@dataclass(frozen=True)
class InspectionRequest:
    document_id: str
    unit_id: str
    episode_start_frame: int
    episode_end_frame: int
    overviews: tuple[ImagePayload, ...]
    keyframe_sheets: tuple[ImagePayload, ...]
    gripper_state: GripperState
    previous_motion_summary: str | None = None

    def __post_init__(self) -> None:
        if len(self.overviews) != 3:
            raise ValueError("InspectionRequest requires three overview images")
        for view, payload in zip(CAMERA_VIEWS, self.overviews, strict=True):
            validate_image_label(payload.label, evidence_role="OVERVIEW", view=view)
        if len(self.keyframe_sheets) != 3:
            raise ValueError("InspectionRequest requires three keyframe-sheet images")
        for view, payload in zip(CAMERA_VIEWS, self.keyframe_sheets, strict=True):
            validate_image_label(
                payload.label,
                evidence_role="KEYFRAME_SHEET",
                view=view,
            )
        if (
            self.previous_motion_summary is not None
            and not self.previous_motion_summary.strip()
        ):
            raise ValueError(
                "InspectionRequest previous_motion_summary must be non-empty when supplied"
            )


@dataclass(frozen=True)
class EvidenceRequest:
    document_id: str
    unit_id: str
    episode_start_frame: int
    episode_end_frame: int
    motion_summary: str
    before_boundary_observation: dict[str, str] | None
    after_boundary_observation: dict[str, str] | None
    task_instruction: str
    detail: ImagePayload | None
    boundary_images: tuple[ImagePayload, ...]
    gripper_state: GripperState

    def __post_init__(self) -> None:
        if not self.motion_summary.strip():
            raise ValueError("EvidenceRequest requires a non-empty motion_summary")
        if self.before_boundary_observation is not None:
            compose_boundary_state_record(
                self.before_boundary_observation,
                quality_status="accepted",
            )
        if self.after_boundary_observation is not None:
            compose_boundary_state_record(
                self.after_boundary_observation,
                quality_status="accepted",
            )
        if len(self.boundary_images) != 6:
            raise ValueError("EvidenceRequest requires six Boundary images")
        expected = tuple(
            (f"BOUNDARY_{role}", view)
            for role in ("BEFORE", "AFTER")
            for view in CAMERA_VIEWS
        )
        for (role, view), payload in zip(expected, self.boundary_images, strict=True):
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


@dataclass(frozen=True)
class RepairRequest:
    document_id: str
    unit_id: str
    task_instruction: str
    issue_reason: str
    call1_motion_summary: str
    call2: dict[str, Any]
    boundary_context: str
    overviews: tuple[ImagePayload, ...]
    keyframe_sheets: tuple[ImagePayload, ...]
    boundary_images: tuple[ImagePayload, ...]
    gripper_state: GripperState
    detail: ImagePayload | None = None
    required_boundary_replacements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "document_id",
            "unit_id",
            "task_instruction",
            "issue_reason",
            "call1_motion_summary",
            "boundary_context",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RepairRequest {field} must be non-empty")
        if (
            len(self.overviews) != 3
            or len(self.keyframe_sheets) != 3
            or len(self.boundary_images) != 6
        ):
            raise ValueError(
                "RepairRequest requires three overviews, three keyframe sheets, "
                "and six Boundary images"
            )
        if self.detail is not None:
            validate_image_label(
                self.detail.label,
                evidence_role="DETAIL",
                view="cam_high",
            )
        if any(
            role not in {"before", "after"}
            for role in self.required_boundary_replacements
        ):
            raise ValueError("RepairRequest Boundary replacements must be before/after")


@dataclass(frozen=True)
class RepairResult:
    repair: dict[str, Any]
    provider: str
    requested_model: str
    prompt_version: str = REPAIR_PROMPT_VERSION
    trace: ProviderTrace = ProviderTrace()


@dataclass(frozen=True)
class SequenceAuditRequest:
    canonical_sequence: str
    task_instruction: str

    def __post_init__(self) -> None:
        if not self.canonical_sequence.strip():
            raise ValueError("SequenceAuditRequest requires canonical sequence text")
        if not self.task_instruction.strip():
            raise ValueError("SequenceAuditRequest requires a task instruction")


@dataclass(frozen=True)
class SequenceAuditResult:
    audit: dict[str, Any]
    provider: str
    requested_model: str
    prompt_version: str = SEQUENCE_AUDIT_PROMPT_VERSION
    trace: ProviderTrace = ProviderTrace()


class InspectionBackend(Protocol):
    provider: str
    model: str

    def inspect(self, request: InspectionRequest) -> InspectionResult: ...


class EvidenceBackend(Protocol):
    provider: str
    model: str

    def annotate(self, request: EvidenceRequest) -> EvidenceResult: ...


class RepairBackend(Protocol):
    provider: str
    model: str

    def repair(self, request: RepairRequest) -> RepairResult: ...

    def audit_sequence(self, request: SequenceAuditRequest) -> SequenceAuditResult: ...


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
        return EvidenceResult(
            evidence=mock_call2_record(
                include_before_boundary=request.before_boundary_observation is None,
                include_after_boundary=request.after_boundary_observation is None,
            ),
            provider=self.provider,
            requested_model=self.model,
        )


class MockRepairBackend:
    provider = "mock"
    model = "deterministic-auto-repair"

    def repair(self, request: RepairRequest) -> RepairResult:
        return RepairResult(
            {
                "evidence_sufficient": False,
                "reason": request.issue_reason,
                "resolved_call2": None,
            },
            self.provider,
            self.model,
        )

    def audit_sequence(self, request: SequenceAuditRequest) -> SequenceAuditResult:
        del request
        return SequenceAuditResult({"issues": []}, self.provider, self.model)


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
    return (*request.overviews, *request.keyframe_sheets)


def _evidence_images(request: EvidenceRequest) -> tuple[ImagePayload, ...]:
    detail = () if request.detail is None else (request.detail,)
    return (*request.boundary_images, *detail)


def _repair_images(request: RepairRequest) -> tuple[ImagePayload, ...]:
    detail = () if request.detail is None else (request.detail,)
    return (
        *request.overviews,
        *request.keyframe_sheets,
        *request.boundary_images,
        *detail,
    )


def _json_arguments(value: Any, provider: str, role: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise AnnotationError(f"{provider} returned non-JSON {role} arguments")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnnotationError(
            f"{provider} returned malformed {role} arguments"
        ) from exc
    if not isinstance(parsed, dict):
        raise AnnotationError(f"{provider} returned non-object {role} arguments")
    return parsed


def _validate_repair_result(value: Any) -> dict[str, Any]:
    required = {
        "evidence_sufficient",
        "reason",
        "resolved_call2",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AnnotationError("repair output has unexpected fields")
    sufficient = value["evidence_sufficient"]
    reason = value["reason"]
    if not isinstance(sufficient, bool):
        raise AnnotationError("repair evidence_sufficient must be boolean")
    if not isinstance(reason, str) or not reason.strip():
        raise AnnotationError("repair reason must be non-empty")
    call2 = value["resolved_call2"]
    if sufficient:
        if not isinstance(call2, dict):
            raise AnnotationError("sufficient repair requires resolved Call 2 output")
        try:
            normalized_call2 = validate_call2_record(call2)
        except EvidenceValidationError as exc:
            raise AnnotationError(
                f"repair returned invalid Call 2 output: {exc}"
            ) from exc
        if normalized_call2["causal_validation"]["status"] != "pass":
            raise AnnotationError("sufficient repair must return causal status=pass")
        return {
            "evidence_sufficient": True,
            "reason": reason.strip(),
            "resolved_call2": normalized_call2,
        }
    if call2 is not None:
        raise AnnotationError("insufficient repair must return null resolved_call2")
    return {
        "evidence_sufficient": False,
        "reason": reason.strip(),
        "resolved_call2": None,
    }


def _validate_sequence_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"issues"}:
        raise AnnotationError("sequence audit output must contain only issues")
    issues = value["issues"]
    if not isinstance(issues, list):
        raise AnnotationError("sequence audit issues must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {
            "target_type",
            "target_id",
            "reason",
        }:
            raise AnnotationError("sequence audit issue has unexpected fields")
        target_type = issue["target_type"]
        target_id = issue["target_id"]
        reason = issue["reason"]
        if (
            target_type not in {"unit", "boundary"}
            or not isinstance(target_id, str)
            or not target_id.strip()
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise AnnotationError("sequence audit issue fields must be non-empty")
        key = (target_type, target_id)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "target_type": target_type,
                "target_id": target_id.strip(),
                "reason": reason.strip(),
            }
        )
    return {"issues": normalized}


def _openai_tool_call(
    *,
    client: Any,
    model: str,
    instructions: str,
    content: list[dict[str, Any]],
    tool_name: str,
    description: str,
    schema: dict[str, Any],
    role: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], ProviderTrace]:
    model_options = (
        {"reasoning": {"effort": "none"}} if model.startswith("deepseek-") else {}
    )
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": content}],
        tools=[
            {
                "type": "function",
                "name": tool_name,
                "description": description,
                "parameters": schema,
                "strict": True,
            }
        ],
        tool_choice={"type": "function", "name": tool_name},
        max_output_tokens=max_output_tokens,
        **model_options,
    )
    for item in response.output:
        if (
            getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == tool_name
        ):
            return _json_arguments(item.arguments, "OpenAI", role), _trace(
                response, model
            )
    raise AnnotationError(f"OpenAI response did not contain the {role} tool call")


def _anthropic_tool_call(
    *,
    client: Any,
    model: str,
    max_tokens: int,
    system: str,
    content: list[dict[str, Any]],
    tool_name: str,
    description: str,
    schema: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], ProviderTrace]:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[
            {
                "name": tool_name,
                "description": description,
                "input_schema": schema,
                "strict": True,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == tool_name
        ):
            return _json_arguments(block.input, "Anthropic", role), _trace(
                response, model
            )
    raise AnnotationError(f"Anthropic response did not contain the {role} tool call")


class OpenAIBackend:
    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 300.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if timeout_s <= 0 or max_retries < 0:
            raise ValueError("provider timeout/retries are invalid")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the providers extra to use OpenAI") from exc
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
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
                    previous_motion_summary=request.previous_motion_summary,
                    gripper_state=request.gripper_state.prompt_text(),
                ),
            },
            *_openai_image_content(_inspection_images(request)),
        ]
        raw, trace = _openai_tool_call(
            client=self.client,
            model=self.model,
            instructions=INSPECTION_SYSTEM_PROMPT,
            content=content,
            tool_name=INSPECTION_TOOL_NAME,
            description="Locate an optional cam_high detail region.",
            schema=INSPECTION_SCHEMA,
            role="inspection",
            max_output_tokens=INSPECTION_MAX_OUTPUT_TOKENS,
        )
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
            trace=trace,
        )

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        images = _evidence_images(request)
        content = [
            *_openai_image_content(images),
            {
                "type": "input_text",
                "text": call2_context_prompt(
                    motion_summary=request.motion_summary,
                    before_boundary_observation=request.before_boundary_observation,
                    after_boundary_observation=request.after_boundary_observation,
                    gripper_state=request.gripper_state.prompt_text(),
                ),
            },
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
                    before_boundary_supplied=(
                        request.before_boundary_observation is not None
                    ),
                    after_boundary_supplied=(
                        request.after_boundary_observation is not None
                    ),
                ),
            }
        )
        raw, trace = _openai_tool_call(
            client=self.client,
            model=self.model,
            instructions=SYSTEM_PROMPT,
            content=content,
            tool_name=TOOL_NAME,
            description="Compile one strict Boundary transition record.",
            schema=EVIDENCE_SCHEMA,
            role="evidence",
            max_output_tokens=EVIDENCE_MAX_OUTPUT_TOKENS,
        )
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
            trace=trace,
        )

    def repair(self, request: RepairRequest) -> RepairResult:
        content = [
            {
                "type": "input_text",
                "text": repair_user_prompt(
                    document_id=request.document_id,
                    unit_id=request.unit_id,
                    task_instruction=request.task_instruction,
                    issue_reason=request.issue_reason,
                    call1=request.call1_motion_summary,
                    call2=request.call2,
                    boundary_context=request.boundary_context,
                    required_boundary_replacements=(
                        request.required_boundary_replacements
                    ),
                    gripper_state=request.gripper_state.prompt_text(),
                ),
            },
            *_openai_image_content(_repair_images(request)),
        ]
        raw, trace = _openai_tool_call(
            client=self.client,
            model=self.model,
            instructions=REPAIR_SYSTEM_PROMPT,
            content=content,
            tool_name=REPAIR_TOOL_NAME,
            description="Resolve one transition inconsistency.",
            schema=REPAIR_SCHEMA,
            role="repair",
            max_output_tokens=REPAIR_MAX_OUTPUT_TOKENS,
        )
        return RepairResult(
            _validate_repair_result(raw),
            self.provider,
            self.model,
            trace=trace,
        )

    def audit_sequence(self, request: SequenceAuditRequest) -> SequenceAuditResult:
        raw, trace = _openai_tool_call(
            client=self.client,
            model=self.model,
            instructions=SEQUENCE_AUDIT_SYSTEM_PROMPT,
            content=[
                {
                    "type": "input_text",
                    "text": sequence_audit_user_prompt(
                        task_instruction=request.task_instruction,
                        canonical_sequence=request.canonical_sequence,
                    ),
                }
            ],
            tool_name=SEQUENCE_AUDIT_TOOL_NAME,
            description="List unresolved sequence issues.",
            schema=SEQUENCE_AUDIT_SCHEMA,
            role="sequence audit",
            max_output_tokens=SEQUENCE_AUDIT_MAX_OUTPUT_TOKENS,
        )
        return SequenceAuditResult(
            _validate_sequence_audit(raw),
            self.provider,
            self.model,
            trace=trace,
        )


class AnthropicBackend:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 300.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if timeout_s <= 0 or max_retries < 0:
            raise ValueError("provider timeout/retries are invalid")
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Install the providers extra to use Anthropic"
                ) from exc
            client = Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
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
                    previous_motion_summary=request.previous_motion_summary,
                    gripper_state=request.gripper_state.prompt_text(),
                ),
            },
            *_anthropic_image_content(_inspection_images(request)),
        ]
        raw, trace = _anthropic_tool_call(
            client=self.client,
            model=self.model,
            max_tokens=INSPECTION_MAX_OUTPUT_TOKENS,
            system=INSPECTION_SYSTEM_PROMPT,
            content=content,
            tool_name=INSPECTION_TOOL_NAME,
            description="Locate an optional cam_high detail region.",
            schema=INSPECTION_SCHEMA,
            role="inspection",
        )
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
            trace=trace,
        )

    def annotate(self, request: EvidenceRequest) -> EvidenceResult:
        images = _evidence_images(request)
        content: list[dict[str, Any]] = [
            *_anthropic_image_content(images),
            {
                "type": "text",
                "text": call2_context_prompt(
                    motion_summary=request.motion_summary,
                    before_boundary_observation=request.before_boundary_observation,
                    after_boundary_observation=request.after_boundary_observation,
                    gripper_state=request.gripper_state.prompt_text(),
                ),
            },
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
                    before_boundary_supplied=(
                        request.before_boundary_observation is not None
                    ),
                    after_boundary_supplied=(
                        request.after_boundary_observation is not None
                    ),
                ),
            }
        )
        raw, trace = _anthropic_tool_call(
            client=self.client,
            model=self.model,
            max_tokens=EVIDENCE_MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            content=content,
            tool_name=TOOL_NAME,
            description="Compile one strict Boundary transition record.",
            schema=EVIDENCE_SCHEMA,
            role="evidence",
        )
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
            trace=trace,
        )

    def repair(self, request: RepairRequest) -> RepairResult:
        content = [
            {
                "type": "text",
                "text": repair_user_prompt(
                    document_id=request.document_id,
                    unit_id=request.unit_id,
                    task_instruction=request.task_instruction,
                    issue_reason=request.issue_reason,
                    call1=request.call1_motion_summary,
                    call2=request.call2,
                    boundary_context=request.boundary_context,
                    required_boundary_replacements=(
                        request.required_boundary_replacements
                    ),
                    gripper_state=request.gripper_state.prompt_text(),
                ),
            },
            *_anthropic_image_content(_repair_images(request)),
        ]
        raw, trace = _anthropic_tool_call(
            client=self.client,
            model=self.model,
            max_tokens=REPAIR_MAX_OUTPUT_TOKENS,
            system=REPAIR_SYSTEM_PROMPT,
            content=content,
            tool_name=REPAIR_TOOL_NAME,
            description="Resolve one transition inconsistency.",
            schema=REPAIR_SCHEMA,
            role="repair",
        )
        return RepairResult(
            _validate_repair_result(raw),
            self.provider,
            self.model,
            trace=trace,
        )

    def audit_sequence(self, request: SequenceAuditRequest) -> SequenceAuditResult:
        raw, trace = _anthropic_tool_call(
            client=self.client,
            model=self.model,
            max_tokens=SEQUENCE_AUDIT_MAX_OUTPUT_TOKENS,
            system=SEQUENCE_AUDIT_SYSTEM_PROMPT,
            content=[
                {
                    "type": "text",
                    "text": sequence_audit_user_prompt(
                        task_instruction=request.task_instruction,
                        canonical_sequence=request.canonical_sequence,
                    ),
                }
            ],
            tool_name=SEQUENCE_AUDIT_TOOL_NAME,
            description="List unresolved sequence issues.",
            schema=SEQUENCE_AUDIT_SCHEMA,
            role="sequence audit",
        )
        return SequenceAuditResult(
            _validate_sequence_audit(raw),
            self.provider,
            self.model,
            trace=trace,
        )


def make_backends(
    provider: str,
    model: str | None,
    *,
    timeout_s: float = 300.0,
    max_retries: int = 2,
) -> tuple[InspectionBackend, EvidenceBackend, RepairBackend]:
    if provider == "mock":
        return MockInspectionBackend(), MockEvidenceBackend(), MockRepairBackend()
    if not model:
        raise ValueError(f"--model is required for provider {provider!r}")
    if provider == "openai":
        backend = OpenAIBackend(
            model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        return backend, backend, backend
    if provider == "anthropic":
        backend = AnthropicBackend(
            model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        return backend, backend, backend
    raise ValueError(f"Unknown provider: {provider}")
