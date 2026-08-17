from __future__ import annotations

import copy
import re
from typing import Any


EVIDENCE_SCHEMA_VERSION = "video-harness.evidence.v1"

CHANGE_STATUSES = (
    "changed",
    "no_task_relevant_change",
    "insufficient_visual_evidence",
)
VISUAL_SUPPORT = ("clear", "ambiguous", "insufficient")
ENTITY_ROLES = (
    "manipulated_object",
    "target_object",
    "target_receptacle",
    "tool",
    "support_surface",
    "context_object",
    "unknown",
)
VISIBLE_IN = ("before", "after", "both")
GROUNDING_SOURCES = ("visual_only", "visual_plus_task")
OPERATION_LABELS = (
    "approach",
    "grasp",
    "release",
    "move",
    "place",
    "insert",
    "remove",
    "open",
    "close",
    "press",
    "rotate",
    "pour",
    "align",
    "arrange",
    "stack",
    "sweep",
)
OPERATION_SUPPORT = (
    "visible_interaction",
    "endpoint_change",
    "endpoint_plus_task_context",
)
END_EFFECTORS = ("left", "right", "both", "none_visible", "uncertain")
TASK_RELEVANCE = ("relevant", "incidental", "uncertain")
VISIBILITY_LIMITS = (
    "motion_path",
    "force",
    "precise_pose",
    "grasp_contact",
    "occluded_state",
    "object_identity",
)
BASE_VISIBILITY_LIMITS = frozenset(("motion_path", "force", "precise_pose"))


class EvidenceValidationError(ValueError):
    """Raised when provider output violates the canonical evidence contract."""


_CONTROL_DETAIL = re.compile(
    r"(?:\b(?:trajectory|waypoint|joint value|velocity|speed)\b|"
    r"\b[xyz]\s*[=:]|\b\d+(?:\.\d+)?\s*(?:px|pixel|degree|meter|metre|cm|mm)\b)",
    flags=re.IGNORECASE,
)


def _exact_object(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EvidenceValidationError(f"{field} must have exactly {sorted(keys)}, got {actual}")
    return value


def _enum(value: Any, field: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise EvidenceValidationError(f"{field} must be one of {choices}, got {value!r}")
    return value


def _plain_text(
    value: Any,
    field: str,
    *,
    min_words: int,
    max_words: int,
    nullable: bool = False,
) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise EvidenceValidationError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise EvidenceValidationError(f"{field} is empty")
    if "\n" in text or "```" in text or text.startswith(("- ", "* ", "#")):
        raise EvidenceValidationError(f"{field} must be plain text")
    if len(re.findall(r"[.!?](?=\s|$)", text)) > 1:
        raise EvidenceValidationError(f"{field} must contain at most one sentence")
    words = re.findall(r"\b[\w'-]+\b", text)
    if not min_words <= len(words) <= max_words:
        raise EvidenceValidationError(
            f"{field} must contain {min_words}-{max_words} words, got {len(words)}"
        )
    if _CONTROL_DETAIL.search(text):
        raise EvidenceValidationError(f"{field} contains forbidden control details")
    return text


def validate_evidence_record(value: Any) -> dict[str, Any]:
    """Validate and normalize the immutable semantic payload returned by a VLM."""

    record = _exact_object(
        value,
        "evidence",
        {
            "change_status",
            "visual_observation",
            "entities",
            "operation_hint",
            "visible_end_effector",
            "task_relevance",
            "visibility_limits",
        },
    )
    change_status = _enum(record["change_status"], "change_status", CHANGE_STATUSES)

    observation = _exact_object(
        record["visual_observation"],
        "visual_observation",
        {"before", "after", "change", "support"},
    )
    support = _enum(observation["support"], "visual_observation.support", VISUAL_SUPPORT)
    before = _plain_text(
        observation["before"], "visual_observation.before", min_words=3, max_words=35, nullable=True
    )
    after = _plain_text(
        observation["after"], "visual_observation.after", min_words=3, max_words=35, nullable=True
    )
    change = _plain_text(
        observation["change"], "visual_observation.change", min_words=4, max_words=35, nullable=True
    )

    if change_status == "changed":
        if before is None or after is None or change is None or support == "insufficient":
            raise EvidenceValidationError(
                "changed evidence requires before, after, change, and non-insufficient support"
            )
    elif change_status == "no_task_relevant_change":
        if before is None or after is None or change is not None or support == "insufficient":
            raise EvidenceValidationError(
                "no_task_relevant_change requires before/after, null change, and visible support"
            )
    elif change is not None or support != "insufficient":
        raise EvidenceValidationError(
            "insufficient_visual_evidence requires null change and insufficient support"
        )

    entities = record["entities"]
    if not isinstance(entities, list) or len(entities) > 5:
        raise EvidenceValidationError("entities must be a list with at most five items")
    normalized_entities: list[dict[str, Any]] = []
    for index, item in enumerate(entities):
        entity = _exact_object(
            item,
            f"entities[{index}]",
            {"name", "visual_description", "role", "visible_in", "grounding", "support"},
        )
        normalized_entities.append(
            {
                "name": _plain_text(entity["name"], f"entities[{index}].name", min_words=1, max_words=8),
                "visual_description": _plain_text(
                    entity["visual_description"],
                    f"entities[{index}].visual_description",
                    min_words=2,
                    max_words=24,
                ),
                "role": _enum(entity["role"], f"entities[{index}].role", ENTITY_ROLES),
                "visible_in": _enum(
                    entity["visible_in"], f"entities[{index}].visible_in", VISIBLE_IN
                ),
                "grounding": _enum(
                    entity["grounding"], f"entities[{index}].grounding", GROUNDING_SOURCES
                ),
                "support": _enum(entity["support"], f"entities[{index}].support", ("clear", "ambiguous")),
            }
        )

    operation = record["operation_hint"]
    normalized_operation: dict[str, Any] | None = None
    if operation is not None:
        operation = _exact_object(
            operation,
            "operation_hint",
            {"label", "description", "support"},
        )
        normalized_operation = {
            "label": _enum(operation["label"], "operation_hint.label", OPERATION_LABELS),
            "description": _plain_text(
                operation["description"], "operation_hint.description", min_words=3, max_words=24
            ),
            "support": _enum(operation["support"], "operation_hint.support", OPERATION_SUPPORT),
        }
    if change_status != "changed" and normalized_operation is not None:
        raise EvidenceValidationError("operation_hint must be null unless change_status is changed")

    visible_end_effector = _enum(
        record["visible_end_effector"], "visible_end_effector", END_EFFECTORS
    )
    task_relevance = _enum(record["task_relevance"], "task_relevance", TASK_RELEVANCE)
    if change_status == "changed" and task_relevance != "relevant":
        raise EvidenceValidationError(
            "changed evidence requires task_relevance to be relevant"
        )

    limits = record["visibility_limits"]
    if not isinstance(limits, list) or any(item not in VISIBILITY_LIMITS for item in limits):
        raise EvidenceValidationError("visibility_limits contains an unsupported value")
    if len(limits) != len(set(limits)):
        raise EvidenceValidationError("visibility_limits must not contain duplicates")
    if not BASE_VISIBILITY_LIMITS.issubset(limits):
        missing = sorted(BASE_VISIBILITY_LIMITS - set(limits))
        raise EvidenceValidationError(f"visibility_limits is missing fixed endpoint limits: {missing}")
    normalized_limits = [item for item in VISIBILITY_LIMITS if item in limits]

    return {
        "change_status": change_status,
        "visual_observation": {
            "before": before,
            "after": after,
            "change": change,
            "support": support,
        },
        "entities": normalized_entities,
        "operation_hint": normalized_operation,
        "visible_end_effector": visible_end_effector,
        "task_relevance": task_relevance,
        "visibility_limits": normalized_limits,
    }


def mock_evidence_record() -> dict[str, Any]:
    return {
        "change_status": "insufficient_visual_evidence",
        "visual_observation": {
            "before": None,
            "after": None,
            "change": None,
            "support": "insufficient",
        },
        "entities": [],
        "operation_hint": None,
        "visible_end_effector": "uncertain",
        "task_relevance": "uncertain",
        "visibility_limits": [item for item in VISIBILITY_LIMITS if item in BASE_VISIBILITY_LIMITS],
    }


def evidence_is_trainable(record: Any, *, allow_ambiguous: bool = False) -> bool:
    normalized = validate_evidence_record(copy.deepcopy(record))
    if normalized["change_status"] != "changed":
        return False
    support = normalized["visual_observation"]["support"]
    return support == "clear" or (allow_ambiguous and support == "ambiguous")
