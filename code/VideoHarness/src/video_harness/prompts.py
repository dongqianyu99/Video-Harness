from __future__ import annotations

from typing import Any

from .evidence import (
    CHANGE_STATUSES,
    END_EFFECTORS,
    ENTITY_ROLES,
    GROUNDING_SOURCES,
    OPERATION_LABELS,
    OPERATION_SUPPORT,
    TASK_RELEVANCE,
    VISIBILITY_LIMITS,
    VISIBLE_IN,
    VISUAL_SUPPORT,
)


PROMPT_VERSION = "video-harness.evidence.v1"
TOOL_NAME = "record_transition_evidence"


def _nullable_string(description: str) -> dict[str, Any]:
    return {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": description,
    }


EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "change_status",
        "visual_observation",
        "entities",
        "operation_hint",
        "visible_end_effector",
        "task_relevance",
        "visibility_limits",
    ],
    "properties": {
        "change_status": {
            "type": "string",
            "enum": list(CHANGE_STATUSES),
            "description": (
                "Whether the sampled endpoints show a task-relevant visible state change, "
                "show no such change, or are too ambiguous to judge."
            ),
        },
        "visual_observation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["before", "after", "change", "support"],
            "properties": {
                "before": _nullable_string(
                    "One short sentence describing only the task-relevant state visible in BEFORE."
                ),
                "after": _nullable_string(
                    "One short sentence describing only the task-relevant state visible in AFTER."
                ),
                "change": _nullable_string(
                    "One short present-state sentence describing the visible endpoint difference, "
                    "not an unobserved process. Null unless change_status is changed."
                ),
                "support": {
                    "type": "string",
                    "enum": list(VISUAL_SUPPORT),
                    "description": "Visibility strength of the endpoint observation, not a probability.",
                },
            },
        },
        "entities": {
            "type": "array",
            "description": "At most five entities needed to understand this local transition.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "visual_description",
                    "role",
                    "visible_in",
                    "grounding",
                    "support",
                ],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A short entity name; use a generic name when identity is unclear.",
                    },
                    "visual_description": {
                        "type": "string",
                        "description": "A concise visible appearance/location description for grounding.",
                    },
                    "role": {"type": "string", "enum": list(ENTITY_ROLES)},
                    "visible_in": {"type": "string", "enum": list(VISIBLE_IN)},
                    "grounding": {
                        "type": "string",
                        "enum": list(GROUNDING_SOURCES),
                        "description": "Whether the role/name is visual-only or uses task context.",
                    },
                    "support": {"type": "string", "enum": ["clear", "ambiguous"]},
                },
            },
        },
        "operation_hint": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "description", "support"],
                    "properties": {
                        "label": {"type": "string", "enum": list(OPERATION_LABELS)},
                        "description": {
                            "type": "string",
                            "description": (
                                "One short local operation hypothesis. It is metadata, not a visual fact."
                            ),
                        },
                        "support": {"type": "string", "enum": list(OPERATION_SUPPORT)},
                    },
                },
            ],
            "description": "A bounded operation hypothesis, or null when endpoints do not support one.",
        },
        "visible_end_effector": {
            "type": "string",
            "enum": list(END_EFFECTORS),
            "description": (
                "Which end effector is directly visible near the relevant endpoint state; "
                "use uncertain rather than inferring which arm caused the change."
            ),
        },
        "task_relevance": {
            "type": "string",
            "enum": list(TASK_RELEVANCE),
            "description": "Relation to the task instruction, never a task-success judgment.",
        },
        "visibility_limits": {
            "type": "array",
            "items": {"type": "string", "enum": list(VISIBILITY_LIMITS)},
            "description": (
                "Include motion_path, force, and precise_pose for every endpoint pair, plus any "
                "other explicitly applicable visibility limits."
            ),
        },
    },
}


SYSTEM_PROMPT = f"""\
You are the Video Harness evidence compiler. You receive exactly two temporally
ordered sampled endpoints from one interval of a successful robot demonstration:
BEFORE and AFTER. You also receive a coarse task instruction as context.

Your job is not to caption everything, reconstruct the hidden motion, or write a
plan. Compile a small, evidence-grounded record that a downstream robot policy can
render in different ways without asking a VLM again. Return exactly one call to
{TOOL_NAME}; output no explanation or reasoning outside the tool.

Follow this evidence hierarchy:

1. visual_observation contains only facts and relations directly supported by the
   two images. BEFORE and AFTER may describe task-relevant visible state. The
   change sentence describes an endpoint state difference, not the unseen process.
2. entities contains only the few objects needed for this interval. Mark an
   entity as manipulated_object, target_object, target_receptacle, tool,
   support_surface, context_object, or unknown. If the task instruction helps name
   or assign the role, grounding must be visual_plus_task. Use a generic visible
   description when identity is uncertain.
3. operation_hint is an explicitly bounded hypothesis. Use null rather than guess.
   Its support must reveal whether an interaction is visible, inferred only from
   the endpoint change, or additionally disambiguated by the task instruction.
4. visible_end_effector records which arm is directly visible near the relevant
   endpoint state: left, right, both, none_visible, or uncertain. It does not
   establish which arm caused the unseen transition.
5. task_relevance may be relevant, incidental, or uncertain. It never means the
   whole task or subtask succeeded.
6. visibility_limits always includes motion_path, force, and precise_pose because
   two sampled endpoints cannot provide them. Add grasp_contact, occluded_state,
   or object_identity only when applicable.

Set change_status to changed only when a task-relevant endpoint difference is
visible. Use no_task_relevant_change when both states are visible but no relevant
change occurs. Use insufficient_visual_evidence when occlusion, blur, or ambiguity
prevents a reliable judgment. The visual support must agree with that status.

The images are authoritative evidence. The task instruction may focus attention
and disambiguate names or operation labels, but it must never create a visual fact.
Do not claim task success. Do not invent hidden substeps, future steps, causal
intent, exact paths, grasp force or quality, trajectories, velocities,
coordinates, distances, angles, 3-D positions, joint values, or precise poses.
Keep every free-text field concise, plain English, and limited to one sentence.
"""


def user_prompt(
    *,
    task_instruction: str,
    camera_key: str,
    unit_id: str,
    before_frame: int,
    after_frame: int,
) -> str:
    return f"""\
Task instruction (context only; it is not visual evidence):
{task_instruction}

Camera: {camera_key}
Transition unit: {unit_id}
BEFORE episode-local frame: {before_frame}
AFTER episode-local frame: {after_frame}

Compile the strict transition evidence record from the labeled sampled endpoints.
"""
