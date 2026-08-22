from __future__ import annotations

import json
from typing import Any

from .camera_contract import CAMERA_VIEWS, camera_spec, system_prompt_camera_contract
from .evidence import (
    CAUSAL_VALIDATION_STATUSES,
    DETAIL_REASONS,
)


PROMPT_VERSION = "video-harness.evidence"
INSPECTION_PROMPT_VERSION = "video-harness.inspection"
TOOL_NAME = "record_transition_evidence"
INSPECTION_TOOL_NAME = "locate_temporal_detail"
CAMERA_CONTRACT = system_prompt_camera_contract()


def _view_string_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(CAMERA_VIEWS),
        "properties": {
            view: {
                "type": "string",
                "description": (
                    f"A concise static-state description grounded in the {view} "
                    f"[{camera_spec(view).role_id}] endpoint and limited to that "
                    "camera's evidence authority."
                ),
            }
            for view in CAMERA_VIEWS
        },
    }


EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "endpoint_observation",
        "detail_observation",
        "unit_interpretation",
        "causal_validation",
    ],
    "properties": {
        "endpoint_observation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["before", "after"],
            "properties": {
                "before": _view_string_schema(),
                "after": _view_string_schema(),
            },
        },
        "detail_observation": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "A direct description of the optional cam_high detail sheet, or null when no detail sheet is supplied.",
        },
        "unit_interpretation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action_description", "task_role"],
            "properties": {
                "action_description": {
                    "type": "string",
                    "description": "What the robot physically does during this Unit, grounded in motion, endpoints, and optional detail.",
                },
                "task_role": {
                    "type": "string",
                    "description": "What this Unit contributes to the task, without claiming more progress than the evidence supports.",
                },
            },
        },
        "causal_validation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "reason"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(CAUSAL_VALIDATION_STATUSES),
                },
                "reason": {
                    "type": "string",
                    "description": "A concise explanation of whether the interpretation obeys basic causal and physical constraints.",
                },
            },
        },
    },
}


INSPECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "motion_summary",
        "interaction_window",
        "needs_detail",
        "detail_request",
    ],
    "properties": {
        "motion_summary": {
            "type": "string",
            "description": "One concise task-blind description of the visible geometric and temporal motion facts.",
        },
        "interaction_window": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start_frame", "end_frame"],
            "properties": {
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer"},
            },
        },
        "needs_detail": {"type": "boolean"},
        "detail_request": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x_min", "y_min", "x_max", "y_max", "reason"],
                    "properties": {
                        "x_min": {"type": "number"},
                        "y_min": {"type": "number"},
                        "x_max": {"type": "number"},
                        "y_max": {"type": "number"},
                        "reason": {
                            "type": "string",
                            "enum": list(DETAIL_REASONS),
                        },
                    },
                },
            ]
        },
    },
}


INSPECTION_SYSTEM_PROMPT = f"""You are the Video Harness task-blind temporal motion analyst. You receive synchronized 5×5 overview sheets and higher-resolution 2×3 stage sheets from cam_high, cam_left_wrist, and cam_right_wrist. All sheets cover the same Unit in chronological order.

{CAMERA_CONTRACT}

Describe the visible geometric and temporal motion facts in one concise motion_summary. Explicitly distinguish wrist-camera ego-motion from physical object motion. Focus on arm and gripper motion, object motion, approach, contact, grasp, release, and changes in spatial relation. Do not infer task intent, success, future steps, trajectories, velocities, forces, coordinates, or precise poses.

Decide whether the final evidence compiler needs one fixed high-resolution crop from cam_high to judge a small or occluded interaction. Request no detail when the supplied sheets already show the relevant interaction. If detail is needed, return one normalized cam_high ROI and the smallest contiguous frame interval that contains the interaction. The ROI must include the end effector, relevant object, and enough context to interpret their relation.

Return exactly one locate_temporal_detail tool call and no other text."""


SYSTEM_PROMPT = f"""You are the Video Harness task-conditioned evidence compiler and lightweight causal validator. You receive a task-blind motion_summary from Call 1, synchronized original-resolution BEFORE and AFTER endpoints from cam_high, cam_left_wrist, and cam_right_wrist, an optional cam_high detail sheet, and a coarse task instruction.

{CAMERA_CONTRACT}

First describe the static state visible in every camera endpoint without collapsing the three views into one imagined camera. Use cam_high to ground global positions, object counts, pad occupancy, and scene-state changes; use wrist endpoints to resolve local identity, gripper closure, contact, grasp, and release. If a detail sheet is supplied, describe only the fine interaction evidence that it directly shows. Then combine those static observations, the optional detail observation, the Call 1 motion_summary, and the task instruction to explain what the robot physically does in this Unit and what role the Unit plays in the task. Treat motion_summary as a useful task-blind temporal observation, not as unquestionable ground truth; correct or qualify it when the endpoint or detail evidence disagrees.

Finally perform a permissive causal validation of your own interpretation. Use status=retry only for a clear violation of basic causal or physical logic, such as claiming object displacement with no compatible interaction, or claiming global movement solely from apparent motion or visibility changes in a moving wrist camera while cam_high remains inconsistent with that claim. Use status=pass when the interpretation is physically plausible even if fine details remain uncertain. Do not retry merely because the exact motion, object identity, or task role is incomplete.

Do not invent hidden substeps, future states, success, trajectories, velocities, coordinates, forces, joint values, or precise poses. Return exactly one record_transition_evidence tool call and no other text."""


def inspection_user_prompt(
    *,
    document_id: str,
    unit_id: str,
    episode_start_frame: int,
    episode_end_frame: int,
) -> str:
    return (
        f"Document={document_id} | Unit={unit_id} | FPS=25 | "
        f"episode_frames={episode_start_frame}..{episode_end_frame}. "
        "Images are authoritative and labels identify view and Unit-local frame. "
        "Describe the task-blind temporal motion and return the strict localization tool call."
    )


def call2_context_prompt(*, motion_summary: str) -> str:
    return (
        "Call 1 task-blind motion summary (temporal evidence to verify, not an "
        f"authoritative conclusion): {motion_summary}"
    )


def evidence_user_prompt(
    *,
    document_id: str,
    unit_id: str,
    episode_start_frame: int,
    episode_end_frame: int,
    task_instruction: str,
    detail_supplied: bool,
    previous_attempt: dict[str, Any] | None,
) -> str:
    retry_context = ""
    if previous_attempt is not None:
        retry_context = (
            " Previous Call 2 interpretation was marked retry. Re-evaluate the "
            "same evidence and correct the causal inconsistency without inventing "
            "new events. Previous output: "
            + json.dumps(previous_attempt, ensure_ascii=False, sort_keys=True)
        )
    return (
        f"Document={document_id} | Unit={unit_id} | FPS=25 | "
        f"episode_frames={episode_start_frame}..{episode_end_frame}. "
        f"Detail sheet supplied: {'yes' if detail_supplied else 'no'}. "
        f"Task instruction (context for interpretation, not visual evidence): {task_instruction}. "
        "Describe all six endpoint states, the optional detail, the Unit action, "
        "its task role, and perform the permissive causal validation."
        + retry_context
    )
