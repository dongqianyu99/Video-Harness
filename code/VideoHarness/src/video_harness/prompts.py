from __future__ import annotations

import json
from typing import Any

from .camera_contract import CAMERA_VIEWS, camera_spec, system_prompt_camera_contract
PROMPT_VERSION = "video-harness.evidence.v9"
INSPECTION_PROMPT_VERSION = "video-harness.inspection.v7"
TOOL_NAME = "record_transition_evidence"
INSPECTION_TOOL_NAME = "locate_temporal_detail"
REPAIR_TOOL_NAME = "resolve_transition_repair"
SEQUENCE_AUDIT_TOOL_NAME = "audit_sequence_consistency"
REPAIR_PROMPT_VERSION = "video-harness.repair.v10"
SEQUENCE_AUDIT_PROMPT_VERSION = "video-harness.sequence-audit.v6"
CAMERA_CONTRACT = system_prompt_camera_contract()
ACTION_EVIDENCE_CONTRACT = """Action-evidence contract:
- Claims of grasp, hold, release, or contact require direct supporting visual evidence from the corresponding wrist camera.
- Decompose motion_summary and action_description into their key atomic actions and find corresponding visual evidence from the appropriate camera view for each action.
- Weaken or remove any action that lacks visual support."""
PHYSICAL_CONTINUITY_CONTRACT = """Maintain one physically coherent account of persistent entities and their relations to the end effectors across the supplied evidence. Any inferred state change must be supported by the temporal and multimodal evidence; otherwise preserve uncertainty."""
GRIPPER_STATE_CONTRACT = """Use the synchronized gripper state together with the visual evidence to determine the gripper-object interaction throughout the Evidence Unit. Do not speculate about, quote, or reproduce any numerical gripper-state values in the output; describe the interaction only in qualitative terms."""

REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_sufficient",
        "reason",
        "resolved_call2",
    ],
    "properties": {
        "evidence_sufficient": {"type": "boolean"},
        "reason": {"type": "string"},
        "resolved_call2": {"type": "null"},
    },
}
SEQUENCE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_type", "target_id", "reason"],
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": ["unit", "boundary"],
                    },
                    "target_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


def _view_string_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(CAMERA_VIEWS),
        "properties": {
            view: {
                "type": "string",
                "description": (
                    f"Exactly one concise static-state sentence grounded in the {view} "
                    f"[{camera_spec(view).role_id}] Boundary view and limited to that "
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
        "motion_summary",
        "before_boundary_observation",
        "after_boundary_observation",
        "detail_observation",
        "unit_interpretation",
    ],
    "properties": {
        "motion_summary": {
            "type": "string",
            "description": "Exactly one concise sentence revising Call 1's task-blind motion summary using the task context, high-resolution BEFORE and AFTER Boundary images, existing Boundary descriptions, and detail evidence.",
        },
        "before_boundary_observation": {
            "anyOf": [_view_string_schema(), {"type": "null"}],
            "description": "A new static description of the synchronized BEFORE Boundary State when no accepted description was supplied, otherwise null.",
        },
        "after_boundary_observation": {
            "anyOf": [_view_string_schema(), {"type": "null"}],
            "description": "A new static description of the synchronized AFTER Boundary State when no accepted description was supplied, otherwise null.",
        },
        "detail_observation": {
            "type": "string",
            "description": "A direct description of the mandatory cam_high detail sheet.",
        },
        "unit_interpretation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action_description", "task_role"],
            "properties": {
                "action_description": {
                    "type": "string",
                    "description": "Exactly one concise, task-grounded restatement of the finalized motion_summary without adding physical claims beyond it.",
                },
                "task_role": {
                    "type": "string",
                    "description": "Exactly one concise sentence stating what this physical action contributes to the task without overstating progress.",
                },
            },
        },
    },
}
REPAIR_SCHEMA["properties"]["resolved_call2"] = {
    "anyOf": [EVIDENCE_SCHEMA, {"type": "null"}]
}


INSPECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "motion_summary",
        "interaction_window",
        "detail_request",
    ],
    "properties": {
        "motion_summary": {
            "type": "string",
            "description": "Exactly one concise, task-blind sentence describing the visible qualitative motion, interaction, and final persistent state across the Evidence Unit without repeating the static scene.",
        },
        "interaction_window": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start_frame", "end_frame"],
            "properties": {
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer"},
            },
            "description": "The shortest continuous interval that preserves the precondition, primary visible change, and immediate outcome; when no localized interaction is resolved, the most informative motion interval.",
        },
        "detail_request": {
            "type": "object",
            "additionalProperties": False,
            "required": ["x_min", "y_min", "x_max", "y_max"],
            "properties": {
                "x_min": {"type": "number"},
                "y_min": {"type": "number"},
                "x_max": {"type": "number"},
                "y_max": {"type": "number"},
            },
        },
    },
}


INSPECTION_SYSTEM_PROMPT = f"""You are the Video Harness task-blind temporal motion analyst. You receive synchronized 5×5 overview sheets and higher-resolution 2×3 keyframe sheets from cam_high, cam_left_wrist, and cam_right_wrist, plus measured gripper aperture aligned with the 2×3 sheets. All inputs cover the same Evidence Unit in chronological order. Inspect all three views and the aperture evidence before forming a conclusion.

{CAMERA_CONTRACT}

{ACTION_EVIDENCE_CONTRACT}

{PHYSICAL_CONTINUITY_CONTRACT}

Produce motion_summary as exactly one concise sentence describing the task-blind qualitative motion, interaction, and final persistent state across the Evidence Unit. Describe only the temporal progression and spatial relations needed to understand what moves or changes, while distinguishing wrist-camera ego-motion from physical motion in the scene. Do not inventory or repeat the static scene; mention an unchanged entity or view only when necessary to disambiguate the motion. A globally stable view does not rule out a meaningful local interaction. Calibrate the sentence to the evidence and preserve uncertainty instead of turning an unresolved observation into either a positive or negative claim.

{GRIPPER_STATE_CONTRACT}

Do not infer task intent, task outcome, or future behavior. Do not estimate or invent precise physical quantities; use qualitative descriptions grounded in the supplied visual evidence. Normalized image coordinates are used only to specify the detail ROI.

Always locate one meaningful interaction_window. Use the shortest continuous interval that still preserves the visible precondition, primary change, and immediate outcome. If no localized interaction can be resolved, locate the most informative motion interval instead.

Always select one cam_high ROI over the interaction_window, using the smallest region that preserves the physical context needed to interpret the observed motion.

When task-blind context from the immediately preceding Evidence Unit is supplied, treat it as a fallible continuity hypothesis. Its final frame and the current Evidence Unit's first frame refer to the shared temporal boundary. Use current visual evidence to confirm, qualify, or correct whether an interaction persists, changes, or ends; never invent continuity merely to agree with the prior summary.

Return exactly one locate_temporal_detail tool call and no other text."""


SYSTEM_PROMPT = f"""You are the Video Harness task-conditioned evidence compiler. You receive a task-blind motion_summary from Call 1, synchronized original-resolution BEFORE and AFTER Boundary images from cam_high, cam_left_wrist, and cam_right_wrist, measured gripper aperture aligned with Call 1 keyframes, existing canonical descriptions of those Boundary States when available, a cam_high detail sheet, and a coarse task instruction.

{CAMERA_CONTRACT}

{ACTION_EVIDENCE_CONTRACT}

{PHYSICAL_CONTINUITY_CONTRACT}

Describe each Boundary State once, using exactly one concise sentence per camera view. When an existing canonical Boundary description is supplied, return null for that boundary output; do not create or verify a second description of the same shared state. When no description is supplied, produce it from the corresponding images. Boundary descriptions report the visible state at that instant rather than inferring an interval event from one image. Keep the three views separate and follow their evidence authority.

Describe only the fine interaction evidence that the detail sheet directly shows. Return motion_summary as exactly one concise sentence that revises Call 1's task-blind draft using the task context, the high-resolution static BEFORE and AFTER Boundary images, existing Boundary descriptions, and detail evidence. Write action_description as exactly one concise, task-grounded restatement of the finalized motion_summary without adding physical claims beyond it. Write task_role as exactly one concise sentence stating only what that action contributes to the task. Treat Call 1 and prior text as useful but fallible evidence; correct them when the Boundary or detail evidence disagrees.

{GRIPPER_STATE_CONTRACT}

Do not invent hidden substeps, future states, success, trajectories, velocities, coordinates, forces, joint values, or precise poses. Return exactly one record_transition_evidence tool call and no other text."""


REPAIR_SYSTEM_PROMPT = f"""You are the Video Harness automatic transition repair adjudicator. A normal compilation attempt was inconsistent or a Sequence Audit identified an unexplained transition. Re-examine the supplied full temporal sheets, Boundary images, measured gripper aperture, canonical context, and prior outputs as fallible evidence.

{CAMERA_CONTRACT}

{ACTION_EVIDENCE_CONTRACT}

{PHYSICAL_CONTINUITY_CONTRACT}

Resolve only the identified Evidence Unit. If the visual evidence supports one coherent account, return evidence_sufficient=true and a complete corrected Call 2 record that follows the same motion_summary, Boundary, action_description, and task_role contract. Return a replacement Boundary observation only when the detected issue explicitly requires correcting that shared Boundary; otherwise reuse its supplied description with a null observation. If the supplied evidence cannot resolve the issue, return evidence_sufficient=false with a null resolved_call2. Do not preserve a prior claim merely for consistency, and do not invent precise physical quantities or unseen events.

{GRIPPER_STATE_CONTRACT}

Return exactly one resolve_transition_repair tool call and no other text."""


SEQUENCE_AUDIT_SYSTEM_PROMPT = """You are the Video Harness sole semantic sequence auditor. Check whether every Boundary State change is explained by its connecting Evidence Unit, whether persistent entities and their relations to the end effectors evolve coherently across adjacent Boundaries and Units, whether each action_description is entailed by its motion_summary and detail_observation, and whether each task_role matches the demonstrated task state. Report only material issues that require automatic repair. Target the Boundary whose static description needs replacement, or otherwise the specific Evidence Unit whose transition text needs repair. Each reason must describe the contradiction at its declared target. An empty issue list means the complete canonical sequence is coherent. Return exactly one audit_sequence_consistency tool call and no other text."""


def inspection_user_prompt(
    *,
    document_id: str,
    unit_id: str,
    episode_start_frame: int,
    episode_end_frame: int,
    previous_motion_summary: str | None,
    gripper_state: str,
) -> str:
    continuity_context = ""
    if previous_motion_summary is not None:
        continuity_context = (
            " Previous Evidence Unit context (task-blind and fallible; verify "
            "against the current images): " + previous_motion_summary
        )
    return (
        f"Document={document_id} | EvidenceUnit={unit_id} | FPS=25 | "
        f"episode_frames={episode_start_frame}..{episode_end_frame}. "
        "Images are authoritative and labels identify view and Evidence-Unit-local frame. "
        "Describe the task-blind temporal motion and return the strict localization tool call."
        + " "
        + gripper_state
        + continuity_context
    )


def call2_context_prompt(
    *,
    motion_summary: str,
    before_boundary_observation: dict[str, str] | None,
    after_boundary_observation: dict[str, str] | None,
    gripper_state: str,
) -> str:
    def boundary_context(
        role: str,
        observation: dict[str, str] | None,
    ) -> str:
        if observation is None:
            return (
                f"No accepted {role} Boundary description is available; create "
                f"it from the synchronized {role} images."
            )
        return (
            f"Accepted shared {role} Boundary description (verify against the "
            f"current {role} images and do not duplicate it): "
            + json.dumps(observation, ensure_ascii=False, sort_keys=True)
        )

    return (
        "Call 1 task-blind motion summary (temporal evidence to verify, not an "
        f"authoritative conclusion): {motion_summary} "
        + boundary_context("BEFORE", before_boundary_observation)
        + " "
        + boundary_context("AFTER", after_boundary_observation)
        + " "
        + gripper_state
    )


def evidence_user_prompt(
    *,
    document_id: str,
    unit_id: str,
    episode_start_frame: int,
    episode_end_frame: int,
    task_instruction: str,
    before_boundary_supplied: bool,
    after_boundary_supplied: bool,
) -> str:
    return (
        f"Document={document_id} | EvidenceUnit={unit_id} | FPS=25 | "
        f"episode_frames={episode_start_frame}..{episode_end_frame}. "
        "A cam_high detail sheet is supplied for this Evidence Unit. "
        f"Existing BEFORE Boundary description supplied: {'yes' if before_boundary_supplied else 'no'}. "
        f"Existing AFTER Boundary description supplied: {'yes' if after_boundary_supplied else 'no'}. "
        f"Task instruction (context for interpretation, not visual evidence): {task_instruction}. "
        "Return only required new Boundary descriptions and the transition interpretation."
    )


def repair_user_prompt(
    *,
    document_id: str,
    unit_id: str,
    task_instruction: str,
    issue_reason: str,
    call1: str,
    call2: dict[str, Any],
    boundary_context: str,
    required_boundary_replacements: tuple[str, ...],
    gripper_state: str,
) -> str:
    replacement_instruction = (
        " No Boundary replacement is required; return null Boundary observations."
        if not required_boundary_replacements
        else " You must regenerate these Boundary descriptions from their supplied "
        "images: " + ", ".join(required_boundary_replacements) + "."
    )
    return (
        f"Document={document_id} | EvidenceUnit={unit_id}. Automatic targeted repair. "
        f"Task instruction: {task_instruction}. Detected issue: {issue_reason}. "
        f"Bounded adjacent context: {boundary_context}. Original Call 1: {call1}. "
        "Original Call 2: "
        + json.dumps(call2, ensure_ascii=False, sort_keys=True)
        + "."
        + replacement_instruction
        + " "
        + gripper_state
        + " Reconcile only from supplied evidence; regenerate the complete corrected "
        "transition record and return the strict repair tool call."
    )


def sequence_audit_user_prompt(
    *, task_instruction: str, canonical_sequence: str
) -> str:
    return (
        f"Task instruction: {task_instruction}. Audit this complete canonical sequence. "
        f"Sequence:\n{canonical_sequence}\nReturn only the strict issue list tool call."
    )
