# Evidence and Inspection Prompt

The harness uses a progressive two-call protocol. Call 1 receives one overview,
one keyframe sheet per camera, and measured left/right gripper aperture aligned
with the keyframe sheet, with no task text. It produces one concise,
task-blind Motion Summary sentence without repeating the static scene, plus a
bounded `cam_high` ROI. Call 2 receives that Motion Summary,
six labeled original-resolution Boundary images, the same gripper samples, any
existing shared Boundary descriptions, required detail, and the Task Instruction. It does not receive
overview or keyframe sheets.

Call 2 serializes visual evidence first: Boundary images and detail,
followed by the Call 1 Motion Summary and existing Boundary text, with the Task
Instruction last. This reduces anchoring on the fallible Call 1 draft.

Call 1 also receives at most the immediately preceding accepted task-blind Motion
Summary as fallible continuity context. The previous final frame and current first
frame share one temporal boundary. Current images remain authoritative, and no
task-conditioned interpretation or longer accumulated history enters Call 1.

The Call 1 `interaction_window` is always meaningful. It preserves the visible
precondition, primary change, and immediate outcome, or the most informative
motion interval when no localized interaction is resolved. Call 1 always selects
one fixed `cam_high` region that preserves the context needed to interpret it.

All image labels include view and Evidence Unit frame metadata. The Call 1 Motion
Summary and gripper samples are passed to Call 2 as temporal evidence, not an authoritative
conclusion. Call 2 revises it using the task context, high-resolution Boundary
images, existing Boundary descriptions, and detail; that final Motion
Summary is stored in canonical evidence. Measured samples remain compiler inputs
and optional debug artifacts rather than document fields. Debug mode retains the Call 2 result, automatic
repair attempts, and final decision; normal mode deletes intermediate media.

Call 2 creates one three-view static description for each previously undescribed
Boundary State. If a canonical Boundary description already exists, Call 2 returns
`null` instead of duplicating or independently validating it. It separately describes
detail and produces a one-sentence revised Motion Summary, Action Description, and Task
Role. Complete documents then receive the sole semantic Sequence Audit. Audit issues
are grouped by owner Unit and enter bounded Targeted Repair; unresolved issues
quarantine the whole document without waiting for human intervention.

Provider output budgets are stage-specific safety ceilings rather than semantic
validators: 768 tokens for Call 1, 1200 for Call 2, 2048 for Targeted
Repair, and 1024 for Sequence Audit. The prompts determine concision; a
response that reaches a hard ceiling is treated as incomplete provider output.

Both calls share one Camera Authority Contract. `cam_high [FIXED_GLOBAL]` is a
fixed elevated oblique view and is the primary source for global scene layout,
entity position, spatial relations, scene state, and displacement. Wrist cameras
move with their grippers and are the primary source for local identity,
end-effector configuration, proximity, contact, and local interaction state.
Both calls combine measured gripper state with the visual evidence to determine
the gripper-object interaction. Numerical state values remain reasoning inputs
and are not quoted or reproduced in generated descriptions.
Grasp, hold, release, and contact claims require direct supporting evidence from
the corresponding wrist camera. Each key Atomic Action Claim must map to visual
evidence from an appropriate view; unsupported claims are weakened or removed.
Both calls maintain one physically coherent account of persistent entities and
their relations to the end effectors; unsupported state changes remain uncertain.
The Action Description is a task-grounded restatement of the finalized Motion
Summary and cannot add physical claims beyond it. Sequence Audit applies the same
continuity and entailment constraints across adjacent Boundaries and Units.
Camera authority is not infallibility; occlusion and unresolved evidence remain
explicitly uncertain. Apparent wrist-image motion is not global entity motion.
Every image label uses:

```text
EVIDENCE=<role> | VIEW=<camera id> | CAMERA_ROLE=<authority id> | ...
```
