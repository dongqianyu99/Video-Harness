# Evidence and Inspection Prompt

The harness uses a progressive two-call protocol. Call 1 receives one overview
and one keyframe sheet per camera with no task text. It produces one concise,
task-blind Motion Summary sentence without repeating the static scene, plus a
bounded optional `cam_high` ROI. Call 2 receives that Motion Summary,
six labeled original-resolution Boundary images, any accepted shared Boundary
descriptions, optional detail, and the Task Instruction. It does not receive
overview or keyframe sheets.

Call 2 serializes visual evidence first: Boundary images and optional detail,
followed by the Call 1 Motion Summary and accepted Boundary text, with the Task
Instruction last. This reduces anchoring on the fallible Call 1 draft.

Call 1 also receives at most the immediately preceding accepted task-blind Motion
Summary as fallible continuity context. The previous final frame and current first
frame share one temporal boundary. Current images remain authoritative, and no
task-conditioned interpretation or longer accumulated history enters Call 1.

The Call 1 `interaction_window` is always meaningful. It preserves the visible
precondition, primary change, and immediate outcome, or the most informative
motion interval when no localized interaction is resolved. A detail crop is
requested only when enlarging a fixed `cam_high` region can add evidence.

All image labels include view and Evidence Unit frame metadata. The Call 1 Motion
Summary is passed to Call 2 as fallible temporal evidence, not an authoritative
conclusion. Call 2 revises it using the task context, high-resolution Boundary
images, accepted Boundary descriptions, and optional detail; only that final
Motion Summary is stored in canonical evidence. Debug mode retains the Call 2 result, automatic
repair attempts, and final decision; normal mode deletes intermediate media.

Call 2 creates one three-view static description for each previously undescribed
Boundary State. If an accepted Boundary description is supplied, Call 2 verifies
it against the corresponding images and returns `null` instead of duplicating it.
If that description materially contradicts the images, Call 2 records a concise
`boundary_conflicts` reason and requests retry; the shared Boundary becomes
ineligible until automatic Targeted Reprocessing resolves or quarantines it.
It separately describes optional detail, produces a one-sentence revised Motion
Summary, a one-sentence Action Description, and a one-sentence Task Role, and performs a
causal validation with a one-sentence reason. A material inconsistency enters
Targeted Reprocessing with the full temporal sheets and prior outputs. Complete
documents then receive a Sequence Audit; unresolved issues quarantine the whole
document without waiting for human intervention.

Provider output budgets are stage-specific safety ceilings rather than semantic
validators: 768 tokens for Call 1, 1200 for Call 2, 2048 for Targeted
Reprocessing, and 1024 for Sequence Audit. The prompts determine concision; a
response that reaches a hard ceiling is treated as incomplete provider output.

Both calls share one Camera Authority Contract. `cam_high [FIXED_GLOBAL]` is a
fixed elevated oblique view and is the primary source for global scene layout,
entity position, spatial relations, scene state, and displacement. Wrist cameras
move with their grippers and are the primary source for local identity,
end-effector configuration, proximity, contact, and local interaction state.
Grasp, hold, release, and contact claims require direct supporting evidence from
the corresponding wrist camera. Before causal status can pass, each key Atomic
Action Claim must map to visual evidence from an appropriate view; unsupported
claims are weakened or removed, and the causal reason summarizes that mapping.
Camera authority is not infallibility; occlusion and unresolved evidence remain
explicitly uncertain. Apparent wrist-image motion is not global entity motion.
Every image label uses:

```text
EVIDENCE=<role> | VIEW=<camera id> | CAMERA_ROLE=<authority id> | ...
```
