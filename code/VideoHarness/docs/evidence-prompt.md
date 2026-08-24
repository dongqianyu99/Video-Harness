# Evidence and Inspection Prompt

The harness uses a progressive two-call protocol. Call 1 receives one overview
and one keyframe sheet per camera with no task text. It produces a task-blind Motion
Summary and a bounded optional `cam_high` ROI. Call 2 receives that Motion Summary,
six labeled original-resolution Boundary images, any accepted shared Boundary
descriptions, optional detail, and the Task Instruction. It does not receive
overview or keyframe sheets.

Call 1 also receives at most the immediately preceding accepted task-blind Motion
Summary as fallible continuity context. The previous final frame and current first
frame share one temporal boundary. Current images remain authoritative, and no
task-conditioned interpretation or longer accumulated history enters Call 1.

The Call 1 `interaction_window` is always meaningful. It preserves the visible
precondition, primary change, and immediate outcome, or the most informative
motion interval when no localized interaction is resolved. A detail crop is
requested only when enlarging a fixed `cam_high` region can add evidence.

All image labels include view and Evidence Unit frame metadata. The Motion Summary is
preserved verbatim in canonical evidence and passed to Call 2 as temporal evidence,
not an authoritative conclusion. Debug mode retains the Call 2 result, automatic
repair attempts, and final decision; normal mode deletes intermediate media.

Call 2 creates one three-view static description for each previously undescribed
Boundary State. If an accepted Boundary description is supplied, Call 2 verifies
it against the corresponding images and returns `null` instead of duplicating it.
If that description materially contradicts the images, Call 2 records a concise
`boundary_conflicts` reason and requests retry; the shared Boundary becomes
ineligible until automatic Targeted Reprocessing resolves or quarantines it.
It separately describes optional detail, explains the Evidence Unit transition
and task role, and performs a causal validation. A material inconsistency enters
Targeted Reprocessing with the full temporal sheets and prior outputs. Complete
documents then receive a Sequence Audit; unresolved issues quarantine the whole
document without waiting for human intervention.

Both calls share one Camera Authority Contract. `cam_high [FIXED_GLOBAL]` is a
fixed elevated oblique view and is the primary source for global scene layout,
entity position, spatial relations, scene state, and displacement. Wrist cameras
move with their grippers and are the primary source for local identity,
end-effector configuration, proximity, contact, and local interaction state.
Camera authority is not infallibility; occlusion and unresolved evidence remain
explicitly uncertain. Apparent wrist-image motion is not global entity motion.
Every image label uses:

```text
EVIDENCE=<role> | VIEW=<camera id> | CAMERA_ROLE=<authority id> | ...
```
