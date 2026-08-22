# Evidence and Inspection Prompt

The harness uses a progressive two-call protocol. Call 1 receives one overview
and one stage sheet per camera with no task text. It produces a task-blind Motion
Summary and a bounded optional `cam_high` ROI. Call 2 receives that Motion Summary,
six labeled original-resolution endpoints, optional detail, and the Task
Instruction. It does not receive overview or stage sheets.

All image labels include view and Unit frame metadata. The Motion Summary is
preserved verbatim in canonical evidence and passed to Call 2 as temporal evidence,
not an authoritative conclusion. Debug mode retains each Call 2 attempt and the
final selection; normal mode deletes intermediate media after the request.

Call 2 describes BEFORE and AFTER separately for `cam_high`, `cam_left_wrist`,
and `cam_right_wrist`, describes optional detail, explains the robot action and
the Unit's task role, and performs a permissive causal validation. A clear basic
causal contradiction triggers another Call 2 attempt; after three unsuccessful
interpretations, the last result is retained with `needs_review`.

Both calls share one Camera Authority Contract. `cam_high [FIXED_GLOBAL]` is a
fixed elevated oblique view and governs global position, pad occupancy, counts,
ordering, scene state, and displacement. Wrist cameras move with their grippers
and govern local identity, gripper opening/closing, contact, grasp, and release.
Apparent wrist-image motion is not global object motion. Every image label uses:

```text
EVIDENCE=<role> | VIEW=<camera id> | CAMERA_ROLE=<authority id> | ...
```
