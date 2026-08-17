# Evidence prompt v1

Canonical prompt version: `video-harness.evidence.v1`

Canonical evidence schema: `video-harness.evidence.v1`

The VLM receives one coarse task instruction and two temporally ordered sampled
endpoints from a successful support trajectory. It compiles a bounded evidence
record; it does not reconstruct the hidden motion or write a full plan.

## Evidence hierarchy

The schema deliberately separates information by epistemic status:

1. `visual_observation` is direct endpoint evidence only.
2. `entities` records a small set of local objects, visible descriptions, and
   task roles. `grounding` states whether task context helped assign a name or
   role.
3. `operation_hint` is explicitly a hypothesis and may be `null`.
4. `visible_end_effector` is a bounded endpoint observation, not a claim about
   which arm caused the hidden transition.
5. `task_relevance` does not imply task or subtask success.
6. `visibility_limits` records what two sampled images cannot provide.

This lets one VLM call per guidance unit preserve potentially useful semantics
while a downstream renderer chooses how much to expose to an Actuator.

## Canonical system prompt

```text
You are the Video Harness evidence compiler. You receive exactly two temporally
ordered sampled endpoints from one interval of a successful robot demonstration:
BEFORE and AFTER. You also receive a coarse task instruction as context.

Your job is not to caption everything, reconstruct the hidden motion, or write a
plan. Compile a small, evidence-grounded record that a downstream robot policy can
render in different ways without asking a VLM again. Return exactly one call to
record_transition_evidence; output no explanation or reasoning outside the tool.

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
```

## Per-unit request

```text
Task instruction (context only; it is not visual evidence):
{task_instruction}

Camera: {camera_key}
Transition unit: {unit_id}
BEFORE episode-local frame: {before_frame}
AFTER episode-local frame: {after_frame}

Compile the strict transition evidence record from the labeled sampled endpoints.

BEFORE image: <image>
AFTER image: <image>
```

## Changed example

```json
{
  "change_status": "changed",
  "visual_observation": {
    "before": "A bread slice rests on the table beside the toaster.",
    "after": "The bread slice is visible inside the toaster slot.",
    "change": "The bread slice is now inside the toaster slot.",
    "support": "clear"
  },
  "entities": [
    {
      "name": "bread slice",
      "visual_description": "A light-brown rectangular slice beside the toaster.",
      "role": "manipulated_object",
      "visible_in": "both",
      "grounding": "visual_plus_task",
      "support": "clear"
    },
    {
      "name": "toaster slot",
      "visual_description": "A dark slot on top of the toaster.",
      "role": "target_receptacle",
      "visible_in": "both",
      "grounding": "visual_plus_task",
      "support": "clear"
    }
  ],
  "operation_hint": {
    "label": "insert",
    "description": "Insert the bread slice into the toaster slot.",
    "support": "endpoint_plus_task_context"
  },
  "visible_end_effector": "right",
  "task_relevance": "relevant",
  "visibility_limits": [
    "motion_path",
    "force",
    "precise_pose",
    "grasp_contact"
  ]
}
```

`visual_observation.change` is the default evidence-safe text. The operation
description is retained for retrieval and experimental renderers, but it is not
silently treated as direct visual supervision.

## Local validation

Provider-side strict tools are necessary but not sufficient. The local validator
also enforces:

- exact keys at every object boundary;
- closed enums and at most five entities;
- one plain-text sentence and bounded words per free-text field;
- agreement between change status, nullable fields, and visual support;
- no operation hint when no change is observed;
- fixed endpoint visibility limits;
- rejection of trajectories, coordinates, distances, velocities, and similar
  control details.

The compiler stores no raw API response, chain-of-thought, or image copy.
