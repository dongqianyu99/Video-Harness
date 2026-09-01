# Architecture

VideoHarness compiles successful robot episodes into Guidance Documents. RoboDojo and OpenPI consume those Documents but do not participate in annotation or document repair.

## Data flow

```text
RoboDojo episode
  -> uniform Boundary and Unit plan
  -> multiview annotation
  -> Sequence Audit and targeted repair
  -> accepted Document
  -> GuidePlan
  -> cached GuideInput
  -> GuideMemory
  -> [Guide | Control | Action] Pi0.5
```

The dependency direction is one way. VideoHarness knows the RoboDojo source format but does not import OpenPI. OpenPI imports the public VideoHarness reader and media interfaces while constructing the Guide cache.

## Guidance Document

A Document contains an ordered chain:

```text
Boundary B0 -> Unit U0 -> Boundary B1 -> Unit U1 -> Boundary B2
```

For `N` Units there are `N+1` Boundaries. Each Boundary stores an episode-local frame reference and one description for each of the three cameras. Each Unit stores the transition between adjacent Boundaries as Motion, Detail, Action, and Task role text. Shared Boundaries appear once.

Documents are stored by task and episode:

```text
documents-openai/<task-name>/episode-XXXXXXX.document.jsonl
```

The reader accepts only complete Documents with `quality_status=accepted`. It builds a stable catalog and projects a selected `document_id` into a `GuidePlan`. Audit reasons, provider metadata, repair history, and other provenance stay outside the model-facing plan.

## Annotation

VideoHarness uses two provider calls per Unit. Call 1 sees temporal summaries from all three cameras and measured gripper state without the task instruction. It describes visible motion and selects a useful `cam_high` detail region. Call 2 sees the task instruction, six original Boundary images, the motion summary, the detail view, gripper samples, and any Boundary descriptions already written by an adjacent Unit.

`cam_high` is the source for global layout and displacement. Wrist cameras are the source for local identity, gripper configuration, proximity, and contact. Grasp, release, and contact claims need support from the relevant wrist view. The second call may revise the first motion summary before the evidence is written.

Units inside one Document run in order because adjacent Units share a Boundary. Different Documents can run in parallel. The final Sequence Audit checks the complete chain. Targeted repair updates the Unit or Boundary named by the audit. A Document becomes accepted when the final audit has no unresolved issue; otherwise it is quarantined.

## Guide representation

`GuidePlan.boundaries` is the deduplicated Boundary bank. `before_slot` and `after_slot` point into that bank and are used only while building the memory order.

The materializer produces:

```text
Boundary images       [G,F,3,224,224,3]
Boundary text         [G,F,3,T_B]
Transition text       [G,U,T_T]
Memory map            [G,S]
```

Each Boundary is encoded from three image and text pairs, then compressed to `K_B` tokens. Each transition is compressed to `K_T` tokens. The memory map packs them as `B0,U0,B1,U1,...`. The current defaults are `K_B=8` and `K_T=4`.

The persistent cache stores compact float32 GuideInput arrays. Cache construction decodes each Guide once. Training workers load the cached arrays, pad them to the shared shape, and keep a small process-local LRU for recently used Documents. Learned image features and GuideMemory are not cached because their values change during fine-tuning.

## Sampling and model input

Each microbatch samples `G` accepted Documents uniformly without replacement. For each Document it samples `Q` queries from the corresponding task, including the source episode when selected. The batch keeps the `[G,Q]` grouping until Guide encoding finishes, then broadcasts each GuideMemory across its `Q` queries.

Pi0.5 receives three regions:

```text
Guide   sees Guide
Control sees Guide and Control
Action  sees Guide, Control, and Action
```

Control is the current three-camera observation, task prompt, and 14D robot state. Action is the noisy action chunk used by the existing flow-matching objective. The guided path keeps the stock optimizer, TrainState, EMA, action horizon, normalization, and checkpoint format.

## Runtime boundaries

Training supports one JAX process with one or more local devices. A stock Pi0.5 checkpoint initializes the native parameter leaves, while the Guide encoder starts from its own initialization. Guided checkpoints resume only with the same training configuration and Document catalog.

Evaluation selects the first accepted Document for each task, loads its cached GuideInput, encodes GuideMemory once, and reuses that memory for each control cycle in the task session.
