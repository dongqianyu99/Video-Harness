# Video Harness Architecture

Video Harness is an evidence-compilation harness, not a captioning script. Its architecture borrows the useful repository-level separation visible in Codex—CLI, core orchestration, configuration, protocol, diagnostics, scripts, and documentation—without copying Codex's language or monorepo scale.

## Dependency direction

```text
CLI
  → typed config
  → annotation pipeline
      → temporal media service
      → inspection/evidence provider
      → debug artifact sink
      → canonical document writer

media/provider/debug implementations
  → protocol dataclasses

canonical document
  ✕ must not depend on temporary execution artifacts
```

The CLI contains no frame-selection, prompt, provider, or document-mutation logic. The pipeline owns the state transition for one Unit. Provider adapters serialize protocol requests but do not decode videos. The media layer does not know task instructions or evidence schemas. The debug sink observes execution but does not change it.

## Modules

```text
src/video_harness/
├── cli.py                 # thin command entry points
├── config.py              # immutable runtime/debug/media budgets
├── pipeline.py            # two-pass Unit annotation state machine
├── temporal_media.py      # multiview decode and temporal image products
├── annotations.py         # provider protocols and SDK adapters
├── prompts.py             # versioned strict tools and prompts
├── evidence.py            # evidence validation
├── debug_artifacts.py     # disabled sink / filesystem debug sink
├── sampling.py            # canonical Unit/source planning
├── robodojo.py            # source dataset adapter
├── reader.py              # canonical document consumer boundary
└── renderer.py            # deterministic Guide Views
```

Existing pairing, training-split, and Actuator reader behavior remain independent from the media execution layer.

## Unit state machine

```text
PLANNED
  → DECODED
  → BASE_PACK_READY
  → MOTION_ANALYZED
  → DETAIL_READY | DETAIL_SKIPPED
  → TASK_INTERPRETED
  → CAUSALLY_VALIDATED | RETRY_REQUESTED
  → ACCEPTED | NEEDS_REVIEW
  → COMMITTED

Any failure before COMMITTED leaves the previous canonical annotation unchanged.
```

The pipeline writes canonical output only after the final evidence validates. Intermediate output belongs to the debug sink and may be absent by design.

## Debug modes

`debug=false` uses an in-memory/no-op sink. Decoded frames, Unit clips, sheets, crops, Call 1 output, and request payloads are released after the Unit finishes.

`debug=true` uses a filesystem sink rooted at an explicit new directory. Per Unit it records the three Unit clips, decoded frames, overview/stage/detail sheets, Call 1 output, each Call 2 attempt, final selection, payload hashes, and a manifest. Debug artifacts are diagnostic evidence, not canonical documents and not training inputs.

## Stable invariants

- all three views cover the exact same Unit-local frame range;
- all provider images carry one stable `EVIDENCE`, `VIEW`, and `CAMERA_ROLE`
  label from the shared Camera Authority Contract;
- `cam_high [FIXED_GLOBAL]` governs global world-state claims, while moving wrist
  cameras govern local identity, gripper state, and contact claims;
- wrist-camera ego-motion alone cannot establish global object displacement;
- each of `cam_high`, `cam_left_wrist`, and `cam_right_wrist` has an independent
  5×5 overview covering Unit-local frames 0–24;
- each view has a 2×3 stage sheet for frames 0,5,10,15,20,25;
- all three views also produce independent BEFORE/AFTER endpoints;
- detail crops use only `cam_high` and one fixed ROI over a contiguous interval;
- Call 1 does not receive task text;
- Call 1 Motion Summary is preserved and passed to Call 2 as task-blind temporal
  evidence that Call 2 may qualify or correct;
- Call 2 receives no overview or stage sheets;
- Call 2 describes all six synchronized endpoint images separately;
- Call 2 may use the optional `cam_high` detail sheet;
- Call 2 requests retry only for a clear basic causal or physical contradiction;
- Call 2 runs at most three times and retains the final result as `needs_review`
  when no attempt passes;
- task text follows visual inputs and only supplies naming/role context;
- canonical documents store source references and evidence, never generated media;
- provider responses bind by document/unit/request role, never completion order.

## Configuration

Every run resolves one immutable configuration snapshot containing schema/prompt versions, media grid sizes, detail ROI limits, provider/model IDs, debug mode/root, retry limits, and output paths. Secrets remain environment/process inputs and are excluded from configuration serialization.

## Testing layers

1. protocol/schema golden tests;
2. media command and pixel-layout tests;
3. provider serialization tests with local HTTP mocks;
4. pipeline state/error/debug-sink tests with fakes;
5. source-contract tests on local synthetic videos;
6. bounded real-data pilot and human semantic review.

Local tests prove execution contracts, not VLM semantic correctness. A generated annotation becomes training-eligible only through the separate quality gate defined by the data workflow.
