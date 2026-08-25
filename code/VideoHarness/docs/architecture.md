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

## Canonical temporal model

```text
Boundary B0
  → Evidence Unit T0
Boundary B1
  → Evidence Unit T1
Boundary B2
  ...
```

For `N` Evidence Units, the document contains exactly `N+1` ordered Boundary
States. A Boundary State owns one sampled frame reference and one synchronized
three-view static description. An Evidence Unit owns only the task-blind motion
evidence and task-conditioned transition interpretation between two adjacent
Boundary IDs. Static Boundary text is never duplicated inside transition records.
Boundary visual quality status is independent from transition causal quality status.

The CLI contains no frame-selection, prompt, provider, or document-mutation logic. The pipeline owns the state transition for one Evidence Unit. Provider adapters serialize protocol requests but do not decode videos. The media layer does not know task instructions or evidence schemas. The debug sink observes execution but does not change it.

## Modules

```text
src/video_harness/
├── cli.py                 # thin command entry points
├── config.py              # immutable runtime/debug/media budgets
├── pipeline.py            # two-pass Evidence Unit annotation state machine
├── reconciliation.py      # automatic Sequence Audit and document repair
├── temporal_media.py      # multiview decode and temporal image products
├── annotations.py         # provider protocols and SDK adapters
├── prompts.py             # versioned strict tools and prompts
├── evidence.py            # evidence validation
├── debug_artifacts.py     # disabled sink / filesystem debug sink
├── sampling.py            # canonical Evidence Unit/source planning
├── robodojo.py            # source dataset adapter
├── reader.py              # canonical document consumer boundary
└── renderer.py            # deterministic Guide Views
```

Existing pairing, training-split, and Actuator reader behavior remain independent from the media execution layer.

## Evidence Unit state machine

```text
PLANNED
  → DECODED
  → BASE_PACK_READY
  → MOTION_ANALYZED
  → DETAIL_READY | DETAIL_SKIPPED
  → TASK_INTERPRETED
  → CAUSAL_PASS → COMMITTED
  → CAUSAL_RETRY → TARGETED_REPROCESS
                   → RESOLVED → COMMITTED
                   → UNRESOLVED → COMMITTED_AS_QUARANTINED
```

Any failure before COMMITTED leaves the previous canonical annotation unchanged.

## Automatic document quality state machine

```text
NORMAL_COMPILE
  → TARGETED_REPROCESS when a Unit is inconsistent
  → SEQUENCE_AUDIT when every Unit is complete
  → ISSUE_DIRECTED_REPAIR → SEQUENCE_AUDIT, within a fixed budget
  → ACCEPTED | QUARANTINED
```

`QUARANTINED` is an automatic terminal state, not a human work queue. Default
Readers and training splits require a complete `ACCEPTED` document. Human review
is reserved for sampled Harness evaluation and does not block corpus generation.

The pipeline writes canonical output only after the final evidence validates. Intermediate output belongs to the debug sink and may be absent by design.

## Batch execution

The Document is the only parallel work unit. Its Evidence Units remain sequential;
different Documents are assigned to deterministic shards and may run through independent
worker-local provider clients. A completed Document is written to an atomic checkpoint.
Resume reuses terminal checkpoints and continues nonterminal ones, while final merge
requires one terminal checkpoint for every source Document and restores source order.

This layer uses local or shared files and the Python standard library. It deliberately
does not require a database, message broker, or external scheduler.

The same checkpoint root owns an append-only JSONL event log and a lock-protected run
state. Provider calls record stage, Document/Unit identity, latency, response identity,
usage, and failure details. An optional shared API-call cap is reserved before external
provider invocation; budget exhaustion interrupts the shard without converting pending
evidence into failed or quarantined data. Without an explicit cap, accounting remains
enabled but enforcement is disabled.

Failure handling follows scope rather than one global catch policy. Configuration,
source contract, and artifact-write errors are run-level and stop the shard. Budget
exhaustion is a resumable control signal. Media/provider/schema/timeout failures are
Evidence-Unit-local: bounded recovery runs first, unresolved evidence quarantines its
Document, and the corpus continues. Call 1 fallback is never sufficient for acceptance;
it must be replaced by successful full-temporal repair or remain quarantined. All FFmpeg
subprocesses and provider clients have explicit configurable timeouts.

## Debug modes

`debug=false` uses an in-memory/no-op sink. Decoded frames, Evidence Unit clips, sheets, crops, Call 1 output, and request payloads are released after the Evidence Unit finishes.

`debug=true` uses a filesystem sink rooted at an explicit new directory. Per Evidence Unit it records the three Evidence Unit clips, decoded frames, overview/keyframe/detail sheets, Call 1 and Call 2 outputs, automatic repair attempts, the final decision, payload hashes, and a manifest. Debug artifacts are diagnostic evidence, not canonical documents and not training inputs.

## Stable invariants

- all three views cover the exact same Evidence-Unit-local frame range;
- adjacent Evidence Units reference one shared Boundary State rather than storing
  separate previous-AFTER and current-BEFORE descriptions;
- Boundary State count is exactly Evidence Unit count plus one, and the first and
  final boundaries reference the first and final episode frames;
- all provider images carry one stable `EVIDENCE`, `VIEW`, and `CAMERA_ROLE`
  label from the shared Camera Authority Contract;
- `cam_high [FIXED_GLOBAL]` governs global world-state claims, while wrist-mounted
  cameras govern local identity, gripper state, and contact claims;
- wrist-camera ego-motion alone cannot establish global object displacement;
- each of `cam_high`, `cam_left_wrist`, and `cam_right_wrist` has an independent
  5×5 overview covering Evidence-Unit-local frames 0–24;
- each view has a 2×3 keyframe sheet for frames 0,5,10,15,20,25;
- all three views also produce independent BEFORE/AFTER Boundary images;
- detail crops use only `cam_high` and one fixed ROI over a contiguous interval;
- Call 1 does not receive task text;
- Call 1 may receive only the immediately preceding accepted task-blind Motion
  Summary as fallible continuity context; current images remain authoritative;
- every Call 1 result locates one meaningful interaction window, whether or not
  a detail crop is requested;
- Call 1 Motion Summary is preserved and passed to Call 2 as task-blind temporal
  evidence that Call 2 may qualify or correct;
- Call 2 receives no overview or keyframe sheets;
- Call 2 sees all six synchronized Boundary images but emits a description only
  for a Boundary that has no accepted canonical description;
- a material conflict with an accepted Boundary is recorded and enters Targeted
  Reprocessing without creating a second canonical description;
- Call 2 may use the optional `cam_high` detail sheet;
- normal Call 2 runs once by default; an inconsistency enters bounded Targeted
  Reprocessing with full temporal evidence;
- every complete document receives a Sequence Audit and bounded issue-directed
  repair before it becomes `accepted` or `quarantined`;
- task text follows visual inputs and only supplies naming/role context;
- canonical documents store source references and evidence, never generated media;
- provider responses bind by document/unit/request role, never completion order.

Pi0.5 keeps its per-transition Guide resampler. Boundary images are decoded and
encoded once as unique frame slots; adjacent transitions may reference the same
encoded Boundary token without duplicating source media or canonical state text.

## Configuration

Every run resolves one immutable configuration snapshot containing schema/prompt versions, media grid sizes, detail ROI limits, provider/model IDs, debug mode/root, retry limits, and output paths. Secrets remain environment/process inputs and are excluded from configuration serialization.

## Testing layers

1. protocol/schema golden tests;
2. media command and pixel-layout tests;
3. provider serialization tests with local HTTP mocks;
4. pipeline state/error/debug-sink tests with fakes;
5. source-contract tests on local synthetic videos;
6. bounded real-data pilot and sampled human semantic QA.

Local tests prove execution contracts, not VLM semantic correctness. A generated annotation becomes training-eligible only through the separate quality gate defined by the data workflow.
