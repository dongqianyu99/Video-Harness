# Video Harness

`VideoHarness` compiles a successful RoboDojo demonstration into a lightweight,
source-grounded behavior document. A canonical document stores only text,
frame references, source identity, and provenance. Images, videos, sheets, and
crops are reconstructed from the source dataset at annotation time and are not
saved unless debug mode is explicitly enabled.

The compiler processes each nominal one-second guidance Unit as 26
consecutive frames at 25 Hz from three synchronized cameras. It uses two VLM
calls:

1. **Motion analysis** receives one 5×5 overview and one higher-resolution 2×3
   stage sheet per camera. It is task-blind, preserves a Motion Summary, and may
   request one fixed `cam_high` detail crop.
2. **Task interpretation** receives that Motion Summary, six original-resolution
   BEFORE/AFTER endpoint images, the optional detail sheet, and finally the coarse
   Task Instruction. It describes each camera endpoint, interprets the Unit, and
   performs a permissive causal self-check.

This is a compiler/harness boundary, not a replacement for the Actuator. The
query policy still receives its native live observations and task prompt; a
different same-task support episode supplies the behavior document.

### Camera authority

- `cam_high [FIXED_GLOBAL]` is a fixed elevated oblique view of the tabletop and
  is authoritative for global position, pad occupancy, object count, ordering,
  scene state, and world-relative displacement.
- `cam_left_wrist [MOVING_LOCAL_LEFT_WRIST]` moves with the left gripper and is
  authoritative for local identity, left-gripper state, contact, grasp, and
  release.
- `cam_right_wrist [MOVING_LOCAL_RIGHT_WRIST]` provides the corresponding local
  evidence for the right gripper.

All views observe the same synchronized scene. Wrist-camera pixel motion includes
camera ego-motion and cannot by itself establish global object movement. Every
provider image label contains `EVIDENCE`, `VIEW`, and `CAMERA_ROLE` fields.

## Repository layout

```text
src/video_harness/
├── cli.py                 # command entry points
├── config.py              # immutable runtime/debug contract
├── pipeline.py            # two-call Unit state machine
├── temporal_media.py      # exact multiview decode and visual products
├── annotations.py         # OpenAI, Anthropic, and mock providers
├── prompts.py             # versioned prompts and strict tool schemas
├── evidence.py            # structured-output validation
├── debug_artifacts.py     # no-op or filesystem diagnostic sink
├── sampling.py            # canonical document planning and validation
├── reader.py              # Actuator-facing document boundary
└── renderer.py            # deterministic derived Guide text
```

See [architecture.md](docs/architecture.md) for module ownership and
[evidence-prompt.md](docs/evidence-prompt.md) for the provider contract.

## Run on a new machine

The interactive runner prepares the environment, downloads RoboDojo, asks for
the provider and hidden API token, builds documents, annotates them, and reports
quality:

```bash
cd code/VideoHarness
bash scripts/run_interactive.sh
```

Machine requirements:

- Linux with network access;
- Python 3.11 or newer, resolved by `uv`;
- `ffmpeg` with PNG, JPEG, and H.264 support;
- about 120 GB for the public joint-space LeRobot v3 dataset;
- an OpenAI or Anthropic token for a real run. The mock backend needs no token.

The runner reads API tokens with hidden terminal input, exports them only to the
current process, and removes them on exit. It does not put them in command-line
arguments or output artifacts.

## Manual setup

```bash
cd code/VideoHarness

# If uv is absent:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Ubuntu example; skip if ffmpeg is already present:
sudo apt-get update
sudo apt-get install -y ffmpeg

uv sync --extra dev --extra providers
uv run pytest -q
uv run hf version
ffmpeg -version | head -n 1
```

Do not copy `.venv` between machines. Recreate it from `pyproject.toml`; this
portable package intentionally does not distribute a machine-generated
`uv.lock`.

## Download RoboDojo

```bash
export ROBODOJO_REPO_ROOT=/path/to/data/robodojo-hf
scripts/download_robodojo.sh "$ROBODOJO_REPO_ROOT"

export ROBODOJO_DATASET_ROOT="$ROBODOJO_REPO_ROOT/data/RoboDojo_lerobot_v30_video"
scripts/verify_robodojo.sh "$ROBODOJO_DATASET_ROOT"
```

This downloads the public joint-space LeRobot v3 export used by the Pi_05 data
contract: three RGB cameras, 14D joint state/action, 25 Hz. The compiler selects
the 34 runnable benchmark tasks and excludes the DLC auxiliary task from
guidance generation.

Metadata-only download remains useful for inventory work, but annotation,
including the mock end-to-end media smoke, requires the full video shards:

```bash
scripts/download_robodojo.sh "$ROBODOJO_REPO_ROOT" --metadata-only
```

## Build documents

Use a new output directory:

```bash
export VH_OUTPUT=/path/to/output/RoboDojo

uv run video-harness inspect \
  --dataset-root "$ROBODOJO_DATASET_ROOT"

uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VH_OUTPUT" \
  --sample-hz 1 \
  --supports-per-query 1 \
  --max-tasks 1 \
  --episodes-per-task 3
```

The build writes immutable source/frame references with pending annotation
slots. It does not decode or duplicate media.

## Annotate

### OpenAI

```bash
read -r -s -p "OpenAI API token: " OPENAI_API_KEY
echo
export OPENAI_API_KEY

uv run video-harness annotate \
  --provider openai \
  --model <vision-capable-model-id> \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.openai.jsonl" \
  --limit-documents 1 \
  --limit-units-per-document 3

unset OPENAI_API_KEY
```

### Anthropic Claude

```bash
read -r -s -p "Anthropic API token: " ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY

uv run video-harness annotate \
  --provider anthropic \
  --model <vision-capable-model-id> \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.anthropic.jsonl" \
  --limit-documents 1 \
  --limit-units-per-document 3

unset ANTHROPIC_API_KEY
```

### Mock pipeline smoke

The mock follows the same decode/sheet/two-pass pipeline but makes no API call
and deliberately produces non-trainable evidence:

```bash
uv run video-harness annotate \
  --provider mock \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.mock.jsonl" \
  --limit-documents 1 \
  --limit-units-per-document 1
```

Validate any result with:

```bash
uv run video-harness report --documents "$VH_OUTPUT/documents.openai.jsonl"
```

## Full run

Omit all build and annotation limits:

```bash
uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VH_OUTPUT" \
  --sample-hz 1 \
  --supports-per-query 1

uv run video-harness annotate \
  --provider openai \
  --model <vision-capable-model-id> \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.openai.jsonl"

uv run video-harness report \
  --documents "$VH_OUTPUT/documents.openai.jsonl"
```

The full 1 Hz corpus contains many thousands of Units and therefore many paid
API calls. Always run a bounded pilot first.

## Debug mode

Normal mode writes no intermediate media. Enable debug explicitly for a small
pilot:

```bash
uv run video-harness annotate \
  --provider openai \
  --model <vision-capable-model-id> \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.openai.debug-pilot.jsonl" \
  --limit-documents 1 \
  --limit-units-per-document 1 \
  --debug \
  --debug-root "$VH_OUTPUT/debug"
```

Each debug Unit contains:

```text
debug/<document>/<unit>/
├── call1.json
├── call2-attempt-01.json       # or attempt-scoped error
├── call2-attempt-02.json       # only when Call 2 requests retry
├── call2-attempt-03.json       # maximum third attempt
├── final.json                  # selected attempt and canonical evidence
├── manifest.json
├── videos/{cam_high,cam_left_wrist,cam_right_wrist}.mp4
├── frames/<view>/frame-00.jpg ... frame-25.jpg
├── sheets/<view>-overview.png
├── sheets/<view>-stage.png
├── sheets/cam_high-detail.png  # only when requested
└── endpoints/*.jpg
```

Debug media is diagnostic only. It is not canonical data and must not be used as
the Actuator training source.

## Evidence contract

The two calls are composed into `video-harness.evidence`:

- the task-blind `motion_summary` from Call 1;
- separate BEFORE and AFTER descriptions for all three camera views;
- an optional `detail_observation`;
- `action_description` and `task_role` from Call 2;
- Call 2 `causal_validation`;
- `review_status: accepted | needs_review`.

Every Guidance Unit is treated as an interval of robot execution. The VLM does
not classify Units as changed/no-change/insufficient or decide whether to discard
them. Even when global object placement appears stable, local action such as
gripper closure, contact, grasp, release, or transport must be recorded.

Call 2 can run at most three times. It requests retry only for a clear basic
causal or physical contradiction. If all three schema-valid interpretations ask
for retry, the final result is retained as `needs_review` and excluded from
training by default.

## Generated artifacts

```text
<output>/
├── dataset.json
├── episodes.jsonl
├── documents.jsonl
├── documents.<provider>.jsonl
└── pairs.jsonl
```

Canonical artifacts contain no copied source media or raw API response. All
files from one build carry the same `build_id`; downstream readers must reject
mismatched build IDs or schema versions.

## Formal training split

After annotation quality review:

```bash
uv run video-harness make-training-split \
  --dataset-artifact "$VH_OUTPUT/dataset.json" \
  --episodes "$VH_OUTPUT/episodes.jsonl" \
  --documents "$VH_OUTPUT/documents.openai.jsonl" \
  --output-root /path/to/output/video-harness-split-seed0 \
  --support-documents-per-task <decide-count> \
  --heldout-documents-per-task <decide-count> \
  --min-trainable-units 1 \
  --seed 0
```

Support, query, held-out, and unused source episodes are role-disjoint. Every
query episode receives one same-task support document for its full lifetime.

## Current validation boundary

Local tests validate source identity, exact frame-index routing, media layout,
provider serialization, structured-output contracts, the two-pass pipeline,
debug/no-debug behavior, canonical document invariants, and reader handoff. They
do not prove VLM semantic correctness. Before training, manually audit a
stratified sample of debug pilots and final canonical records.
