# Video Harness User Guide

This guide covers the VideoHarness compiler workflow for the public RoboDojo LeRobot
v3 dataset. The repository-wide data and Pi0.5 conventions are defined in
[Standard Data and Guided Training](data-and-training.md).

## Requirements

- Linux with network access during setup and annotation;
- Python 3.11 or newer;
- `uv`;
- FFmpeg with H.264, PNG, and JPEG support;
- approximately 120 GB for the RoboDojo dataset;
- an OpenAI or Anthropic API key for real annotation.

With debug disabled, the full 3,400-episode output is expected to occupy well below
1 GB including Documents, checkpoints, and run logs.

## 1. Configure the environment

```bash
cd Video-Harness
source scripts/env.sh
cd code/VideoHarness
```

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Install FFmpeg on Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Create the environment with both supported provider SDKs:

```bash
uv sync --extra providers
uv run video-harness --help
ffmpeg -version | head -n 1
```

Do not copy `.venv` between machines. Recreate it from `pyproject.toml` on each target
machine.

## 2. Download RoboDojo

Use the repository's ignored `data/` directory:

```bash
cd ../..
scripts/download_robodojo.sh
scripts/verify_robodojo.sh
cd code/VideoHarness
```

The downloader resumes existing files from the public
`RoboDojo-Benchmark/RoboDojo` dataset. Verification checks the released file and byte
counts, metadata, three RGB cameras, 25 Hz timing, and 14-dimensional state/action
contract.

## 3. Configure a provider

### OpenAI

```bash
read -r -s -p "OpenAI API key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
export VH_PROVIDER=openai
export VH_MODEL="your-vision-model-id"
```

For an OpenAI-compatible endpoint, also set `OPENAI_BASE_URL`. For example:

```bash
export OPENAI_BASE_URL=https://api.deepseek.com
```

OpenAI-compatible providers can return structured data through the existing forced
tool-call protocol or through strict JSON Schema Output. JSON Schema Output supports
thinking, requests `response_format=json_schema` with `strict=true`, and retains local
schema validation as a second check:

```bash
--output-mode json --thinking --reasoning-effort high
```

Thinking is enabled by default in JSON mode. Use `--no-thinking` to disable it, or
choose `low`, `high`, or `max` with `--reasoning-effort`. Tool mode remains the
default compatibility path and ignores the thinking options. JSON mode does not
silently fall back to `json_object` or repair malformed provider output.

### Anthropic

```bash
read -r -s -p "Anthropic API key: " ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY
export VH_PROVIDER=anthropic
export VH_MODEL="your-vision-model-id"
```

API keys are process inputs. They are not written to Documents, checkpoints, debug
artifacts, or run logs.

## 4. Run one complete episode in debug mode

The annotation limit below processes exactly one complete episode. The build is
document-only and may select a single episode for a bounded smoke run.

```bash
export VH_DEBUG_RUN="$VIDEO_HARNESS_DATA_ROOT/video-harness/debug"

uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VH_DEBUG_RUN" \
  --sample-hz 1 \
  --max-tasks 1 \
  --episodes-per-task 1

uv run video-harness annotate \
  --provider "$VH_PROVIDER" \
  --model "$VH_MODEL" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_DEBUG_RUN/documents.jsonl" \
  --output "$VH_DEBUG_RUN/documents.$VH_PROVIDER.jsonl" \
  --document-root "$VH_DEBUG_RUN/documents-$VH_PROVIDER" \
  --checkpoint-root "$VH_DEBUG_RUN/checkpoints-$VH_PROVIDER" \
  --limit-documents 1 \
  --debug \
  --debug-root "$VH_DEBUG_RUN/debug"

uv run video-harness report \
  --documents "$VH_DEBUG_RUN/documents.$VH_PROVIDER.jsonl"

uv run video-harness summarize-run \
  --checkpoint-root "$VH_DEBUG_RUN/checkpoints-$VH_PROVIDER"
```

Debug mode retains decoded frames, three-view Unit videos, overview/keyframe/detail
sheets, gripper samples, provider outputs, repair attempts, and manifests. Use it only
for bounded inspection. Structured-output failures retain only safe metadata in normal
run events; debug mode additionally writes the provider's final content under
`debug/<document>/provider-errors/`. Reasoning content is never stored there.

The final episode Document is written to:

```text
<VH_DEBUG_RUN>/documents-<provider>/<official-task-name>/episode-XXXXXXX.document.jsonl
```

## 5. Run the complete corpus with multiple workers

The full build contains 3,400 benchmark episodes and roughly 72,000–76,000 Evidence
Units. Expect at least approximately 150,000 logical VLM calls before repair or retry.
Run a bounded debug episode first and confirm provider limits before starting.

```bash
export VH_RUN="$VIDEO_HARNESS_RUN_ROOT"

uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VH_RUN" \
  --sample-hz 1
```

Start with two to four Document workers. Evidence Units inside one Document remain
sequential; different Documents run concurrently.

```bash
uv run video-harness annotate \
  --provider "$VH_PROVIDER" \
  --model "$VH_MODEL" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_RUN/documents.jsonl" \
  --output "$VH_RUN/documents.$VH_PROVIDER.jsonl" \
  --document-root "$VH_RUN/documents-$VH_PROVIDER" \
  --checkpoint-root "$VH_RUN/checkpoints-$VH_PROVIDER" \
  --workers 4
```

The CLI displays one progress bar per RoboDojo task. Each bar counts completed
Documents in the current shard.

An optional shared call cap can stop the run before the next provider call:

```bash
--max-api-calls 180000
```

Omit the option to disable enforcement. Call accounting and usage logging remain active.

## 6. Resume an interrupted run

Repeat the same annotation command with identical configuration and add `--resume`:

```bash
uv run video-harness annotate \
  --provider "$VH_PROVIDER" \
  --model "$VH_MODEL" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_RUN/documents.jsonl" \
  --output "$VH_RUN/documents.$VH_PROVIDER.jsonl" \
  --document-root "$VH_RUN/documents-$VH_PROVIDER" \
  --checkpoint-root "$VH_RUN/checkpoints-$VH_PROVIDER" \
  --workers 4 \
  --resume
```

Accepted and semantic-quarantined Documents are reused. Documents carrying technical
failure records are rescheduled. FFmpeg, provider, schema, and timeout failures use
bounded local retries before they become resumable technical failures.

Do not change the provider, model, sampling, retry configuration, shard count, source
manifest, or dataset root while resuming the same checkpoint root.

## 7. Inspect the result

```bash
uv run video-harness report \
  --documents "$VH_RUN/documents.$VH_PROVIDER.jsonl"

uv run video-harness summarize-run \
  --checkpoint-root "$VH_RUN/checkpoints-$VH_PROVIDER"
```

`report` validates canonical Documents and reports accepted/quarantined Units,
Boundaries, and Documents. `summarize-run` writes
`<checkpoint-root>/run-summary.json` containing:

- per-task planned, completed, accepted, and quarantined Documents;
- Call 1, Call 2, repair, and Sequence Audit counts;
- p50, p95, and maximum provider latency per stage;
- provider usage totals and shared API-call count;
- timeout, provider, Unit, Document, and shard error counts.

## 8. Use Documents with Pi0.5

Pi0.5 training and evaluation consume the per-episode directory, not the aggregate
JSONL:

```bash
--documents-root "$VH_RUN/documents-$VH_PROVIDER"
```

The loader scans `<official-task-name>/*.document.jsonl`, validates accepted and
quarantined Documents, and builds a stable in-memory catalog. Training batches do not
repeatedly scan the directory.

## Output layout

```text
<VH_RUN>/
├── dataset.json
├── episodes.jsonl
├── documents.jsonl                     # planned source manifest
├── documents.<provider>.jsonl          # merge/report compatibility index
├── documents-<provider>/               # Pi0.5 Guidance input
│   └── <official-task-name>/
│       └── episode-XXXXXXX.document.jsonl
├── checkpoints-<provider>/
│   ├── run.json
│   ├── run-state.json
│   ├── events.jsonl
│   ├── run-summary.json
│   └── documents/<document-id-sha256>.json
└── debug/                               # only when --debug is enabled
```

## Operational notes

- Normal mode stores no generated images or videos.
- Sheet composition uses Pillow; FFmpeg is used only for source decode and debug-video
  encoding.
- Every Unit and Boundary must be technically complete. Sequence Audit is the sole
  semantic gate, and only an accepted complete Document reaches the default Reader.
- Shared multi-machine checkpoint storage must support atomic rename and POSIX advisory
  locks.
- `VideoHarness.zip`, datasets, checkpoints, runs, `.venv`, and caches are release
  artifacts or local state and should not be committed.
