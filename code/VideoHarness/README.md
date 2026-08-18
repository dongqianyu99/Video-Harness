# Video Harness

`VideoHarness` converts one successful RoboDojo support demonstration into a
lightweight behavior-document sidecar. It samples the head-camera video at 1 Hz,
sends each ordered BEFORE/AFTER pair to a VLM, validates the returned structured
evidence, and stores only text, frame indices, provenance, and support/query
pairing. It does not copy images or videos into the generated document.

## Run on a new machine

The recommended entry point is the interactive runner:

```bash
cd code/VideoHarness
bash scripts/run_interactive.sh
```

It performs the complete workflow in this order:

1. checks or installs `uv`;
2. checks or installs `ffmpeg` on an apt-based Linux system;
3. resolves a Python environment from `pyproject.toml` and runs the test suite;
4. downloads `RoboDojo_lerobot_v30_video` with the official `hf` CLI;
5. verifies the public Pi_05-compatible source contract and excludes DLC from
   guidance generation;
6. asks you to select OpenAI, Anthropic Claude, or the no-token mock backend;
7. reads the API token with hidden terminal input and asks for a model ID;
8. asks for the processing scale;
9. builds documents, decodes referenced frames, annotates them, and prints an
   evidence-quality report;
10. keeps the generated result directory locally and prints its artifact paths.

The default **API smoke** processes only three guidance units. Use it before a
larger paid run. The script also offers a one-task pilot, all 34 benchmark tasks,
and a custom subset. Full 1 Hz processing requires tens of thousands of VLM
calls, so it requires an explicit `FULL` confirmation.

OpenAI and Anthropic tokens exist only in the current script process environment.
They are not written to a file, included in a command-line argument, or copied
into an artifact. The runner contains no upload credential or automatic upload
destination. RoboDojo is public and needs no HF token.

### Machine requirements

- Linux with outbound network access;
- approximately 120 GB for RoboDojo, plus environment and output space;
- `curl` if `uv` is not already installed;
- root or `sudo` only if the script must install `ffmpeg` automatically.

On a managed system without apt/root access, ask the administrator to provide
`uv` and `ffmpeg`, then rerun the same script.

## Manual workflow

The following commands are equivalent to the interactive path.

### 1. Prepare the environment

```bash
cd code/VideoHarness

# Only needed when uv is absent.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Ubuntu example; skip when ffmpeg is already installed.
sudo apt-get update
sudo apt-get install -y ffmpeg

uv sync --extra dev --extra providers
uv run pytest -q
uv run hf version
```

Do not copy `.venv` between machines. Recreate it from `pyproject.toml` and
let `uv` resolve the environment on each machine. A locally generated
`uv.lock` is intentionally ignored and must not be committed or uploaded.

### 2. Download RoboDojo with `hf`

Choose a location outside the source package:

```bash
export ROBODOJO_REPO_ROOT=/path/to/data/robodojo-hf

scripts/download_robodojo.sh "$ROBODOJO_REPO_ROOT"

export ROBODOJO_DATASET_ROOT="$ROBODOJO_REPO_ROOT/data/RoboDojo_lerobot_v30_video"
scripts/verify_robodojo.sh "$ROBODOJO_DATASET_ROOT"
```

The downloader runs the equivalent of:

```bash
uv run hf download RoboDojo-Benchmark/RoboDojo \
  --type dataset \
  --revision main \
  --include 'data/RoboDojo_lerobot_v30_video/**' \
  --local-dir "$ROBODOJO_REPO_ROOT"
```

Downloads are resumable by running the same command again. For metadata-only
development, use:

```bash
scripts/download_robodojo.sh "$ROBODOJO_REPO_ROOT" --metadata-only
```

Metadata is enough for `inspect`, `build`, mock annotation, and `report`. A real
VLM run and `decode-smoke` require the full video download.

### 3. Build behavior documents

Use a new output directory. This small example selects one task and three
episodes:

```bash
export VH_OUTPUT=/path/to/output/video-harness-smoke

uv run video-harness inspect \
  --dataset-root "$ROBODOJO_DATASET_ROOT"

uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VH_OUTPUT" \
  --sample-hz 1 \
  --supports-per-query 1 \
  --max-tasks 1 \
  --episodes-per-task 3

uv run video-harness decode-smoke \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --limit-frames 4
```

The compiler always selects the 34 runnable benchmark tasks: 3,400 episodes in
total. The legacy DLC task is deliberately excluded from VLM processing because
it is auxiliary Pi_05 fine-tuning data rather than a benchmark guidance task.
When assembling the final policy-training dataset, add the original 100 DLC
trajectories separately, without generated guidance. This preserves their role
as auxiliary policy data without paying for or learning a DLC behavior document.

### Complete-run output

When the interactive runner uses **Scale 3: All benchmark data**, successful
annotation and validation remain under the explicitly selected local output
directory. The runner does not upload, publish, or copy that directory to any
remote service. Review and archive the output separately after the evidence
report and a manual sample audit pass.

### 4. Annotate with OpenAI

```bash
read -r -s -p "OpenAI API token: " OPENAI_API_KEY
echo
export OPENAI_API_KEY

uv run video-harness annotate \
  --provider openai \
  --model <vision-capable-openai-model-id> \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.openai.jsonl" \
  --limit-documents 1 \
  --limit-units-per-document 3

unset OPENAI_API_KEY

uv run video-harness report \
  --documents "$VH_OUTPUT/documents.openai.jsonl"
```

### 5. Annotate with Anthropic Claude

```bash
read -r -s -p "Anthropic API token: " ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY

uv run video-harness annotate \
  --provider anthropic \
  --model <vision-capable-claude-model-id> \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.anthropic.jsonl" \
  --limit-documents 1 \
  --limit-units-per-document 3

unset ANTHROPIC_API_KEY

uv run video-harness report \
  --documents "$VH_OUTPUT/documents.anthropic.jsonl"
```

### 6. No-token pipeline smoke

Mock annotation validates the entire artifact pipeline but deliberately returns
non-trainable `insufficient_visual_evidence` records:

```bash
uv run video-harness annotate \
  --provider mock \
  --documents "$VH_OUTPUT/documents.jsonl" \
  --output "$VH_OUTPUT/documents.mock.jsonl"

uv run video-harness report \
  --documents "$VH_OUTPUT/documents.mock.jsonl"
```

## Generated artifacts

```text
<output>/
├── dataset.json
├── episodes.jsonl
├── documents.jsonl
├── documents.<provider>.jsonl
└── pairs.jsonl
```

| Artifact | Purpose |
| --- | --- |
| `dataset.json` | Benchmark-34 source, sampling, selection, and build identity |
| `episodes.jsonl` | Episode and video-shard routing |
| `documents.jsonl` | Immutable frame references with pending annotation slots |
| `documents.<provider>.jsonl` | Structured VLM evidence and per-unit provenance |
| `pairs.jsonl` | Same-task, different-episode support/query binding |

All artifacts from one build carry the same `build_id`. Training code must reject
mismatched build IDs or schema versions.

## Evidence contract

Each VLM call returns a strict `video-harness.evidence.v1` record containing:

- direct BEFORE, AFTER, and visible endpoint change;
- task-relevant entities and roles, including the manipulated object and target;
- a bounded operation hypothesis with explicit evidence source;
- the end effector visible near the endpoint;
- task relevance without a success claim;
- visibility limits such as unseen motion path, force, and precise pose.

Task-conditioned names and operation hypotheses are marked separately from
direct visual facts. By default, only `complete + changed + clear + relevant`
evidence is considered trainable. The full prompt and schema rationale are in
[`docs/evidence-prompt-v1.md`](docs/evidence-prompt-v1.md).

## Reader handoff

The query episode remains the native Pi_05 observation/action sample. Only a
different, same-task support episode is rendered as guidance. The public folder
can be exposed to OpenPI as:

```bash
export HF_LEROBOT_HOME=/parent/of/RoboDojo_lerobot_v30_video
export OPENPI_LEROBOT_REPO_ID=RoboDojo_lerobot_v30_video
```

## Current limitations

- VLM annotation currently writes its output after the requested run completes;
  full-dataset retry/checkpoint support remains to be implemented.
- Frame decoding launches one `ffmpeg` process per referenced frame; use a small
  API smoke before scaling and replace this with batched decoding for production.
- Schema validity does not guarantee semantic correctness. Manually audit a
  stratified sample before using generated evidence for training.
- The 1 Hz units are uniform temporal samples, not discovered semantic stages.
- Actuator attention, Guide memory, retrieval, caching, and training integration
  remain separate work.
