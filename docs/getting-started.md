# Getting started

Run the commands in this guide from the repository root unless a section changes directory. Generated data stays under the ignored `data/` directory.

## Data layout

```text
data/
├── RoboDojo_lerobot_v30_video/
├── video-harness/
│   ├── documents-openai/
│   └── checkpoints-openai/
├── guide-cache/
└── runs/guided-pi05/
```

Load the shared paths in each new shell:

```bash
source scripts/env.sh
```

This sets `HF_LEROBOT_HOME` to `data/`, uses `RoboDojo_lerobot_v30_video` as the LeRobot repo ID, and points VideoHarness and OpenPI at the same dataset directory. Set an environment variable before sourcing the script when you need a different location.

## Download RoboDojo

Install the current `hf` CLI and FFmpeg, then run:

```bash
scripts/download_robodojo.sh
```

The script downloads the joint-space LeRobot v3 dataset from [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/data/RoboDojo_lerobot_v30_video) at the revision recorded in the script. The full download is about 120 GB. Pass `--metadata-only` when you only need the metadata.

## Install VideoHarness

```bash
cd code/VideoHarness
uv sync --extra providers
ffmpeg -version
```

Set the credentials and model name required by the annotation provider. For an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export VH_MODEL=...
```

Use `ANTHROPIC_API_KEY` when running the Anthropic provider. Credentials stay in the process environment and are not written to Documents.

VideoHarness uses strict JSON Schema Output with thinking enabled by default. Pass `--output-mode tool` for a provider that only supports forced tool calls, or `--no-thinking` when the selected JSON endpoint does not support thinking.

## Build and annotate Documents

The default build reads `data/RoboDojo_lerobot_v30_video`, writes to `data/video-harness`, uses 1 Hz sampling, and excludes the 100 auxiliary DLC episodes.

```bash
uv run video-harness build --sample-hz 1

uv run video-harness annotate \
  --provider openai \
  --model "$VH_MODEL" \
  --document-root "$GUIDE_DOCUMENTS_ROOT" \
  --checkpoint-root "$VIDEO_HARNESS_RUN_ROOT/checkpoints-openai" \
  --workers 4
```

Repeat the annotation command with `--resume` after an interruption. Keep the provider, model, sampling rate, dataset, and checkpoint directory unchanged for the same run.

The model-facing files are stored here:

```text
data/video-harness/documents-openai/<task-name>/episode-XXXXXXX.document.jsonl
```

## Structural settings

The official metadata contains 3,500 episodes and 1,859,602 frames. VideoHarness trains on 3,400 benchmark episodes. At 25 FPS and 1 Hz sampling, the corpus contains 73,248 Units. The longest Document has 62 Units and 63 Boundaries.

| Statistic | Episode frames | Units |
|---|---:|---:|
| minimum | 105 | 5 |
| median | 459 | 18 |
| p95 | 1,171 | 48 |
| p99 | 1,419 | 57 |
| maximum | 1,537 | 62 |

The OpenPI commands therefore use these defaults:

```text
max_boundaries = 63
max_units = 62
max_boundary_text_tokens = 128
max_transition_text_tokens = 128
K_B = 8
K_T = 4
G = 1
Q = 64
gradient_accumulation_steps = 4
```

The text limits are starting values. Increase one only when cache construction reports that a generated description is longer.

## Build the Guide cache

```bash
cd ../RoboDojo/XPolicyLab/policy/Pi_05/openpi
uv sync
uv pip install -e "$VIDEO_HARNESS_ROOT/code/VideoHarness"
uv run python scripts/build_guide_materialization_cache.py
```

The cache stores the float32 GuideInput arrays used by training. Existing artifacts are reused when their Document identity and tensor settings match.

## Train guided Pi0.5

```bash
uv run python scripts/train_guided.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path /path/to/pi05_base/params \
  --experiment-name guided-task-pool \
  --run-dir "$GUIDED_RUN_ROOT" \
  --overwrite
```

The command inherits the official RoboDojo optimizer, learning-rate schedule, seed, EMA, 60,000 training steps, logging interval, checkpoint interval, FSDP setting, and W&B setting from the selected native config. The guided path changes only the Guide model, Guide data loader, microbatch shape, and gradient accumulation. CLI options remain available for Guidance paths and tensor limits.
