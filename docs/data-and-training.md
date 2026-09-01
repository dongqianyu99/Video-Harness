# Standard Data and Guided-Training Workflow

Run this workflow from the repository root. Source `scripts/env.sh` in every new
shell; it makes VideoHarness, LeRobot, and guided Pi0.5 resolve the same physical
dataset instead of maintaining separate `repo_id` and `dataset_root` locations.

## Canonical layout

```text
Video-Harness/
├── data/
│   ├── RoboDojo_lerobot_v30_video/       # official LeRobot v3 dataset
│   ├── video-harness/default/            # compiler manifests and Documents
│   │   └── documents-openai/             # accepted/quarantined per-episode files
│   ├── guide-cache/default/               # persistent compact GuideInput artifacts
│   └── runs/guided-pi05/                  # checkpoints, logs, W&B, eval
├── code/VideoHarness/
└── code/RoboDojo/
```

`data/` is ignored by Git. Do not put datasets, provider credentials, checkpoints,
or generated Documents under `code/`.

```bash
source scripts/env.sh
printf '%s\n' "$ROBODOJO_DATASET_ROOT" "$GUIDE_DOCUMENTS_ROOT"
```

The standard LeRobot identity is `RoboDojo_lerobot_v30_video`, and
`HF_LEROBOT_HOME=$VIDEO_HARNESS_ROOT/data`. Therefore OpenPI's
`repo_id=RoboDojo_lerobot_v30_video` and VideoHarness's dataset root refer to the
same directory.

## Download and verify RoboDojo

Install the current `hf` CLI, then download the pinned public dataset revision:

```bash
scripts/download_robodojo.sh
scripts/verify_robodojo.sh
```

The full joint-space dataset is approximately 120 GB. For metadata-only inspection:

```bash
scripts/download_robodojo.sh --metadata-only
```

The default revision is
`e2f40904e7b039b46e512e1443ad5055984d3344`; override it deliberately with
`ROBODOJO_REVISION`. The source is the official
[RoboDojo Hugging Face dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/data/RoboDojo_lerobot_v30_video).

## Compile Guidance Documents

```bash
cd code/VideoHarness
uv sync --extra providers

uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VIDEO_HARNESS_RUN_ROOT" \
  --sample-hz 1

uv run video-harness annotate \
  --provider openai \
  --model "$VH_MODEL" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents "$VIDEO_HARNESS_RUN_ROOT/documents.jsonl" \
  --output "$VIDEO_HARNESS_RUN_ROOT/documents.openai.jsonl" \
  --document-root "$GUIDE_DOCUMENTS_ROOT" \
  --checkpoint-root "$VIDEO_HARNESS_RUN_ROOT/checkpoints-openai" \
  --workers 4
```

Use a one-episode debug run before the full provider job. Resume with exactly the
same dataset root, sampling, provider, model, shard, and checkpoint configuration.

## Real structural limits

The official v3 metadata reports 3,500 episodes and 1,859,602 frames. VideoHarness
excludes the 100 auxiliary DLC episodes, leaving 3,400 benchmark Documents. At the
standard 25 FPS and `sample_hz=1`, a length-`L` episode produces
`ceil((L-1)/25)` Units and one additional Boundary.

| Statistic | Episode frames | Units |
|---|---:|---:|
| minimum | 105 | 5 |
| median | 459 | 18 |
| p95 | 1,171 | 48 |
| p99 | 1,419 | 57 |
| maximum | 1,537 | 62 |

The exact full-corpus result is 73,248 Units with a maximum of 62 Units and 63
Boundaries. These figures are derived from the official
[episode metadata parquet](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/blob/main/data/RoboDojo_lerobot_v30_video/meta/episodes/chunk-000/file-000.parquet)
and VideoHarness's current uniform-boundary rule.

Use these defaults for the single-shape path:

```text
max_boundaries = 63
max_units = 62
max_boundary_text_tokens = 128   # provisional until real Documents are tokenized
max_transition_text_tokens = 128 # provisional until real Documents are tokenized
K_B = 8
K_T = 4
G = 1
Q = 64
gradient_accumulation_steps = 4
effective batch = 256
```

The episode lengths are not actually uniform: the maximum Unit count is over three
times the median. A single maximum shape therefore spends substantial compute on
padding. `G=1,Q=64,accumulation=4` keeps the same effective batch and four Guide
draws per optimizer step while reducing peak HBM relative to `G=4,Q=64`.

## Build and validate the Guide cache

```bash
cd code/RoboDojo/XPolicyLab/policy/Pi_05/openpi
uv sync
uv pip install -e "$VIDEO_HARNESS_ROOT/code/VideoHarness"

uv run python scripts/build_guide_materialization_cache.py
uv run python scripts/smoke_robodojo_guided_batch.py
```

The first command invokes FFmpeg only for missing or corrupt artifacts. Subsequent
training and evaluation read the exact float32 GuideInput cache. Cache reuse validates
the artifacts themselves and intentionally does not revalidate source video bytes.

## Guided Pi0.5 smoke and training

```bash
uv run python scripts/smoke_guided_forward_backward.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path /path/to/pi05_base/params \
  --fsdp-devices 2 \
  --no-optimizer-update

uv run python scripts/train_guided.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path /path/to/pi05_base/params \
  --experiment-name guided-task-pool \
  --run-dir "$GUIDED_RUN_ROOT" \
  --num-train-steps 60000 \
  --fsdp-devices 2 \
  --wandb-enabled
```

Path, structural, text, G/Q, and accumulation options default to the values above
and remain explicitly overridable. Before the full run, require a real 200-step GPU
gate with peak HBM below 90% and dataloader p95 wait below 10% of step time.
