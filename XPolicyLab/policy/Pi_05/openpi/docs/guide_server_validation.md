# Guide-conditioned server validation

This runbook covers the first Linux/GPU validation of the Actuator v0 Guide path.
It is intentionally limited to one real batch, one forward/backward pass, and
short debugging runs. All paths below are placeholders and must be supplied
explicitly for the server; do not infer them from the working directory.

## Environment

Install VideoHarness into the OpenPI environment using its actual checkout:

```text
uv pip install -e /path/to/VideoHarness
```

Confirm that the OpenPI and VideoHarness revisions, FFmpeg availability, model
checkpoint, norm-stat assets, RoboDojo dataset, and three artifact files are
recorded in the run log. Never put credentials or tokens in commands or logs.

Define the server-specific paths once. These names are examples; every value
must be replaced explicitly:

```sh
export OPENPI_DIR=/path/to/RoboDojo/XPolicyLab/policy/Pi_05/openpi
export VIDEO_HARNESS_DIR=/path/to/VideoHarness
export HF_LEROBOT_HOME=/path/to/lerobot-datasets
export ROBODOJO_DATASET_ROOT=/path/to/RoboDojo_lerobot_v30_video
export GUIDE_ARTIFACT_ROOT=/path/to/Guidance-Documents/RoboDojo
export GUIDE_DOCUMENTS=/path/to/annotated-documents.jsonl
export PI05_BASE_PARAMS=/path/to/pi05_base/params
export ROBODOJO_NORM_ASSET=/path/to/arx_x5_sim
export OPENPI_DATA_HOME=/path/to/openpi-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GUIDE_QUERY_EPISODE=0
export GUIDE_BATCH_SIZE=1
export GUIDE_MAX_FRAMES=64
export GUIDE_MAX_UNITS=32
export GUIDE_MAX_TEXT_TOKENS=128

cd "$OPENPI_DIR"
uv pip install -e "$VIDEO_HARNESS_DIR"
ffmpeg -version
test -d "$HF_LEROBOT_HOME/RoboDojo_lerobot_v30_video"
test -f "$OPENPI_DATA_HOME/gs/big_vision/paligemma_tokenizer.model"
```

The three Guide budgets above are not recommended scientific defaults. Set
them from the generated corpus report; overflow is intentionally an error.
`ROBODOJO_DATASET_ROOT` and
`$HF_LEROBOT_HOME/RoboDojo_lerobot_v30_video` must identify the same dataset.
The native RoboDojo config expects the normalization asset at
`$OPENPI_DIR/assets/RoboDojo_assets/arx_x5_sim`; copy or symlink the explicit
`ROBODOJO_NORM_ASSET` there before running, and record its source. On an offline
server, pre-populate `OPENPI_DATA_HOME` before requesting GPU resources; the
smoke scripts must not rely on a runtime download.

## Gates

1. **M4d real batch.** Run `smoke_robodojo_guided_batch.py` with explicit
   `--dataset-root`, `--repo-id`, `--dataset-artifact`,
   `--documents-artifact`, `--pairs-artifact`, and one
   `--query-episode-index`. Confirm support-only decoding, native norm stats,
   `G=1,Q=batch_size`, frame/text/unit masks, and finite shapes.
2. **M5 forward/backward.** After the batch gate passes, run
   `smoke_guided_forward_backward.py --no-optimizer-update` with the exact
   same data/artifact arguments and an explicit `--base-params-path`. Confirm
   strict `pi05_base` loading, finite loss, and both Guide/native gradient
   summaries.
3. **One optimizer update.** Remove `--no-optimizer-update` only after the
   no-update pass is stable. Save the complete output and peak-memory report.
4. **Multi-device check.** Run the single-update test on one device first,
   then validate the multi-device mesh and Q-axis sharding. The requested
   batch size must be divisible by the data-device count. Do not start a long
   run until this gate passes.
5. **Short training debug.** Use `train_guided.py` for a small, explicit step
   count. Verify checkpoint save/restore and EMA before considering a longer
   experiment.

## Commands

First validate the real data and artifact path without initializing Pi0.5:

```sh
uv run python scripts/smoke_robodojo_guided_batch.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --repo-id RoboDojo_lerobot_v30_video \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --dataset-artifact "$GUIDE_ARTIFACT_ROOT/dataset.json" \
  --documents-artifact "$GUIDE_DOCUMENTS" \
  --pairs-artifact "$GUIDE_ARTIFACT_ROOT/pairs.jsonl" \
  --query-episode-index "$GUIDE_QUERY_EPISODE" \
  --batch-size "$GUIDE_BATCH_SIZE" \
  --max-frames "$GUIDE_MAX_FRAMES" \
  --max-units "$GUIDE_MAX_UNITS" \
  --max-text-tokens "$GUIDE_MAX_TEXT_TOKENS"
```

Then run exactly one sharded forward/backward without creating optimizer
updates:

```sh
uv run python scripts/smoke_guided_forward_backward.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path "$PI05_BASE_PARAMS" \
  --repo-id RoboDojo_lerobot_v30_video \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --dataset-artifact "$GUIDE_ARTIFACT_ROOT/dataset.json" \
  --documents-artifact "$GUIDE_DOCUMENTS" \
  --pairs-artifact "$GUIDE_ARTIFACT_ROOT/pairs.jsonl" \
  --query-episode-index "$GUIDE_QUERY_EPISODE" \
  --batch-size "$GUIDE_BATCH_SIZE" \
  --max-frames "$GUIDE_MAX_FRAMES" \
  --max-units "$GUIDE_MAX_UNITS" \
  --max-text-tokens "$GUIDE_MAX_TEXT_TOKENS" \
  --fsdp-devices 1 \
  --no-optimizer-update
```

Only after that succeeds, remove `--no-optimizer-update` for one optimizer
step. For a two-device FSDP test, set `--fsdp-devices 2` and use a batch size
divisible by two. Do not change the dataset, query episode, Guide budgets, or
checkpoint between the single- and multi-device comparisons.

Finally, a bounded training debug run can use:

```sh
uv run python scripts/train_guided.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path "$PI05_BASE_PARAMS" \
  --repo-id RoboDojo_lerobot_v30_video \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --dataset-artifact "$GUIDE_ARTIFACT_ROOT/dataset.json" \
  --documents-artifact "$GUIDE_DOCUMENTS" \
  --pairs-artifact "$GUIDE_ARTIFACT_ROOT/pairs.jsonl" \
  --query-episode-index "$GUIDE_QUERY_EPISODE" \
  --batch-size "$GUIDE_BATCH_SIZE" \
  --max-frames "$GUIDE_MAX_FRAMES" \
  --max-units "$GUIDE_MAX_UNITS" \
  --max-text-tokens "$GUIDE_MAX_TEXT_TOKENS" \
  --experiment-name guided-server-debug \
  --checkpoint-dir /path/to/guided-debug-checkpoint \
  --num-train-steps 10 \
  --log-interval 1 \
  --save-interval 5 \
  --fsdp-devices 1
```

## Failure handling

Stop at the first failed gate. Preserve the command, structured stdout, stack
trace, device information, and relevant artifact build/schema identifiers.
Check dataset/artifact identity, episode ranges, FPS/video paths, FFmpeg,
norm-stat availability, checkpoint type, and batch divisibility before changing
model code. The base checkpoint and a guided resume checkpoint are distinct and
must never be substituted for one another.

## Evidence labels

- **VERIFIED:** fake/local contract tests or a server gate that actually ran.
- **NOT RUN:** a real-data or GPU gate not attempted, including all real
  RoboDojo tests on a MacBook.
- **BLOCKED:** a required asset or dependency is unavailable, such as a
  dataset, artifact, norm stats, FFmpeg, checkpoint, or GPU mesh.

Passing local CPU tests does not certify real data, full-model initialization,
GPU memory behavior, multi-device sharding, or task success. Do not report
rollout success or training quality from these validation gates.
