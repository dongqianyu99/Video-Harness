# Guide-conditioned server validation

This runbook covers the first Linux/GPU validation of the Actuator Guide path.
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
export GUIDE_DOCUMENTS_ROOT=/path/to/documents-openai
export GUIDE_SPLIT_ROOT=/path/to/video-harness-split-seed0
export GUIDE_TRAIN_SPLIT="$GUIDE_SPLIT_ROOT/training-split.json"
export GUIDE_TRAIN_PAIRS="$GUIDE_SPLIT_ROOT/train-pairs.jsonl"
export PI05_BASE_PARAMS=/path/to/pi05_base/params
export ROBODOJO_NORM_ASSET=/path/to/arx_x5_sim
export OPENPI_DATA_HOME=/path/to/openpi-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GUIDE_QUERY_EPISODE=0
export GUIDE_BATCH_SIZE=1
export GUIDE_GROUPS_PER_BATCH=1
export GUIDE_MAX_FRAMES=64
export GUIDE_MAX_UNITS=32
export GUIDE_MAX_TEXT_TOKENS=128
export GUIDED_RUN_ROOT=/path/to/runs/guided-server-debug

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

## Production data path

The real training loader uses a fixed `G guides x Q queries` batch, where
`batch_size = G * Q`. Query samples remain group-major and every query group
is bound to one immutable support document for the whole query episode. The
grouped sampler prefers different tasks and requires distinct support
documents inside a batch. `remainder_strategy=drop` discards incomplete query
groups explicitly. `pad_mask` preserves them by repeating only the storage
slot and setting `query_mask=false`; masked slots are processed but contribute
zero loss and zero gradient.

Each persistent data worker loads its VideoHarness bundle and tokenizer once,
decodes all unique frames of one Guide with one FFmpeg process, and keeps a
bounded in-memory LRU of fully materialized Guide tensors. PyTorch worker
prefetch overlaps native LeRobot/video work with training, while a separate
JAX queue asynchronously places upcoming grouped batches on device. These are
performance mechanisms only: they do not cache SigLIP features, change Guide
bindings, or modify the native Pi0.5 loss.

Every new training invocation uses one explicit run directory:

```text
<run-dir>/
├── run.json
├── logs/train.log
├── wandb/
├── checkpoints/
└── eval/
```

The terminal logger and the file logger receive the same training records.
When W&B is enabled, its local files and stable resume ID remain under this
run. Set `ROBODOJO_EVAL_ROOT=<run-dir>/eval` when evaluating a checkpoint from
the run so RoboDojo results and videos remain colocated.

Guide length buckets use repeated `--guide-length-bucket
MAX_UNITS:MAX_FRAMES` arguments. Every support document is assigned to the
smallest fitting bucket; batches and gradient-accumulation blocks never mix
buckets. Oversized Guides fail rather than being truncated. First run the data
benchmark with one generous `max_units/max_frames` budget and use its
`guide_length_summary` to choose bucket boundaries.
With strict `drop`, every bucket must contain at least `G` distinct support
documents; otherwise merge that sparse bucket with a neighbor or reduce `G`.

The official RoboDojo Pi0.5 recipe in the vendored release uses global batch
256, two FSDP devices, and 60,000 optimizer steps. In this single-process JAX
path, `batch_size=G*Q` is already the global microbatch distributed across the
devices—do not multiply it by the FSDP device count. Alignment is therefore:

```text
effective_global_batch = (G * Q) * gradient_accumulation_steps
```

Formal training checks this value against 256. Gradient accumulation applies
one optimizer update, one EMA update, and one step increment after all
microbatches. Strict alignment requires `remainder_strategy=drop`; `pad_mask`
reports its actual valid-query count and requires an explicit batch-mismatch
opt-out.

Start with `G=1`, one worker, and one manifest query episode for correctness. For the
first throughput run, use `G=2` or `G=4`, at least four workers, and omit
`--debug-query-episode-index` so that the loader can mix the full task set. Monitor
host RAM, `/dev/shm`, disk read rate, GPU utilization, and batch wait time.
The defaults cap each worker's Guide LRU at two entries and 256 MiB; multiply
that bound by the worker count when planning host memory. Fixed Guide budgets
remain fail-closed—there is no silent truncation.

## Gates

1. **M4d real batch.** Run `smoke_robodojo_guided_batch.py` with explicit
   `--dataset-root`, `--repo-id`, `--dataset-artifact`,
   `--documents-root`, `--pairs-artifact`, and one
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
6. **Closed-loop guided eval.** Only after a trained guided checkpoint exists,
   enable the `PI05_GUIDANCE_*` variables documented in
   `policy/Pi_05/README.md`. Confirm that the selected support document is the
   dataset-first episode for the task and that repeated eval episodes report a
   single Guide materialization.

## Commands

First validate the real data and artifact path without initializing Pi0.5:

```sh
uv run python scripts/smoke_robodojo_guided_batch.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --repo-id RoboDojo_lerobot_v30_video \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --dataset-artifact "$GUIDE_ARTIFACT_ROOT/dataset.json" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
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
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
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
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --pairs-artifact "$GUIDE_TRAIN_PAIRS" \
  --split-manifest "$GUIDE_TRAIN_SPLIT" \
  --debug-query-episode-index "$GUIDE_QUERY_EPISODE" \
  --batch-size "$GUIDE_BATCH_SIZE" \
  --guides-per-batch "$GUIDE_GROUPS_PER_BATCH" \
  --num-workers 1 \
  --prefetch-factor 2 \
  --device-prefetch-size 2 \
  --allow-effective-batch-mismatch \
  --max-frames "$GUIDE_MAX_FRAMES" \
  --max-units "$GUIDE_MAX_UNITS" \
  --max-text-tokens "$GUIDE_MAX_TEXT_TOKENS" \
  --experiment-name guided-server-debug \
  --run-dir "$GUIDED_RUN_ROOT" \
  --num-train-steps 10 \
  --log-interval 1 \
  --save-interval 5 \
  --fsdp-devices 1
```

After the single-episode debug passes, the corresponding full-corpus data
command removes `--debug-query-episode-index` and enables grouped workers, for
example:

```sh
uv run python scripts/train_guided.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path "$PI05_BASE_PARAMS" \
  --repo-id RoboDojo_lerobot_v30_video \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --dataset-artifact "$GUIDE_ARTIFACT_ROOT/dataset.json" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --pairs-artifact "$GUIDE_TRAIN_PAIRS" \
  --split-manifest "$GUIDE_TRAIN_SPLIT" \
  --batch-size 16 \
  --guides-per-batch 4 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --device-prefetch-size 2 \
  --gradient-accumulation-steps 16 \
  --reference-global-batch-size 256 \
  --remainder-strategy drop \
  --guide-cache-entries 2 \
  --guide-cache-max-bytes 268435456 \
  --max-frames "$GUIDE_MAX_FRAMES" \
  --max-units "$GUIDE_MAX_UNITS" \
  --max-text-tokens "$GUIDE_MAX_TEXT_TOKENS" \
  --experiment-name guided-full-data \
  --run-dir /path/to/runs/guided-full-data \
  --num-train-steps 60000 \
  --log-interval 100 \
  --save-interval 10000 \
  --fsdp-devices 2
```

The numeric batch/worker values above are a bring-up profile, not a published
training recommendation. Tune them only after recording batches/s, data-wait
time, peak host RAM, `/dev/shm` use, GPU memory, and GPU utilization.

Measure the loader independently before spending GPU time:

```sh
uv run python scripts/benchmark_guided_data_loader.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --repo-id RoboDojo_lerobot_v30_video \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --dataset-artifact "$GUIDE_ARTIFACT_ROOT/dataset.json" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --pairs-artifact "$GUIDE_TRAIN_PAIRS" \
  --split-manifest "$GUIDE_TRAIN_SPLIT" \
  --batch-size 16 \
  --guides-per-batch 4 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --gradient-accumulation-steps 16 \
  --remainder-strategy drop \
  --max-frames "$GUIDE_MAX_FRAMES" \
  --max-units "$GUIDE_MAX_UNITS" \
  --max-text-tokens "$GUIDE_MAX_TEXT_TOKENS" \
  --warmup-batches 8 \
  --measured-batches 50 \
  --output /path/to/logs/guided-data-benchmark.json
```

Repeat with worker counts `1, 2, 4, 8` while holding all other settings fixed.
Choose the smallest worker count that saturates throughput without excessive
RAM, `/dev/shm`, or disk contention; more workers are not automatically better.

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
