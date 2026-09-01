# Task-level Guidance Pool server validation

This guide validates the document-only RoboDojo path:

```text
documents-root -> accepted Guide catalog -> task-level G x Q batch
-> three-view Boundary memory + text-only Transition memory -> GuidePi0
```

Local CPU tests establish contracts only. Real corpus decoding, GPU training,
and simulator evaluation remain separate gates.

## Environment

```bash
export OPENPI_DIR=/path/to/Video-Harness/code/RoboDojo/XPolicyLab/policy/Pi_05/openpi
export VIDEO_HARNESS_DIR=/path/to/Video-Harness/code/VideoHarness
export ROBODOJO_DATASET_ROOT=/path/to/RoboDojo_lerobot_v30_video
export GUIDE_DOCUMENTS_ROOT=/path/to/documents-openai
export GUIDE_MATERIALIZATION_CACHE_ROOT=/path/to/cache/robodojo-guides
export PI05_BASE_PARAMS=/path/to/pi05_base/params
export GUIDED_RUN_ROOT=/path/to/runs/guided-task-pool

export NATIVE_CONFIG=pi05_base_aloha_full_sim_arx-x5_seed_0
export LEROBOT_REPO_ID=RoboDojo_sim_arx-x5_v30
export GUIDE_GROUPS=4
export QUERIES_PER_GUIDE=64
export MAX_BOUNDARIES=64
export MAX_UNITS=32
export MAX_BOUNDARY_TEXT_TOKENS=128
export MAX_TRANSITION_TEXT_TOKENS=128
export BOUNDARY_QUERIES=8
export TRANSITION_QUERIES=4
```

The structural and text budgets must cover the generated corpus. Overflow is
an error; the loader never truncates Guidance. `BOUNDARY_QUERIES=8` and
`TRANSITION_QUERIES=4` are the current capacities and must match the checkpoint.

Install VideoHarness in the OpenPI environment and verify the source assets:

```bash
cd "$OPENPI_DIR"
uv pip install -e "$VIDEO_HARNESS_DIR"
ffmpeg -version
test -d "$ROBODOJO_DATASET_ROOT"
test -d "$GUIDE_DOCUMENTS_ROOT"
```

Build the persistent exact-float32 GuideInput cache once. The command fully
validates existing artifacts and invokes FFmpeg only for missing or corrupt
artifacts:

```bash
uv run python scripts/build_guide_materialization_cache.py \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --guide-materialization-cache-root "$GUIDE_MATERIALIZATION_CACHE_ROOT" \
  --max-boundaries "$MAX_BOUNDARIES" \
  --max-units "$MAX_UNITS" \
  --max-boundary-text-tokens "$MAX_BOUNDARY_TEXT_TOKENS" \
  --max-transition-text-tokens "$MAX_TRANSITION_TEXT_TOKENS" \
  --guide-boundary-num-queries "$BOUNDARY_QUERIES" \
  --guide-transition-num-queries "$TRANSITION_QUERIES"
```

The cache is content-addressed by the Document, GuidePlan, tokenizer, and
materialization contract. Existing cache artifacts are authoritative: source
video bytes are not revalidated after a successful build.

## Gate 1: one real task-pool batch

```bash
uv run python scripts/smoke_robodojo_guided_batch.py \
  --native-config-name "$NATIVE_CONFIG" \
  --repo-id "$LEROBOT_REPO_ID" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --guide-materialization-cache-root "$GUIDE_MATERIALIZATION_CACHE_ROOT" \
  --guides-per-batch "$GUIDE_GROUPS" \
  --queries-per-guide "$QUERIES_PER_GUIDE" \
  --max-boundaries "$MAX_BOUNDARIES" \
  --max-units "$MAX_UNITS" \
  --max-boundary-text-tokens "$MAX_BOUNDARY_TEXT_TOKENS" \
  --max-transition-text-tokens "$MAX_TRANSITION_TEXT_TOKENS" \
  --guide-boundary-num-queries "$BOUNDARY_QUERIES" \
  --guide-transition-num-queries "$TRANSITION_QUERIES"
```

Require:

- accepted/excluded catalog counts are plausible;
- every valid Guide has three Boundary views;
- G and Q match the request;
- sampler statistics contain no task mismatch or silent padding;
- image values are finite and within the normalized range.

## Gate 2: forward/backward

```bash
uv run python scripts/smoke_guided_forward_backward.py \
  --native-config-name "$NATIVE_CONFIG" \
  --base-params-path "$PI05_BASE_PARAMS" \
  --repo-id "$LEROBOT_REPO_ID" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --guide-materialization-cache-root "$GUIDE_MATERIALIZATION_CACHE_ROOT" \
  --guides-per-batch "$GUIDE_GROUPS" \
  --queries-per-guide "$QUERIES_PER_GUIDE" \
  --max-boundaries "$MAX_BOUNDARIES" \
  --max-units "$MAX_UNITS" \
  --max-boundary-text-tokens "$MAX_BOUNDARY_TEXT_TOKENS" \
  --max-transition-text-tokens "$MAX_TRANSITION_TEXT_TOKENS" \
  --guide-boundary-num-queries "$BOUNDARY_QUERIES" \
  --guide-transition-num-queries "$TRANSITION_QUERIES" \
  --fsdp-devices 1 \
  --no-optimizer-update
```

Require finite loss and gradients, nonzero Guide-encoder gradients, and the
expected `[G,Q]` batch plus Boundary/Transition tensor shapes. Then repeat
without `--no-optimizer-update` and on the intended single-process mesh.

## Gate 3: throughput

```bash
uv run python scripts/benchmark_guided_data_loader.py \
  --native-config-name "$NATIVE_CONFIG" \
  --repo-id "$LEROBOT_REPO_ID" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --guide-materialization-cache-root "$GUIDE_MATERIALIZATION_CACHE_ROOT" \
  --guides-per-batch "$GUIDE_GROUPS" \
  --queries-per-guide "$QUERIES_PER_GUIDE" \
  --max-boundaries "$MAX_BOUNDARIES" \
  --max-units "$MAX_UNITS" \
  --max-boundary-text-tokens "$MAX_BOUNDARY_TEXT_TOKENS" \
  --max-transition-text-tokens "$MAX_TRANSITION_TEXT_TOKENS" \
  --guide-boundary-num-queries "$BOUNDARY_QUERIES" \
  --guide-transition-num-queries "$TRANSITION_QUERIES" \
  --num-workers 8 \
  --warmup-batches 8 \
  --measured-batches 50 \
  --output /path/to/logs/guided-data-benchmark.json
```

Record data wait p50/p95, batch bytes, persistent-cache build/reuse counts,
Guide padding, and valid/padded query counts. The parent fully validates every
compact artifact before workers start. Workers receive only artifact records
and the fixed materialization shape; they do not receive Documents, GuidePlans,
tokenizers, media routes, or FFmpeg loaders. Their bounded in-memory LRU caches
expanded GuideInput values by document ID and never caches trainable features.

## Gate 4: tracked training

```bash
uv run python scripts/train_guided.py \
  --native-config-name "$NATIVE_CONFIG" \
  --base-params-path "$PI05_BASE_PARAMS" \
  --repo-id "$LEROBOT_REPO_ID" \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --documents-root "$GUIDE_DOCUMENTS_ROOT" \
  --guide-materialization-cache-root "$GUIDE_MATERIALIZATION_CACHE_ROOT" \
  --guides-per-batch "$GUIDE_GROUPS" \
  --queries-per-guide "$QUERIES_PER_GUIDE" \
  --max-boundaries "$MAX_BOUNDARIES" \
  --max-units "$MAX_UNITS" \
  --max-boundary-text-tokens "$MAX_BOUNDARY_TEXT_TOKENS" \
  --max-transition-text-tokens "$MAX_TRANSITION_TEXT_TOKENS" \
  --guide-boundary-num-queries "$BOUNDARY_QUERIES" \
  --guide-transition-num-queries "$TRANSITION_QUERIES" \
  --gradient-accumulation-steps 1 \
  --reference-global-batch-size 256 \
  --num-workers 8 \
  --num-train-steps 60000 \
  --log-interval 100 \
  --save-interval 1000 \
  --fsdp-devices 2 \
  --experiment-name guided-task-pool \
  --run-dir "$GUIDED_RUN_ROOT" \
  --wandb-enabled
```

The effective batch is `G * Q * gradient_accumulation_steps`. Resume is valid
only when the catalog, task-sample index, camera order, shared maximum Guide
shape, token budgets, and Boundary/Transition capacities match the recorded run. Resume restores
TrainState, optimizer, EMA, and step but starts a fresh sampler shuffle.

## Gate 5: closed-loop evaluation

Evaluation filters the catalog first, then uses the first accepted document in
stable `(source_episode_index, document_id)` order for each task. Record that
document ID with every result. Compare matched simulator seeds for aligned,
no-Guide, stage-shuffled, and same-task mismatched Guidance. Offline loss alone
does not establish that the policy uses stage-aligned Guidance.

Capacity selection compares the default `(8,4)` against a higher-capacity
reference under identical training streams and seeds. Keep `(8,4)` only if its
paired closed-loop success is non-inferior within the preregistered margin.
