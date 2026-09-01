# Video Harness

Video Harness turns RoboDojo demonstration episodes into task-level Guidance Documents. RoboDojo then uses those Documents to fine-tune a Guide-conditioned Pi0.5 policy.

## Quick start

```bash
source scripts/env.sh
scripts/download_robodojo.sh

cd code/VideoHarness
uv sync --extra providers
uv run video-harness build --sample-hz 1
```

Annotate the planned Documents with your provider, then build the Guide cache:

```bash
uv run video-harness annotate \
  --provider openai \
  --model "$VH_MODEL" \
  --workers 4

cd ../RoboDojo/XPolicyLab/policy/Pi_05/openpi
uv sync
uv pip install -e "$VIDEO_HARNESS_ROOT/code/VideoHarness"
uv run python scripts/build_guide_materialization_cache.py
```

The guided training command uses the standard data paths and the 1 Hz structural limits by default:

```bash
uv run python scripts/train_guided.py \
  --native-config-name pi05_base_aloha_full_sim_arx-x5_seed_0 \
  --base-params-path /path/to/pi05_base/params \
  --experiment-name guided-task-pool \
  --run-dir "$GUIDED_RUN_ROOT"
```

See [Getting started](docs/getting-started.md) for setup and command details. See [Architecture](docs/architecture.md) for the Document and guided Pi0.5 interfaces.
