# Video Harness

Video Harness compiles RoboDojo demonstrations into task-level Guidance Documents
and uses them to fine-tune a Guide-conditioned Pi0.5 policy.

## Quick start

```bash
source scripts/env.sh
scripts/download_robodojo.sh
scripts/verify_robodojo.sh

cd code/VideoHarness
uv sync --extra providers
uv run video-harness build \
  --dataset-root "$ROBODOJO_DATASET_ROOT" \
  --output-root "$VIDEO_HARNESS_RUN_ROOT" \
  --sample-hz 1
```

After annotation, build the persistent Guide cache and run the guided-training
smokes before starting a full job. See the [standard data and training
workflow](docs/data-and-training.md) for the exact commands and recommended
`63 Boundaries / 62 Units` structural limits.

- [Documentation index](docs/index.md)
- [VideoHarness compiler guide](docs/getting-started.md)
- [Guided Pi0.5 server validation](code/RoboDojo/XPolicyLab/policy/Pi_05/openpi/docs/guide_server_validation.md)
