#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--metadata-only]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_root=${VIDEO_HARNESS_ROOT:-$(cd "${script_dir}/.." && pwd)}
mode=${1:-}
if [[ -n "${mode}" && "${mode}" != "--metadata-only" ]]; then
  echo "Unknown option: ${mode}" >&2
  exit 2
fi

include_pattern="data/RoboDojo_lerobot_v30_video/**"
if [[ "${mode}" == "--metadata-only" ]]; then
  include_pattern="data/RoboDojo_lerobot_v30_video/meta/**"
fi

mkdir -p "${workspace_root}/data"
hf download RoboDojo-Benchmark/RoboDojo \
  --type dataset \
  --revision "${ROBODOJO_REVISION:-e2f40904e7b039b46e512e1443ad5055984d3344}" \
  --include "${include_pattern}" \
  --local-dir "${workspace_root}" \
  --max-workers "${HF_MAX_WORKERS:-8}"

dataset_root="${workspace_root}/data/RoboDojo_lerobot_v30_video"
test -f "${dataset_root}/meta/info.json"
echo "RoboDojo dataset root: ${dataset_root}"
