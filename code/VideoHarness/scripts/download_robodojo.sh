#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <local-repo-root> [--metadata-only]" >&2
  exit 2
fi

repo_root=$1
mode=${2:-}
if [[ -n "${mode}" && "${mode}" != "--metadata-only" ]]; then
  echo "Unknown option: ${mode}" >&2
  exit 2
fi

include_pattern="data/RoboDojo_lerobot_v30_video/**"
if [[ "${mode}" == "--metadata-only" ]]; then
  include_pattern="data/RoboDojo_lerobot_v30_video/meta/**"
fi

mkdir -p "${repo_root}"
uv run hf download RoboDojo-Benchmark/RoboDojo \
  --type dataset \
  --revision main \
  --include "${include_pattern}" \
  --local-dir "${repo_root}" \
  --max-workers "${HF_MAX_WORKERS:-8}"

dataset_root="${repo_root}/data/RoboDojo_lerobot_v30_video"
test -f "${dataset_root}/meta/info.json"
echo "RoboDojo dataset root: ${dataset_root}"
