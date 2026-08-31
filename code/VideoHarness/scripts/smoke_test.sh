#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <RoboDojo_lerobot_v30_video-root> <output-root>" >&2
  exit 2
fi

dataset_root=$1
output_root=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "${script_dir}/.." && pwd)
cd "${project_dir}"

uv run video-harness inspect --dataset-root "${dataset_root}"
uv run video-harness build \
  --dataset-root "${dataset_root}" \
  --output-root "${output_root}" \
  --sample-hz 1 \
  --max-tasks 1 \
  --episodes-per-task 3
uv run video-harness annotate \
  --provider mock \
  --dataset-root "${dataset_root}" \
  --documents "${output_root}/documents.jsonl" \
  --output "${output_root}/documents.mock.jsonl"
uv run video-harness report \
  --documents "${output_root}/documents.mock.jsonl"

echo "Metadata, mock-evidence, and report smoke test passed: ${output_root}"
