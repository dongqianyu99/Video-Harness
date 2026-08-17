#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <RoboDojo_lerobot_v30_video-root>" >&2
  exit 2
fi

dataset_root=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "${script_dir}/.." && pwd)

file_count=$(find "${dataset_root}" -type f | wc -l)
byte_count=$(find "${dataset_root}" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')
expected_files=603
expected_bytes=119958373793

if [[ "${file_count}" -ne "${expected_files}" ]]; then
  echo "Expected ${expected_files} files, found ${file_count}." >&2
  exit 1
fi
if [[ "${byte_count}" -ne "${expected_bytes}" ]]; then
  echo "Expected ${expected_bytes} bytes, found ${byte_count}." >&2
  exit 1
fi

cd "${project_dir}"
uv run video-harness inspect --dataset-root "${dataset_root}"
echo "RoboDojo full-data verification passed: ${file_count} files, ${byte_count} bytes"
