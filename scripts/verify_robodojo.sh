#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [RoboDojo_lerobot_v30_video-root]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "${script_dir}/.." && pwd)
dataset_root=${1:-${project_dir}/data/RoboDojo_lerobot_v30_video}

read -r file_count byte_count < <(
  python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); fs=[x for x in p.rglob("*") if x.is_file()]; print(len(fs), sum(x.stat().st_size for x in fs))' "${dataset_root}"
)
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

cd "${project_dir}/code/VideoHarness"
uv run video-harness inspect --dataset-root "${dataset_root}"
echo "RoboDojo full-data verification passed: ${file_count} files, ${byte_count} bytes"
