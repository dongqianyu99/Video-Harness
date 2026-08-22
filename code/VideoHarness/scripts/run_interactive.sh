#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_interactive.sh

Interactive end-to-end Video Harness runner:
  1. prepare the uv environment;
  2. download and validate RoboDojo with the official `hf` CLI;
  3. build behavior-document sidecars;
  4. select OpenAI, Anthropic Claude, or mock annotation;
  5. annotate, validate, and report the result.

OpenAI and Anthropic tokens are read interactively and never saved. Real API
processing requires network access. Generated artifacts remain in the selected
local output directory.
EOF
}

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 2
fi
if [[ ! -t 0 ]]; then
  echo "This runner requires an interactive terminal." >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "${script_dir}/.." && pwd)
repository_root=$(cd "${project_dir}/../.." && pwd)

prompt_default() {
  local prompt=$1
  local default=$2
  local value
  read -r -p "${prompt} [${default}]: " value
  printf '%s' "${value:-${default}}"
}

require_positive_integer() {
  local value=$1
  local field=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${field} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

confirm() {
  local prompt=$1
  local answer
  read -r -p "${prompt} [y/N]: " answer
  [[ "${answer}" =~ ^[Yy]$ ]]
}

confirm_yes() {
  local prompt=$1
  local answer
  read -r -p "${prompt} [Y/n]: " answer
  [[ -z "${answer}" || "${answer}" =~ ^[Yy]$ ]]
}

install_uv_if_needed() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  echo "uv was not found. The official uv installer is required."
  if ! confirm "Install uv into the current user account now?"; then
    echo "Install uv first, then rerun this script." >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation finished, but uv is not on PATH." >&2
    echo "Open a new shell or add ${HOME}/.local/bin to PATH, then rerun." >&2
    exit 1
  fi
}

install_ffmpeg_if_needed() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return
  fi
  echo "ffmpeg is required to read the referenced RoboDojo frames."
  if ! confirm "Install ffmpeg with apt now?"; then
    echo "Install ffmpeg with this machine's package manager, then rerun." >&2
    exit 1
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Automatic installation currently supports apt-based Linux only." >&2
    exit 1
  fi
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
  else
    echo "Root or sudo access is required to install ffmpeg automatically." >&2
    exit 1
  fi
}

cleanup_secrets() {
  unset api_token OPENAI_API_KEY ANTHROPIC_API_KEY 2>/dev/null || true
}
trap cleanup_secrets EXIT
trap 'cleanup_secrets; exit 130' INT
trap 'cleanup_secrets; exit 143' TERM

echo
echo "== Video Harness: environment =="
install_uv_if_needed
install_ffmpeg_if_needed
cd "${project_dir}"
uv sync --extra dev --extra providers
uv run pytest -q
uv run hf version
ffmpeg -version | head -n 1

echo
echo "== Video Harness: RoboDojo download =="
echo "The joint-space LeRobot v3 release requires approximately 120 GB."
default_repo_root="${repository_root}/data/robodojo-hf"
robodojo_repo_root=$(prompt_default "RoboDojo download directory" "${default_repo_root}")
mkdir -p "${robodojo_repo_root}"
df -h "${robodojo_repo_root}" | tail -n 1

dataset_root="${robodojo_repo_root}/data/RoboDojo_lerobot_v30_video"
if [[ -f "${dataset_root}/meta/info.json" ]]; then
  echo "An existing dataset was found. The hf downloader can verify and resume it."
fi
if ! confirm_yes "Download or resume the official RoboDojo files now?"; then
  if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
    echo "No reusable RoboDojo dataset was found at ${dataset_root}." >&2
    exit 1
  fi
else
  scripts/download_robodojo.sh "${robodojo_repo_root}"
fi
scripts/verify_robodojo.sh "${dataset_root}"

echo
echo "== Video Harness: annotation provider =="
cat <<'EOF'
Choose a provider:
  1) OpenAI
  2) Anthropic Claude
  3) Mock (no token; validates the pipeline but produces no trainable evidence)
EOF
read -r -p "Provider [1]: " provider_choice
provider_choice=${provider_choice:-1}

provider=
model=
case "${provider_choice}" in
  1)
    provider="openai"
    read -r -s -p "OpenAI API token (hidden; not saved): " api_token
    echo
    if [[ -z "${api_token}" ]]; then
      echo "The OpenAI token cannot be empty." >&2
      exit 2
    fi
    export OPENAI_API_KEY="${api_token}"
    read -r -p "Vision-capable OpenAI model ID: " model
    ;;
  2)
    provider="anthropic"
    read -r -s -p "Anthropic API token (hidden; not saved): " api_token
    echo
    if [[ -z "${api_token}" ]]; then
      echo "The Anthropic token cannot be empty." >&2
      exit 2
    fi
    export ANTHROPIC_API_KEY="${api_token}"
    read -r -p "Vision-capable Claude model ID: " model
    ;;
  3)
    provider="mock"
    ;;
  *)
    echo "Unknown provider choice: ${provider_choice}" >&2
    exit 2
    ;;
esac
if [[ "${provider}" != "mock" && -z "${model}" ]]; then
  echo "A model ID is required for ${provider}." >&2
  exit 2
fi

echo
echo "== Video Harness: processing scale =="
cat <<'EOF'
  1) API smoke (recommended first): build 3 episodes, annotate 3 units
  2) One-task pilot: build and annotate 3 complete episodes
  3) All benchmark data: annotate every 1 Hz unit in all 3,400 benchmark episodes
  4) Custom subset
EOF
read -r -p "Scale [1]: " scale_choice
scale_choice=${scale_choice:-1}

build_limits=()
annotation_limits=()
case "${scale_choice}" in
  1)
    build_limits=(--max-tasks 1 --episodes-per-task 3)
    annotation_limits=(--limit-documents 1 --limit-units-per-document 3)
    ;;
  2)
    build_limits=(--max-tasks 1 --episodes-per-task 3)
    ;;
  3)
    if [[ "${provider}" == "mock" ]]; then
      echo "A full mock run makes no external API calls."
    else
      echo "WARNING: a full 1 Hz run involves tens of thousands of paid VLM calls."
    fi
    read -r -p "Type FULL to continue: " full_confirmation
    if [[ "${full_confirmation}" != "FULL" ]]; then
      echo "Full processing was not confirmed." >&2
      exit 1
    fi
    ;;
  4)
    max_tasks=$(prompt_default "Maximum task count" "1")
    episodes_per_task=$(prompt_default "Episodes per task (at least 2)" "3")
    require_positive_integer "${max_tasks}" "Maximum task count"
    require_positive_integer "${episodes_per_task}" "Episodes per task"
    if (( episodes_per_task < 2 )); then
      echo "Episodes per task must be at least 2 for support/query pairing." >&2
      exit 2
    fi
    build_limits=(--max-tasks "${max_tasks}" --episodes-per-task "${episodes_per_task}")
    limit_documents=$(prompt_default "Documents to annotate; 0 means all built documents" "1")
    limit_units=$(prompt_default "Units per annotated document; 0 means all units" "3")
    if [[ ! "${limit_documents}" =~ ^[0-9]+$ || ! "${limit_units}" =~ ^[0-9]+$ ]]; then
      echo "Annotation limits must be non-negative integers." >&2
      exit 2
    fi
    if (( limit_documents > 0 )); then
      annotation_limits+=(--limit-documents "${limit_documents}")
    fi
    if (( limit_units > 0 )); then
      annotation_limits+=(--limit-units-per-document "${limit_units}")
    fi
    ;;
  *)
    echo "Unknown processing scale: ${scale_choice}" >&2
    exit 2
    ;;
esac

echo
echo "== Video Harness: diagnostic artifacts =="
debug_args=()
if confirm "Enable debug mode and retain Unit videos, frames, sheets, crops, and provider outputs?"; then
  debug_args=(--debug)
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
default_output="${repository_root}/run/video-harness-${provider}-${timestamp}"
output_root=$(prompt_default "New output directory" "${default_output}")
if [[ -e "${output_root}" ]]; then
  if [[ ! -d "${output_root}" || -n "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "The output path must be new or empty: ${output_root}" >&2
    exit 1
  fi
fi

echo
echo "== Video Harness: build and frame check =="
uv run video-harness build \
  --dataset-root "${dataset_root}" \
  --output-root "${output_root}" \
  --sample-hz 1 \
  --supports-per-query 1 \
  "${build_limits[@]}"
uv run video-harness decode-smoke \
  --dataset-root "${dataset_root}" \
  --documents "${output_root}/documents.jsonl" \
  --limit-frames 4

echo
echo "== Video Harness: annotate and validate =="
annotation_output="${output_root}/documents.${provider}.jsonl"
annotation_command=(
  uv run video-harness annotate
  --provider "${provider}"
  --dataset-root "${dataset_root}"
  --documents "${output_root}/documents.jsonl"
  --output "${annotation_output}"
)
if [[ "${provider}" != "mock" ]]; then
  annotation_command+=(--model "${model}")
fi
if [[ ${#debug_args[@]} -gt 0 ]]; then
  annotation_command+=("${debug_args[@]}" --debug-root "${output_root}/debug")
fi
annotation_command+=("${annotation_limits[@]}")
"${annotation_command[@]}"
uv run video-harness report --documents "${annotation_output}"

cleanup_secrets
trap - EXIT INT TERM

echo
echo "Video Harness completed successfully."
echo "Dataset:   ${dataset_root}"
echo "Documents: ${output_root}/documents.jsonl"
echo "Evidence:  ${annotation_output}"
echo "Pairs:     ${output_root}/pairs.jsonl"
echo "Review the evidence report and manually audit a sample before training."
