from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def test_interactive_runner_has_side_effect_free_help() -> None:
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_interactive.sh"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Interactive end-to-end Video Harness runner" in result.stdout
    assert "official `hf` CLI" in result.stdout


def test_interactive_runner_refuses_noninteractive_execution() -> None:
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_interactive.sh")],
        input="",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires an interactive terminal" in result.stderr


def test_interactive_runner_keeps_api_tokens_out_of_arguments() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_interactive.sh").read_text()
    assert 'read -r -s -p "OpenAI API token' in script
    assert 'read -r -s -p "Anthropic API token' in script
    assert "--token" not in script
    assert "cleanup_secrets" in script
    assert "source profile" not in script
    assert "official-pi05" not in script
    assert "--profile" not in script


def test_interactive_runner_keeps_outputs_local_without_upload_credentials() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_interactive.sh").read_text()
    assert "hf " + "upload" not in script
    assert "HF_" + "TOKEN" not in script
    assert "guidance_" + "documents" not in script
    assert "Uploaded:" not in script
    assert "Generated artifacts remain" in script


def test_interactive_runner_exposes_debug_and_always_routes_source_media() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_interactive.sh").read_text()
    assert "Enable debug mode" in script
    assert '--debug-root "${output_root}/debug"' in script
    assert '--dataset-root "${dataset_root}"' in script
    assert "--provider mock" not in script


def test_all_guidance_build_entrypoints_exclude_dlc() -> None:
    interactive = (PROJECT_ROOT / "scripts" / "run_interactive.sh").read_text()
    smoke = (PROJECT_ROOT / "scripts" / "smoke_test.sh").read_text()
    cli = (PROJECT_ROOT / "src" / "video_harness" / "cli.py").read_text()
    robodojo = (PROJECT_ROOT / "src" / "video_harness" / "robodojo.py").read_text()
    assert "official-pi05" not in interactive + smoke + cli
    assert "--profile" not in interactive + smoke + cli
    assert 'if task_kind == "dlc":' in robodojo


def test_download_script_uses_locked_hf_cli_without_mirror() -> None:
    script = (WORKSPACE_ROOT / "scripts" / "download_robodojo.sh").read_text()
    assert "hf download RoboDojo-Benchmark/RoboDojo" in script
    assert "--type dataset" in script
    assert "e2f40904e7b039b46e512e1443ad5055984d3344" in script
    assert "hfd" not in script
    assert "HF_ENDPOINT" not in script
