from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import smoke_robodojo_guided_batch as _smoke


class _StopAfterConfigError(RuntimeError):
    pass


def test_smoke_cli_builds_task_level_guidance_config(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        _smoke.importlib,
        "import_module",
        lambda _name: SimpleNamespace(get_config=lambda _config_name: object()),
    )

    def capture_loader(_native_config, guided_config, *, num_batches):
        captured["guided_config"] = guided_config
        captured["num_batches"] = num_batches
        raise _StopAfterConfigError

    monkeypatch.setattr(
        _smoke,
        "create_robodojo_guided_data_loader",
        capture_loader,
    )

    args = Namespace(
        native_config_name="native-pi05",
        repo_id="RoboDojo_lerobot_v30_video",
        dataset_root=Path("dataset"),
        documents_root=Path("documents"),
        guides_per_batch=2,
        queries_per_guide=3,
        max_boundaries=8,
        max_units=4,
        max_boundary_text_tokens=32,
        max_transition_text_tokens=24,
        guide_boundary_num_queries=12,
        guide_transition_num_queries=8,
        seed=0,
        num_batches=1,
    )

    with pytest.raises(_StopAfterConfigError):
        _smoke._run(args)  # noqa: SLF001

    config = captured["guided_config"]
    assert (config.guides_per_batch, config.queries_per_guide) == (2, 3)
    assert (config.guide_boundary_num_queries, config.guide_transition_num_queries) == (12, 8)
    assert captured["num_batches"] == 1
