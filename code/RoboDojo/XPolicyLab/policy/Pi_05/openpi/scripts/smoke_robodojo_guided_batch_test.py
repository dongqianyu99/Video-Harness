from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import smoke_robodojo_guided_batch as _smoke


class _StopAfterConfigError(RuntimeError):
    pass


def test_smoke_cli_scopes_loader_to_requested_query_episode(monkeypatch):
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
        dataset_artifact=Path("dataset.json"),
        documents_artifact=Path("documents.jsonl"),
        pairs_artifact=Path("pairs.jsonl"),
        query_episode_index=37,
        batch_size=2,
        max_frames=8,
        max_units=4,
        max_text_tokens=32,
        seed=0,
        profile="actuator",
        num_batches=1,
    )

    with pytest.raises(_StopAfterConfigError):
        _smoke._run(args)  # noqa: SLF001

    assert captured["guided_config"].query_episode_indices == (37,)
    assert captured["num_batches"] == 1
