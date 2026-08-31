from __future__ import annotations

from argparse import Namespace
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi.models.guide_pi0_config import GuidePi0Config
from openpi.shared.normalize import NormStats
from openpi.training import checkpoints
from openpi.training.guide_train_config import GuidedTrainRunConfig
from openpi.training.guide_train_step_test import _make_batch
from openpi.training.guide_train_step_test import _make_state
from openpi.training.guide_train_step_test import _TinyGuidedModel
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig
import openpi.training.sharding as stock_sharding
from scripts import train_guided as _train


def _guided_data(tmp_path: Path) -> RoboDojoGuidedDataConfig:
    return RoboDojoGuidedDataConfig(
        repo_id="fake",
        dataset_root=tmp_path / "dataset",
        documents_root=tmp_path / "documents",
        guides_per_batch=1,
        queries_per_guide=2,
        seed=0,
        max_boundaries=2,
        max_units=1,
        max_boundary_text_tokens=4,
        max_transition_text_tokens=4,
    )


def _run_config(tmp_path: Path) -> GuidedTrainRunConfig:
    return GuidedTrainRunConfig(
        native_config_name="native-pi05",
        base_params_path=tmp_path / "base" / "params",
        guided_data=_guided_data(tmp_path),
        experiment_name="guided",
        checkpoint_dir=tmp_path / "guided-checkpoint",
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        fsdp_devices=1,
    )


def test_argument_entry_builds_task_level_data_config(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(run_config):
        captured["run_config"] = run_config
        return object()

    monkeypatch.setattr(_train, "run_guided_training", fake_run)
    args = Namespace(
        native_config_name="native-pi05",
        base_params_path=tmp_path / "base",
        repo_id="RoboDojo",
        dataset_root=tmp_path / "dataset",
        documents_root=tmp_path / "documents",
        guides_per_batch=2,
        queries_per_guide=3,
        max_boundaries=2,
        max_units=1,
        max_boundary_text_tokens=5,
        max_transition_text_tokens=4,
        guide_boundary_num_queries=12,
        guide_transition_num_queries=8,
        experiment_name="guided",
        checkpoint_dir=tmp_path / "checkpoint",
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        fsdp_devices=1,
        seed=0,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    )

    _train._run_from_args(args)  # noqa: SLF001

    data = captured["run_config"].guided_data
    assert (data.guides_per_batch, data.queries_per_guide) == (2, 3)
    assert (data.guide_boundary_num_queries, data.guide_transition_num_queries) == (12, 8)
    assert data.guide_length_buckets is None
    assert captured["run_config"].base_params_path == tmp_path / "base"

    args.guide_length_bucket = ["1:2"]
    args.remainder_strategy = "pad_mask"
    args.gradient_accumulation_steps = 2
    args.reference_global_batch_size = 4
    args.allow_effective_batch_mismatch = True
    _train._run_from_args(args)  # noqa: SLF001
    formal_data = captured["run_config"].guided_data
    assert formal_data.require_all_tasks
    assert formal_data.guide_length_buckets[0].bucket_id == "u1-b2"
    assert formal_data.remainder_strategy == "pad_mask"
    assert formal_data.gradient_accumulation_steps == 2
    assert captured["run_config"].gradient_accumulation_steps == 2
    assert not captured["run_config"].enforce_reference_batch_size

    args.run_dir = tmp_path / "tracked-run"
    args.checkpoint_dir = None
    _train._run_from_args(args)  # noqa: SLF001
    tracked = captured["run_config"]
    assert tracked.run_dir == args.run_dir
    assert tracked.checkpoint_dir == args.run_dir / "checkpoints"


def test_runtime_validation_requires_full_dense_guided_config(tmp_path: Path) -> None:
    run_config = _run_config(tmp_path)
    valid = SimpleNamespace(
        model=GuidePi0Config(),
        batch_size=2,
        num_workers=0,
        freeze_filter=nnx.Nothing(),
    )
    _train._validate_runtime_config(run_config, valid)  # noqa: SLF001

    with pytest.raises(ValueError, match="full dense"):
        _train._validate_runtime_config(  # noqa: SLF001
            run_config,
            SimpleNamespace(
                model=GuidePi0Config(),
                batch_size=2,
                num_workers=0,
                freeze_filter=nnx.All(nnx.Param),
            ),
        )


def test_runtime_validation_separates_base_and_resume_paths(tmp_path: Path) -> None:
    run_config = _run_config(tmp_path)
    same_path = run_config.checkpoint_dir
    invalid = GuidedTrainRunConfig(
        native_config_name=run_config.native_config_name,
        base_params_path=same_path,
        guided_data=run_config.guided_data,
        experiment_name=run_config.experiment_name,
        checkpoint_dir=same_path,
        num_train_steps=run_config.num_train_steps,
        log_interval=run_config.log_interval,
        save_interval=run_config.save_interval,
        fsdp_devices=run_config.fsdp_devices,
    )

    with pytest.raises(ValueError, match="different"):
        _train._validate_runtime_config(  # noqa: SLF001
            invalid,
            SimpleNamespace(
                model=GuidePi0Config(),
                batch_size=2,
                num_workers=0,
                freeze_filter=nnx.Nothing(),
            ),
        )


def test_runtime_validation_aligns_effective_batch_with_official_global_256(
    tmp_path: Path,
) -> None:
    guided_data = dataclasses.replace(
        _guided_data(tmp_path),
        guides_per_batch=4,
        queries_per_guide=16,
        gradient_accumulation_steps=4,
    )
    run_config = dataclasses.replace(
        _run_config(tmp_path),
        guided_data=guided_data,
        gradient_accumulation_steps=4,
        reference_global_batch_size=256,
        enforce_reference_batch_size=True,
    )
    resolved = SimpleNamespace(
        model=GuidePi0Config(),
        batch_size=64,
        num_workers=0,
        freeze_filter=nnx.Nothing(),
    )

    _train._validate_runtime_config(run_config, resolved)  # noqa: SLF001

    mismatched_data = dataclasses.replace(guided_data, queries_per_guide=8)
    mismatched = dataclasses.replace(run_config, guided_data=mismatched_data)
    with pytest.raises(ValueError, match="effective global batch"):
        _train._validate_runtime_config(  # noqa: SLF001
            mismatched,
            SimpleNamespace(**{**vars(resolved), "batch_size": 32}),
        )

    padded_data = dataclasses.replace(guided_data, remainder_strategy="pad_mask")
    padded = dataclasses.replace(run_config, guided_data=padded_data)
    with pytest.raises(ValueError, match="remainder_strategy='drop'"):
        _train._validate_runtime_config(padded, resolved)  # noqa: SLF001


def test_checkpoint_wrappers_delegate_to_stock_format(monkeypatch, tmp_path: Path) -> None:
    calls = []
    sentinel = object()

    class _FakeLoader:
        def data_config(self):
            return object()

    def save_state(manager, state, loader, step):
        calls.append(("save", manager, state, loader, step))
        return sentinel

    def restore_state(manager, state, loader, step):
        calls.append(("restore", manager, state, loader, step))
        return sentinel

    monkeypatch.setattr(
        _train,
        "_load_checkpoints",
        lambda: SimpleNamespace(save_state=save_state, restore_state=restore_state),
    )
    loader = _FakeLoader()

    assert _train.save_guided_state("manager", "state", loader, 4) is sentinel
    assert _train.restore_guided_state("manager", "state", loader, 4) is sentinel
    assert calls == [
        ("save", "manager", "state", loader, 4),
        ("restore", "manager", "state", loader, 4),
    ]


def test_checkpoint_save_fails_for_fake_loader_without_data_config() -> None:
    with pytest.raises(ValueError, match="data_config"):
        _train.save_guided_state("manager", "state", object(), 0)


def test_stock_checkpoint_roundtrip_keeps_state_and_norm_stats(tmp_path: Path) -> None:
    state = _make_state()
    data_config = SimpleNamespace(
        norm_stats={
            "state": NormStats(
                mean=np.zeros(2, dtype=np.float32),
                std=np.ones(2, dtype=np.float32),
            )
        },
        asset_id="synthetic",
    )
    loader = SimpleNamespace(data_config=lambda: data_config)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoint",
        keep_period=None,
        overwrite=False,
        resume=False,
    )
    assert not resuming

    _train.save_guided_state(manager, state, loader, 1)
    manager.wait_until_finished()
    restored = _train.restore_guided_state(manager, state, loader, 1)

    assert int(restored.step) == int(state.step)
    for path, variable in state.params.flat_state().items():
        np.testing.assert_array_equal(variable.value, restored.params.flat_state()[path].value)
    for path, variable in state.ema_params.flat_state().items():
        np.testing.assert_array_equal(variable.value, restored.ema_params.flat_state()[path].value)
    jax.tree.map(np.testing.assert_array_equal, state.opt_state, restored.opt_state)
    assert (tmp_path / "checkpoint" / "1" / "assets" / "synthetic" / "norm_stats.json").is_file()


def test_one_step_fake_guided_entry_uses_sharding_and_checkpoint_hooks(monkeypatch, tmp_path: Path) -> None:
    run_config = _run_config(tmp_path)
    resolved = SimpleNamespace(
        model=GuidePi0Config(),
        batch_size=2,
        num_workers=0,
        freeze_filter=nnx.Nothing(),
        trainable_filter=nnx.All(nnx.Param),
        checkpoint_dir=run_config.checkpoint_dir,
        keep_period=None,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
        fsdp_devices=1,
        seed=0,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        lr_schedule=SimpleNamespace(create=lambda: lambda _step: 1e-5),
    )
    state = _make_state()
    batch = _make_batch()
    calls = {"saved": 0, "waited": 0}
    wandb_logs = []

    class _WandbRun:
        def log(self, payload, *, step):
            wandb_logs.append((payload, step))

        def finish(self, *, exit_code):
            calls["wandb_exit_code"] = exit_code

    class _Manager:
        def wait_until_finished(self):
            calls["waited"] += 1

    class _Loader:
        def __iter__(self):
            while True:
                yield batch

        def data_config(self):
            return object()

    loader = _Loader()

    fake_train = SimpleNamespace(
        init_wandb=lambda *args, **kwargs: None,
        init_train_state=lambda *args, **kwargs: (state, None),
    )
    fake_data = SimpleNamespace(
        create_robodojo_guided_data_loader=lambda *args, **kwargs: loader,
    )
    fake_checkpoints = SimpleNamespace(
        initialize_checkpoint_dir=lambda *args, **kwargs: (_Manager(), False),
    )

    def fake_import(module_name):
        if module_name == "scripts.train":
            return fake_train
        if module_name == "openpi.training.robodojo_guide_data":
            return fake_data
        if module_name == "openpi.training.sharding":
            return stock_sharding
        return original_import(module_name)

    original_import = _train.importlib.import_module
    monkeypatch.setattr(_train.importlib, "import_module", fake_import)
    monkeypatch.setattr(_train, "resolve_guided_train_config", lambda _run: resolved)
    monkeypatch.setattr(_train, "_load_native_config", lambda _name: object())
    monkeypatch.setattr(_train, "_load_checkpoints", lambda: fake_checkpoints)
    monkeypatch.setattr(_train, "GuidePi0", _TinyGuidedModel)
    monkeypatch.setattr(
        _train,
        "init_guided_wandb",
        lambda **_kwargs: _WandbRun(),
    )
    monkeypatch.setattr(
        _train,
        "save_guided_state",
        lambda *args, **kwargs: calls.__setitem__("saved", calls["saved"] + 1),
    )

    result = _train.run_guided_training(run_config)

    assert int(result.step) == 1
    assert calls == {"saved": 1, "waited": 1, "wandb_exit_code": 0}
    assert wandb_logs[0][1] == 0
    assert "train/loss" in wandb_logs[0][0]
    assert "performance/optimizer_steps_per_s" in wandb_logs[0][0]
    run_root = run_config.checkpoint_dir.parent / f"{run_config.checkpoint_dir.name}.run"
    manifest = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert (run_root / "logs" / "train.log").is_file()
    assert (run_root / "eval").is_dir()


def test_run_failure_is_recorded_in_manifest(monkeypatch, tmp_path: Path) -> None:
    run_config = _run_config(tmp_path)

    def fail(_run_config, _layout, _session):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(_train, "_run_guided_training_impl", fail)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        _train.run_guided_training(run_config)

    run_root = run_config.checkpoint_dir.parent / f"{run_config.checkpoint_dir.name}.run"
    manifest = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "synthetic failure",
    }
