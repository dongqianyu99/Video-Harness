from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
import pytest

from openpi.models.guide_pi0_config import GuidePi0Config
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as stock_config
from openpi.training.guide_train_config import GuidedTrainRunConfig
from openpi.training.guide_train_config import make_guide_pi0_config
from openpi.training.guide_train_config import resolve_guided_train_config
from openpi.training.guide_weight_loaders import GuidePi0BaseWeightLoader
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig


def _guided_data(tmp_path: Path, *, batch_size: int = 4) -> RoboDojoGuidedDataConfig:
    return RoboDojoGuidedDataConfig(
        repo_id="fake",
        dataset_root=tmp_path / "dataset",
        dataset_artifact_path=tmp_path / "dataset.json",
        documents_artifact_path=tmp_path / "documents.jsonl",
        pairs_artifact_path=tmp_path / "pairs.jsonl",
        batch_size=batch_size,
        seed=7,
        profile="actuator-v0",
        max_frames=4,
        max_units=2,
        max_text_tokens=8,
        query_episode_indices=(11,),
    )


def _run_config(tmp_path: Path, *, batch_size: int = 4) -> GuidedTrainRunConfig:
    return GuidedTrainRunConfig(
        native_config_name="native-pi05",
        base_params_path=tmp_path / "pi05_base" / "params",
        guided_data=_guided_data(tmp_path, batch_size=batch_size),
        experiment_name="guided-test",
        checkpoint_dir=tmp_path / "checkpoints" / "guided-test",
        num_train_steps=3,
        log_interval=1,
        save_interval=2,
        fsdp_devices=1,
    )


def test_make_guide_pi0_config_copies_all_native_model_fields() -> None:
    native_model = Pi0Config(
        pi05=True,
        action_dim=7,
        action_horizon=13,
        max_token_len=91,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )

    guide_model = make_guide_pi0_config(native_model)

    assert isinstance(guide_model, GuidePi0Config)
    for field in dataclasses.fields(native_model):
        assert getattr(guide_model, field.name) == getattr(native_model, field.name)

    assert guide_model.guide_num_queries == 8
    assert guide_model.guide_resampler_width == 1024
    assert guide_model.guide_resampler_num_heads == 8
    assert guide_model.guide_resampler_ffn_hidden_dim is None


def test_resolve_guided_train_config_preserves_native_optimizer_and_uses_strict_loader(tmp_path: Path) -> None:
    native = stock_config.get_config("pi05_wuji_marvin_54d")
    run_config = dataclasses.replace(
        _run_config(tmp_path, batch_size=5),
        native_config_name=native.name,
    )

    resolved = resolve_guided_train_config(run_config)

    assert isinstance(resolved.model, GuidePi0Config)
    assert isinstance(resolved.weight_loader, GuidePi0BaseWeightLoader)
    assert resolved.weight_loader.params_path == str(run_config.base_params_path)
    assert isinstance(resolved.freeze_filter, nnx.Nothing)
    assert resolved.batch_size == run_config.guided_data.batch_size
    assert resolved.num_workers == 0
    assert resolved.optimizer == native.optimizer
    assert resolved.lr_schedule == native.lr_schedule
    assert resolved.ema_decay == native.ema_decay
    assert resolved.checkpoint_dir == run_config.checkpoint_dir.resolve()
    assert resolved.model.action_dim == native.model.action_dim
    assert resolved.model.action_horizon == native.model.action_horizon
    assert resolved.model.max_token_len == native.model.max_token_len
    assert native.batch_size == 64
    assert native.model.__class__ is Pi0Config


def test_native_pi0_config_is_rejected(monkeypatch, tmp_path: Path) -> None:
    native = stock_config.get_config("pi05_wuji_marvin_54d")
    fake_native = dataclasses.replace(native, model=Pi0Config(pi05=False))
    monkeypatch.setattr(
        "openpi.training.guide_train_config._load_native_config",
        lambda _name: fake_native,
    )

    with pytest.raises(ValueError, match="native Pi05"):
        resolve_guided_train_config(_run_config(tmp_path))


@pytest.mark.parametrize(
    ("field_name", "variant"),
    [
        ("paligemma_variant", "gemma_2b_lora"),
        ("action_expert_variant", "gemma_300m_LoRA"),
    ],
)
def test_lora_variants_are_rejected_before_guided_model_construction(
    tmp_path: Path, field_name: str, variant: str
) -> None:
    native_model = Pi0Config(
        pi05=True,
        action_dim=7,
        action_horizon=13,
        max_token_len=91,
        **{field_name: variant},
    )

    with pytest.raises(ValueError, match="LoRA variants"):
        make_guide_pi0_config(native_model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_train_steps": 0},
        {"log_interval": 0},
        {"save_interval": 0},
        {"fsdp_devices": 0},
    ],
)
def test_guided_run_config_rejects_invalid_schedule_values(tmp_path: Path, kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dataclasses.replace(_run_config(tmp_path), **kwargs)


def test_guided_run_config_separates_overwrite_and_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overwrite and resume"):
        dataclasses.replace(_run_config(tmp_path), overwrite=True, resume=True)
