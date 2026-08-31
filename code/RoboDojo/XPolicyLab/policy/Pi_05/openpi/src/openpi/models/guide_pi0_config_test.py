from __future__ import annotations

import sys
import types

import flax.nnx as nnx
import jax
import pytest

from openpi.models import guide_pi0_config
from openpi.models import model as _model
from openpi.models import pi0_config


def test_guide_pi0_config_defaults_to_pi05_and_guide_resampler_settings() -> None:
    config = guide_pi0_config.GuidePi0Config()

    assert config.pi05 is True
    assert config.guide_boundary_num_queries == 8
    assert config.guide_transition_num_queries == 4
    assert config.guide_resampler_width == 1024
    assert config.guide_resampler_num_heads == 8
    assert config.guide_resampler_ffn_hidden_dim is None
    assert config.model_type is _model.ModelType.PI05


def test_guide_pi0_config_rejects_pi0_mode() -> None:
    with pytest.raises(ValueError, match="pi05"):
        guide_pi0_config.GuidePi0Config(pi05=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"guide_boundary_num_queries": 0}, "positive"),
        ({"guide_transition_num_queries": 0}, "positive"),
        ({"guide_resampler_width": 0}, "positive"),
        ({"guide_resampler_num_heads": 0}, "positive"),
        ({"guide_resampler_width": 10, "guide_resampler_num_heads": 3}, "divisible"),
        ({"guide_resampler_ffn_hidden_dim": 0}, "ffn_hidden_dim"),
    ],
)
def test_guide_pi0_config_rejects_invalid_resampler_settings(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        guide_pi0_config.GuidePi0Config(**kwargs)


def test_guide_pi0_config_keeps_native_inputs_spec() -> None:
    config_kwargs = {
        "action_dim": 7,
        "action_horizon": 11,
        "max_token_len": 13,
    }
    guide_config = guide_pi0_config.GuidePi0Config(**config_kwargs)
    native_config = pi0_config.Pi0Config(pi05=True, **config_kwargs)

    guide_observation, guide_actions = guide_config.inputs_spec(batch_size=3)
    native_observation, native_actions = native_config.inputs_spec(batch_size=3)

    assert jax.tree_util.tree_structure(guide_observation) == jax.tree_util.tree_structure(native_observation)
    assert jax.tree_util.tree_structure(guide_actions) == jax.tree_util.tree_structure(native_actions)

    for guide_leaf, native_leaf in zip(
        jax.tree_util.tree_leaves(guide_observation),
        jax.tree_util.tree_leaves(native_observation),
        strict=True,
    ):
        assert guide_leaf.shape == native_leaf.shape
        assert guide_leaf.dtype == native_leaf.dtype

    for guide_leaf, native_leaf in zip(
        jax.tree_util.tree_leaves(guide_actions),
        jax.tree_util.tree_leaves(native_actions),
        strict=True,
    ):
        assert guide_leaf.shape == native_leaf.shape
        assert guide_leaf.dtype == native_leaf.dtype


def test_guide_pi0_config_keeps_dense_full_finetuning_filter() -> None:
    config = guide_pi0_config.GuidePi0Config()

    assert config.get_freeze_filter() is nnx.Nothing


def test_create_constructs_guide_pi0_without_initializing_a_real_model(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeGuidePi0:
        def __init__(self, config, *, rngs):
            calls["config"] = config
            calls["rngs"] = rngs

    fake_guide_pi0 = types.ModuleType("openpi.models.guide_pi0")
    fake_guide_pi0.GuidePi0 = _FakeGuidePi0
    monkeypatch.setitem(sys.modules, "openpi.models.guide_pi0", fake_guide_pi0)

    config = guide_pi0_config.GuidePi0Config()
    result = config.create(jax.random.key(0))

    assert isinstance(result, _FakeGuidePi0)
    assert calls["config"] is config
    assert isinstance(calls["rngs"], nnx.Rngs)
