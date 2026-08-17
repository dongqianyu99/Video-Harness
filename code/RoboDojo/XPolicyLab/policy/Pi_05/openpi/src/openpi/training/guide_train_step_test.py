from __future__ import annotations

import copy
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.training.guide_train_step import guided_train_step
from openpi.training.utils import TrainState

_CALL_COUNTER = {"count": 0}


class _TinyGuidedModel(_model.BaseModel):
    def __init__(self) -> None:
        super().__init__(action_dim=2, action_horizon=3, max_token_len=8)
        self.native_backbone = nnx.Param(jnp.asarray(0.5, dtype=jnp.float32))
        self.guide_encoder = nnx.Param(jnp.asarray(0.25, dtype=jnp.float32))

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        del rng, observation, train
        return jnp.square(actions * self.native_backbone)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return jnp.zeros((observation.state.shape[0], self.action_horizon, self.action_dim))

    def compute_guided_loss(self, rng, batch: GuideConditionedBatch, *, train: bool = False):
        del rng, train
        _CALL_COUNTER["count"] += 1
        guide_signal = jnp.mean(batch.guide.images) + jnp.mean(batch.guide.text_tokens)
        prediction = batch.actions * self.native_backbone + guide_signal * self.guide_encoder
        return jnp.square(prediction)


@dataclasses.dataclass(frozen=True)
class _TinyConfig:
    trainable_filter: object = dataclasses.field(default_factory=lambda: nnx.All(nnx.Param))


def _make_batch() -> GuideConditionedBatch:
    images = {
        key: jnp.ones((1, 2, 4, 4, 3), dtype=jnp.float32)
        for key in _model.IMAGE_KEYS
    }
    image_masks = {
        key: jnp.ones((1, 2), dtype=jnp.bool_)
        for key in _model.IMAGE_KEYS
    }
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.ones((1, 2, 3), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((1, 2, 4), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 2, 4), dtype=jnp.bool_),
    )
    guide = GuideInput(
        images=jnp.ones((1, 2, 4, 4, 3), dtype=jnp.float32),
        image_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        text_tokens=jnp.ones((1, 1, 4), dtype=jnp.int32),
        text_mask=jnp.ones((1, 1, 4), dtype=jnp.bool_),
        unit_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        before_slot=jnp.zeros((1, 1), dtype=jnp.int32),
        after_slot=jnp.ones((1, 1), dtype=jnp.int32),
    )
    return GuideConditionedBatch(
        observation=observation,
        actions=jnp.ones((1, 2, 3, 2), dtype=jnp.float32),
        guide=guide,
    )


def _make_state() -> TrainState:
    model = _TinyGuidedModel()
    tx = optax.sgd(learning_rate=0.1, momentum=0.9)
    params = nnx.state(model)
    return TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params.filter(nnx.All(nnx.Param))),
        tx=tx,
        ema_decay=0.5,
        ema_params=params,
    )


def _state_values(state) -> dict[tuple[object, ...], np.ndarray]:
    return {
        path: np.asarray(variable.value).copy()
        for path, variable in state.flat_state().items()
    }


def test_guided_train_step_calls_guided_loss_updates_all_params_optimizer_step_and_ema() -> None:
    _CALL_COUNTER["count"] = 0
    state = _make_state()
    old_params = _state_values(state.params)
    old_ema = _state_values(state.ema_params)
    old_opt_state = jax.tree.map(lambda leaf: np.asarray(leaf).copy(), state.opt_state)

    new_state, info = guided_train_step(
        _TinyConfig(),
        jax.random.key(3),
        state,
        _make_batch(),
    )

    assert _CALL_COUNTER["count"] > 0
    assert int(new_state.step) == 1
    assert np.isfinite(float(info["loss"]))
    assert np.isfinite(float(info["grad_norm"]))
    assert np.isfinite(float(info["guide_encoder_grad_norm"]))
    assert np.isfinite(float(info["native_backbone_grad_norm"]))
    assert int(info["G"]) == 1
    assert int(info["Q"]) == 2

    new_params = _state_values(new_state.params)
    assert not np.array_equal(new_params[("native_backbone",)], old_params[("native_backbone",)])
    assert not np.array_equal(new_params[("guide_encoder",)], old_params[("guide_encoder",)])
    assert not jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, old_opt_state, new_state.opt_state))

    expected_ema = {
        path: 0.5 * old_ema[path] + 0.5 * new_params[path]
        for path in old_ema
    }
    np.testing.assert_allclose(_state_values(new_state.ema_params)[("native_backbone",)], expected_ema[("native_backbone",)])
    np.testing.assert_allclose(_state_values(new_state.ema_params)[("guide_encoder",)], expected_ema[("guide_encoder",)])

    np.testing.assert_equal(_state_values(state.params)[("native_backbone",)], old_params[("native_backbone",)])
    np.testing.assert_equal(_state_values(state.ema_params)[("guide_encoder",)], old_ema[("guide_encoder",)])


def test_guided_train_step_is_reproducible_for_fixed_seed() -> None:
    first_state, first_info = guided_train_step(_TinyConfig(), jax.random.key(17), _make_state(), _make_batch())
    second_state, second_info = guided_train_step(_TinyConfig(), jax.random.key(17), _make_state(), _make_batch())

    for path, value in _state_values(first_state.params).items():
        np.testing.assert_array_equal(value, _state_values(second_state.params)[path])
    np.testing.assert_array_equal(first_info["loss"], second_info["loss"])


def test_invalid_batch_fails_before_state_update() -> None:
    state = _make_state()
    invalid_batch = dataclasses.replace(_make_batch(), actions=jnp.ones((2, 3, 2), dtype=jnp.float32))
    before = _state_values(state.params)

    with pytest.raises(ValueError, match="actions must have shape"):
        guided_train_step(_TinyConfig(), jax.random.key(0), state, invalid_batch)

    for path, value in before.items():
        np.testing.assert_array_equal(value, _state_values(state.params)[path])


def test_batch_is_not_mutated() -> None:
    batch = _make_batch()
    before = copy.deepcopy(jax.tree.map(np.asarray, batch))

    guided_train_step(_TinyConfig(), jax.random.key(4), _make_state(), batch)

    def _assert_equal(old, new):
        np.testing.assert_array_equal(old, new)

    jax.tree.map(_assert_equal, before, batch)
