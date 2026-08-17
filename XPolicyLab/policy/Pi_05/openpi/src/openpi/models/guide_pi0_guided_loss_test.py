from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import guide_attention
from openpi.models import guide_encoder
from openpi.models import guide_pi0
from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.models.pi0 import Pi0

_GROUPS = 2
_QUERIES = 2
_BATCH = _GROUPS * _QUERIES
_ACTION_HORIZON = 3
_ACTION_DIM = 2
_PREFIX_WIDTH = 4
_GUIDE_TOKENS = 2
_CONTROL_TOKENS = 3


def _make_observation() -> _model.Observation:
    images = {
        key: jnp.zeros((_GROUPS, _QUERIES, 1, 1, 3), dtype=jnp.float32)
        for key in _model.IMAGE_KEYS
    }
    image_masks = {
        key: jnp.ones((_GROUPS, _QUERIES), dtype=jnp.bool_)
        for key in _model.IMAGE_KEYS
    }
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((_GROUPS, _QUERIES, _ACTION_DIM), dtype=jnp.float32),
        tokenized_prompt=jnp.zeros((_GROUPS, _QUERIES, 4), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((_GROUPS, _QUERIES, 4), dtype=jnp.bool_),
    )


def _make_guide() -> GuideInput:
    return GuideInput(
        images=jnp.zeros((_GROUPS, 2, 1, 1, 3), dtype=jnp.float32),
        image_mask=jnp.ones((_GROUPS, 2), dtype=jnp.bool_),
        text_tokens=jnp.zeros((_GROUPS, 1, 2), dtype=jnp.int32),
        text_mask=jnp.ones((_GROUPS, 1, 2), dtype=jnp.bool_),
        unit_mask=jnp.ones((_GROUPS, 1), dtype=jnp.bool_),
        before_slot=jnp.zeros((_GROUPS, 1), dtype=jnp.int32),
        after_slot=jnp.ones((_GROUPS, 1), dtype=jnp.int32),
    )


def _make_batch() -> GuideConditionedBatch:
    return GuideConditionedBatch(
        observation=_make_observation(),
        actions=jnp.ones(
            (_GROUPS, _QUERIES, _ACTION_HORIZON, _ACTION_DIM),
            dtype=jnp.float32,
        ),
        guide=_make_guide(),
    )


class _RecordingPrefix:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.tokens = jnp.arange(
            _BATCH * _CONTROL_TOKENS * _PREFIX_WIDTH,
            dtype=jnp.float32,
        ).reshape(_BATCH, _CONTROL_TOKENS, _PREFIX_WIDTH)
        self.mask = jnp.array(
            [[True, True, False]] * _BATCH,
            dtype=jnp.bool_,
        )

    def __call__(self, observation):
        self.calls.append(observation)
        return self.tokens, self.mask, jnp.zeros((_CONTROL_TOKENS,), dtype=jnp.bool_)


class _RecordingSuffix:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[int, ...], tuple[int, ...]]] = []
        self.tokens = jnp.arange(
            _BATCH * _ACTION_HORIZON * _PREFIX_WIDTH,
            dtype=jnp.float32,
        ).reshape(_BATCH, _ACTION_HORIZON, _PREFIX_WIDTH)
        self.mask = jnp.array(
            [[True, True, False]] * _BATCH,
            dtype=jnp.bool_,
        )
        self.ar_mask = jnp.array([True, False, False], dtype=jnp.bool_)

    def __call__(self, observation, noisy_actions, timestep):
        self.calls.append((observation, noisy_actions.shape, timestep.shape))
        return self.tokens, self.mask, self.ar_mask, None


class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, embedded, *, mask, positions, adarms_cond):
        self.calls.append(
            {
                "embedded": embedded,
                "mask": mask,
                "positions": positions,
                "adarms_cond": adarms_cond,
            }
        )
        return (None, embedded[1]), None


class _RecordingActionOutProjection:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, values):
        self.calls.append(values.shape)
        return jnp.zeros(
            (values.shape[0], values.shape[1], _ACTION_DIM),
            dtype=values.dtype,
        )


def _make_fake_model():
    guide_memory = guide_encoder.GuideMemory(
        tokens=jnp.arange(
            _GROUPS * _GUIDE_TOKENS * _PREFIX_WIDTH,
            dtype=jnp.float32,
        ).reshape(_GROUPS, _GUIDE_TOKENS, _PREFIX_WIDTH),
        token_mask=jnp.array(
            [[True, False], [True, True]],
            dtype=jnp.bool_,
        ),
    )
    guide_calls: list[object] = []
    prefix = _RecordingPrefix()
    suffix = _RecordingSuffix()
    llm = _RecordingLLM()
    action_out_proj = _RecordingActionOutProjection()

    def encode_guide(guide):
        guide_calls.append(guide)
        return guide_memory

    model = types.SimpleNamespace(
        action_horizon=_ACTION_HORIZON,
        action_dim=_ACTION_DIM,
        PaliGemma=types.SimpleNamespace(llm=llm),
        action_out_proj=action_out_proj,
        embed_prefix=prefix,
        embed_suffix=suffix,
        encode_guide=encode_guide,
    )
    prefix_helper = guide_pi0.GuidePi0._embed_guide_control_prefix  # noqa: SLF001
    model._embed_guide_control_prefix = types.MethodType(prefix_helper, model)  # noqa: SLF001
    prepare_helper = guide_pi0.GuidePi0._prepare_guided_loss_inputs  # noqa: SLF001
    model._prepare_guided_loss_inputs = types.MethodType(prepare_helper, model)  # noqa: SLF001
    forward_helper = guide_pi0.GuidePi0._run_guided_joint_forward  # noqa: SLF001
    model._run_guided_joint_forward = types.MethodType(forward_helper, model)  # noqa: SLF001
    return model, guide_memory, guide_calls, prefix, suffix, llm, action_out_proj


def test_compute_guided_loss_uses_two_experts_and_native_action_loss(monkeypatch) -> None:
    model, guide_memory, guide_calls, prefix, suffix, llm, action_out_proj = _make_fake_model()
    preprocess_calls: list[tuple[object, object, bool]] = []

    def fake_preprocess(rng, observation, *, train):
        preprocess_calls.append((rng, observation, train))
        return observation

    monkeypatch.setattr(_model, "preprocess_observation", fake_preprocess)

    loss = guide_pi0.GuidePi0.compute_guided_loss(
        model,
        jax.random.key(0),
        _make_batch(),
        train=True,
    )

    assert loss.shape == (_BATCH, _ACTION_HORIZON)
    assert len(preprocess_calls) == 1
    assert preprocess_calls[0][2] is True
    assert len(guide_calls) == 1
    assert len(prefix.calls) == 1
    assert len(suffix.calls) == 1
    assert len(llm.calls) == 1
    assert len(action_out_proj.calls) == 1

    llm_call = llm.calls[0]
    embedded = llm_call["embedded"]
    assert len(embedded) == 2
    assert embedded[0].shape == (_BATCH, _GUIDE_TOKENS + _CONTROL_TOKENS, _PREFIX_WIDTH)
    assert embedded[1].shape == (_BATCH, _ACTION_HORIZON, _PREFIX_WIDTH)

    attention_mask = llm_call["mask"]
    positions = llm_call["positions"]
    assert attention_mask.shape == (
        _BATCH,
        _GUIDE_TOKENS + _CONTROL_TOKENS + _ACTION_HORIZON,
        _GUIDE_TOKENS + _CONTROL_TOKENS + _ACTION_HORIZON,
    )

    guide_mask = jnp.repeat(guide_memory.token_mask, _QUERIES, axis=0)
    expected_mask = guide_attention.make_gca_attn_mask(
        guide_mask,
        prefix.mask,
        suffix.mask,
    )
    np.testing.assert_array_equal(attention_mask, expected_mask)

    input_mask = jnp.concatenate([guide_mask, prefix.mask, suffix.mask], axis=1)
    expected_positions = jnp.cumsum(input_mask, axis=1) - 1
    np.testing.assert_array_equal(positions, expected_positions)


def test_guided_pi0_keeps_stock_forward_methods() -> None:
    assert guide_pi0.GuidePi0.compute_loss is Pi0.compute_loss
    assert guide_pi0.GuidePi0.sample_actions is Pi0.sample_actions
