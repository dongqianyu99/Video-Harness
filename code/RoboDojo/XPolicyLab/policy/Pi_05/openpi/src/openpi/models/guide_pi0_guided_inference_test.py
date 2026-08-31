from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import guide_attention
from openpi.models import guide_encoder
from openpi.models import guide_inputs
from openpi.models import guide_pi0
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models.guide_inputs import GuideInput
from openpi.shared import array_typing as at

_GROUPS = 2
_QUERIES = 3
_GUIDE_TOKENS = 2
_CONTROL_TOKENS = 3
_PREFIX_WIDTH = 4
_ACTION_HORIZON = 2
_ACTION_DIM = 2


def _make_observation(*, include_prompt: bool = True) -> _model.Observation:
    images = {
        key: jnp.arange(
            _GROUPS * _QUERIES * 2 * 2 * 3,
            dtype=jnp.float32,
        ).reshape(_GROUPS, _QUERIES, 2, 2, 3)
        for key in _model.IMAGE_KEYS
    }
    image_masks = {key: jnp.ones((_GROUPS, _QUERIES), dtype=jnp.bool_) for key in _model.IMAGE_KEYS}

    if include_prompt:
        tokenized_prompt = jnp.arange(
            _GROUPS * _QUERIES * 4,
            dtype=jnp.int32,
        ).reshape(_GROUPS, _QUERIES, 4)
        tokenized_prompt_mask = jnp.ones(
            (_GROUPS, _QUERIES, 4),
            dtype=jnp.bool_,
        )
    else:
        tokenized_prompt = None
        tokenized_prompt_mask = None

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.arange(
            _GROUPS * _QUERIES * 4,
            dtype=jnp.float32,
        ).reshape(_GROUPS, _QUERIES, 4),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def _make_guide() -> GuideInput:
    return GuideInput(
        boundary_images=jnp.zeros((_GROUPS, 2, 3, 2, 2, 3), dtype=jnp.float32),
        boundary_image_mask=jnp.ones((_GROUPS, 2, 3), dtype=jnp.bool_),
        boundary_text_tokens=jnp.zeros((_GROUPS, 2, 3, 4), dtype=jnp.int32),
        boundary_text_mask=jnp.ones((_GROUPS, 2, 3, 4), dtype=jnp.bool_),
        transition_text_tokens=jnp.zeros((_GROUPS, 1, 4), dtype=jnp.int32),
        transition_text_mask=jnp.ones((_GROUPS, 1, 4), dtype=jnp.bool_),
        boundary_mask=jnp.ones((_GROUPS, 2), dtype=jnp.bool_),
        unit_mask=jnp.ones((_GROUPS, 1), dtype=jnp.bool_),
        memory_source_kind=jnp.zeros((_GROUPS, 3), dtype=jnp.int32),
        memory_source_index=jnp.zeros((_GROUPS, 3), dtype=jnp.int32),
        memory_source_offset=jnp.zeros((_GROUPS, 3), dtype=jnp.int32),
        memory_mask=jnp.ones((_GROUPS, 3), dtype=jnp.bool_),
    )


def test_validate_grouped_observation_returns_g_and_q() -> None:
    assert guide_inputs.validate_guide_conditioned_observation(
        _make_observation(),
        _make_guide(),
    ) == (_GROUPS, _QUERIES)


def test_validate_grouped_observation_allows_optional_none_fields() -> None:
    assert guide_inputs.validate_guide_conditioned_observation(
        _make_observation(include_prompt=False),
        _make_guide(),
    ) == (_GROUPS, _QUERIES)


def test_validate_grouped_observation_rejects_state_without_g_and_q() -> None:
    observation = _make_observation()
    with at.disable_typechecking():
        observation = dataclasses.replace(
            observation,
            state=jnp.zeros((_GROUPS,), dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match=r"state|G|Q"):
        guide_inputs.validate_guide_conditioned_observation(observation, _make_guide())


def test_validate_grouped_observation_rejects_observation_group_query_mismatch() -> None:
    observation = _make_observation()
    with at.disable_typechecking():
        observation = dataclasses.replace(
            observation,
            state=jnp.zeros((_GROUPS, _QUERIES - 1, 4), dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match=r"observation|state"):
        guide_inputs.validate_guide_conditioned_observation(observation, _make_guide())


def test_validate_grouped_observation_rejects_guide_group_mismatch() -> None:
    guide = dataclasses.replace(
        _make_guide(),
        boundary_images=_make_guide().boundary_images[:1],
    )

    with pytest.raises(ValueError, match=r"guide|images"):
        guide_inputs.validate_guide_conditioned_observation(_make_observation(), guide)


def test_flatten_grouped_observation_is_group_major() -> None:
    observation = _make_observation()

    flat_observation = guide_inputs.flatten_grouped_observation(observation)

    assert flat_observation.state.shape == (_GROUPS * _QUERIES, 4)
    np.testing.assert_array_equal(
        flat_observation.state,
        observation.state.reshape(_GROUPS * _QUERIES, 4),
    )
    np.testing.assert_array_equal(
        flat_observation.images["base_0_rgb"],
        observation.images["base_0_rgb"].reshape(_GROUPS * _QUERIES, 2, 2, 3),
    )


def test_flatten_grouped_observation_preserves_optional_none_fields() -> None:
    flat_observation = guide_inputs.flatten_grouped_observation(_make_observation(include_prompt=False))

    assert flat_observation.tokenized_prompt is None
    assert flat_observation.tokenized_prompt_mask is None


def test_flatten_grouped_control_reuses_observation_flatten_helper(monkeypatch) -> None:
    observation = _make_observation()
    actions = jnp.zeros((_GROUPS, _QUERIES, 5, 2), dtype=jnp.float32)
    sentinel = object()
    calls = []

    def fake_flatten_grouped_observation(value):
        calls.append(value)
        return sentinel

    monkeypatch.setattr(
        guide_inputs,
        "flatten_grouped_observation",
        fake_flatten_grouped_observation,
    )

    flat_observation, flat_actions = guide_inputs.flatten_grouped_control(
        observation,
        actions,
    )

    assert calls == [observation]
    assert flat_observation is sentinel
    assert flat_actions.shape == (_GROUPS * _QUERIES, 5, 2)


def test_gc_ar_mask_has_guide_and_control_boundaries() -> None:
    actual = guide_attention.make_gc_ar_mask(
        guide_tokens=3,
        control_tokens=2,
    )
    expected = jnp.array(
        [False, False, False, True, False],
        dtype=jnp.bool_,
    )

    np.testing.assert_array_equal(actual, expected)


def test_gc_attention_mask_has_asymmetric_visibility_and_padding() -> None:
    guide_mask = jnp.array([[True, False, True]], dtype=jnp.bool_)
    control_mask = jnp.array([[True, True, False]], dtype=jnp.bool_)

    actual = guide_attention.make_gc_attn_mask(guide_mask, control_mask)

    assert actual.shape == (1, 6, 6)
    np.testing.assert_array_equal(
        actual[0, 0],
        jnp.array([True, False, True, False, False, False]),
    )
    np.testing.assert_array_equal(
        actual[0, 2],
        jnp.array([True, False, True, False, False, False]),
    )
    np.testing.assert_array_equal(
        actual[0, 3],
        jnp.array([True, False, True, True, True, False]),
    )

    for index in (1, 5):
        assert not np.any(actual[:, index, :])
        assert not np.any(actual[:, :, index])


def test_gc_prefix_matches_gca_prefix_submatrix() -> None:
    guide_mask = jnp.array([[True, False, True]], dtype=jnp.bool_)
    control_mask = jnp.array([[True, True, False]], dtype=jnp.bool_)
    action_mask = jnp.array([[True, True, False]], dtype=jnp.bool_)

    gc_mask = guide_attention.make_gc_attn_mask(guide_mask, control_mask)
    gca_mask = guide_attention.make_gca_attn_mask(
        guide_mask,
        control_mask,
        action_mask,
    )

    prefix_tokens = guide_mask.shape[1] + control_mask.shape[1]
    np.testing.assert_array_equal(
        gc_mask,
        gca_mask[:, :prefix_tokens, :prefix_tokens],
    )


class _RecordingGuideEncoder:
    def __init__(self) -> None:
        self.calls: list[GuideInput] = []
        self.memory = guide_encoder.GuideMemory(
            tokens=jnp.arange(
                _GROUPS * _GUIDE_TOKENS * _PREFIX_WIDTH,
                dtype=jnp.float32,
            ).reshape(_GROUPS, _GUIDE_TOKENS, _PREFIX_WIDTH),
            token_mask=jnp.array(
                [[True, False], [True, True]],
                dtype=jnp.bool_,
            ),
        )

    def __call__(self, guide: GuideInput) -> guide_encoder.GuideMemory:
        self.calls.append(guide)
        return self.memory


class _RecordingPrefillPrefix:
    def __init__(self) -> None:
        batch_size = _GROUPS * _QUERIES
        self.calls: list[object] = []
        self.tokens = jnp.arange(
            batch_size * _CONTROL_TOKENS * _PREFIX_WIDTH,
            dtype=jnp.float32,
        ).reshape(batch_size, _CONTROL_TOKENS, _PREFIX_WIDTH)
        self.mask = jnp.array(
            [[True, True, False]] * batch_size,
            dtype=jnp.bool_,
        )

    def __call__(self, observation):
        self.calls.append(observation)
        return self.tokens, self.mask, jnp.zeros((_CONTROL_TOKENS,), dtype=jnp.bool_)


class _RecordingPrefillLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []
        self.kv_cache = object()

    def __call__(self, embedded, *, mask, positions):
        self.calls.append((embedded, mask, positions))
        return (None, None), self.kv_cache


def test_prefill_guided_prefix_encodes_and_prefills_once(monkeypatch) -> None:
    observation = _make_observation()
    guide = _make_guide()
    guide_encoder_stub = _RecordingGuideEncoder()
    prefix = _RecordingPrefillPrefix()
    llm = _RecordingPrefillLLM()
    preprocess_calls: list[tuple[object, object, bool]] = []

    def fake_preprocess(rng, value, *, train):
        preprocess_calls.append((rng, value, train))
        return value

    monkeypatch.setattr(_model, "preprocess_observation", fake_preprocess)

    def encode_guide(value):
        return guide_encoder_stub(value)

    model = type("FakeGuidePi0", (), {})()
    model.PaliGemma = type("FakePaliGemma", (), {"llm": llm})()
    model.encode_guide = encode_guide
    model.embed_prefix = prefix
    model._embed_guide_control_prefix = (  # noqa: SLF001
        guide_pi0.GuidePi0._embed_guide_control_prefix.__get__(model)  # noqa: SLF001
    )
    model._prefill_guided_prefix_with_memory = (  # noqa: SLF001
        guide_pi0.GuidePi0._prefill_guided_prefix_with_memory.__get__(model)  # noqa: SLF001
    )
    model._validate_guide_memory_observation = (  # noqa: SLF001
        guide_pi0.GuidePi0._validate_guide_memory_observation  # noqa: SLF001
    )

    flat_observation, prefix_mask, kv_cache, groups, queries = guide_pi0.GuidePi0._prefill_guided_prefix(  # noqa: SLF001
        model,
        observation,
        guide,
    )

    assert (groups, queries) == (_GROUPS, _QUERIES)
    assert len(guide_encoder_stub.calls) == 1
    assert guide_encoder_stub.calls[0] is guide
    assert len(preprocess_calls) == 1
    assert preprocess_calls[0][0] is None
    assert preprocess_calls[0][1].state.shape == (_GROUPS * _QUERIES, 4)
    assert preprocess_calls[0][2] is False
    assert len(prefix.calls) == 1
    assert prefix.calls[0] is flat_observation
    assert len(llm.calls) == 1
    assert kv_cache is llm.kv_cache

    embedded, attention_mask, positions = llm.calls[0]
    expected_guide_tokens = jnp.repeat(
        guide_encoder_stub.memory.tokens,
        _QUERIES,
        axis=0,
    )
    expected_guide_mask = jnp.repeat(
        guide_encoder_stub.memory.token_mask,
        _QUERIES,
        axis=0,
    )
    expected_prefix_tokens = jnp.concatenate(
        [expected_guide_tokens, prefix.tokens],
        axis=1,
    )
    expected_prefix_mask = jnp.concatenate(
        [expected_guide_mask, prefix.mask],
        axis=1,
    )

    assert len(embedded) == 2
    np.testing.assert_array_equal(embedded[0], expected_prefix_tokens)
    assert embedded[1] is None
    np.testing.assert_array_equal(prefix_mask, expected_prefix_mask)
    np.testing.assert_array_equal(
        attention_mask,
        guide_attention.make_gc_attn_mask(expected_guide_mask, prefix.mask),
    )
    np.testing.assert_array_equal(
        positions,
        jnp.cumsum(expected_prefix_mask, axis=1) - 1,
    )


class _RecordingSamplingSuffix:
    def __init__(self) -> None:
        batch_size = _GROUPS * _QUERIES
        self.calls: list[tuple[object, object, object]] = []
        self.tokens = jnp.arange(
            batch_size * _ACTION_HORIZON * _PREFIX_WIDTH,
            dtype=jnp.float32,
        ).reshape(batch_size, _ACTION_HORIZON, _PREFIX_WIDTH)
        self.mask = jnp.ones(
            (batch_size, _ACTION_HORIZON),
            dtype=jnp.bool_,
        )
        self.ar_mask = jnp.array(
            [True, False],
            dtype=jnp.bool_,
        )
        self.adarms_cond = jnp.ones(
            (batch_size, _PREFIX_WIDTH),
            dtype=jnp.float32,
        )

    def __call__(self, observation, noisy_actions, timestep):
        self.calls.append((observation, noisy_actions, timestep))
        return self.tokens, self.mask, self.ar_mask, self.adarms_cond


class _RecordingSamplingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.runtime_calls: list[dict[str, object]] = []
        self.kv_cache = object()

    def __call__(
        self,
        embedded,
        *,
        mask,
        positions,
        kv_cache=None,
        adarms_cond=None,
    ):
        if embedded[1] is None:
            self.calls.append(
                {
                    "embedded": embedded,
                    "mask": mask,
                    "positions": positions,
                    "kv_cache": kv_cache,
                    "adarms_cond": adarms_cond,
                }
            )
            return (None, None), self.kv_cache

        self.calls.append(
            {
                "embedded": embedded,
                "kv_cache": kv_cache,
                "adarms_cond": adarms_cond,
            }
        )

        def record_runtime(mask_value, positions_value):
            self.runtime_calls.append(
                {
                    "mask": mask_value,
                    "positions": positions_value,
                }
            )

        jax.debug.callback(record_runtime, mask, positions)
        return (None, embedded[1]), self.kv_cache


class _RecordingActionOutProjection:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, values):
        self.calls.append(values)
        return jnp.zeros(
            (values.shape[0], values.shape[1], _ACTION_DIM),
            dtype=values.dtype,
        )


def _make_sampling_model(monkeypatch):
    guide_encoder_stub = _RecordingGuideEncoder()
    prefix = _RecordingPrefillPrefix()
    suffix = _RecordingSamplingSuffix()
    llm = _RecordingSamplingLLM()
    action_out_proj = _RecordingActionOutProjection()

    def fake_preprocess(rng, value, *, train):
        del rng, train
        return value

    monkeypatch.setattr(_model, "preprocess_observation", fake_preprocess)

    def encode_guide(value):
        return guide_encoder_stub(value)

    model = type("FakeGuidePi0", (), {})()
    model.action_horizon = _ACTION_HORIZON
    model.action_dim = _ACTION_DIM
    model.PaliGemma = type("FakePaliGemma", (), {"llm": llm})()
    model.encode_guide = encode_guide
    model.embed_prefix = prefix
    model.embed_suffix = suffix
    model.action_out_proj = action_out_proj
    model._embed_guide_control_prefix = (  # noqa: SLF001
        guide_pi0.GuidePi0._embed_guide_control_prefix.__get__(model)  # noqa: SLF001
    )
    model._prefill_guided_prefix = (  # noqa: SLF001
        guide_pi0.GuidePi0._prefill_guided_prefix.__get__(model)  # noqa: SLF001
    )
    model._prefill_guided_prefix_with_memory = (  # noqa: SLF001
        guide_pi0.GuidePi0._prefill_guided_prefix_with_memory.__get__(model)  # noqa: SLF001
    )
    model._validate_guide_memory_observation = (  # noqa: SLF001
        guide_pi0.GuidePi0._validate_guide_memory_observation  # noqa: SLF001
    )
    model.sample_guided_actions_with_memory = guide_pi0.GuidePi0.sample_guided_actions_with_memory.__get__(model)
    return model, guide_encoder_stub, prefix, suffix, llm, action_out_proj


def test_sample_guided_actions_prefills_once_and_reuses_kv_cache(monkeypatch) -> None:
    model, guide_encoder_stub, prefix, suffix, llm, action_out_proj = _make_sampling_model(monkeypatch)
    observation = _make_observation()
    guide = _make_guide()
    grouped_noise = jnp.arange(
        _GROUPS * _QUERIES * _ACTION_HORIZON * _ACTION_DIM,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _QUERIES, _ACTION_HORIZON, _ACTION_DIM)

    actions = guide_pi0.GuidePi0.sample_guided_actions(
        model,
        jax.random.key(0),
        observation,
        guide=guide,
        num_steps=1,
        noise=grouped_noise,
    )

    assert actions.shape == grouped_noise.shape
    np.testing.assert_array_equal(actions, grouped_noise)
    assert len(guide_encoder_stub.calls) == 1
    assert len(prefix.calls) == 1
    assert len(suffix.calls) == 1
    assert len(llm.calls) == 2
    assert len(llm.runtime_calls) == 1
    assert len(action_out_proj.calls) == 1

    prefill_call, denoise_call = llm.calls
    runtime_call = llm.runtime_calls[0]
    assert prefill_call["embedded"][1] is None
    assert prefill_call["kv_cache"] is None
    assert prefill_call["adarms_cond"] is None

    assert denoise_call["embedded"][0] is None
    assert denoise_call["kv_cache"] is llm.kv_cache
    assert denoise_call["adarms_cond"][0] is None
    np.testing.assert_array_equal(
        denoise_call["adarms_cond"][1],
        suffix.adarms_cond,
    )

    guide_mask = jnp.repeat(
        guide_encoder_stub.memory.token_mask,
        _QUERIES,
        axis=0,
    )
    prefix_mask = jnp.concatenate(
        [guide_mask, prefix.mask],
        axis=1,
    )
    suffix_attn_mask = pi0.make_attn_mask(
        suffix.mask,
        suffix.ar_mask,
    )
    prefix_to_suffix_mask = jnp.broadcast_to(
        prefix_mask[:, None, :],
        (_GROUPS * _QUERIES, _ACTION_HORIZON, prefix_mask.shape[1]),
    )
    expected_full_mask = jnp.concatenate(
        [prefix_to_suffix_mask, suffix_attn_mask],
        axis=-1,
    )

    np.testing.assert_array_equal(
        runtime_call["mask"],
        expected_full_mask,
    )
    np.testing.assert_array_equal(
        runtime_call["positions"],
        jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix.mask, axis=-1) - 1,
    )

    # The action block is one bidirectional block, not token-by-token causal.
    assert bool(runtime_call["mask"][0, 0, -1])
    assert bool(runtime_call["mask"][0, 1, -2])


def test_sample_guided_actions_with_memory_skips_guide_encoder(monkeypatch) -> None:
    model, guide_encoder_stub, prefix, suffix, llm, action_out_proj = _make_sampling_model(monkeypatch)
    grouped_noise = jnp.arange(
        _GROUPS * _QUERIES * _ACTION_HORIZON * _ACTION_DIM,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _QUERIES, _ACTION_HORIZON, _ACTION_DIM)

    actions = guide_pi0.GuidePi0.sample_guided_actions_with_memory(
        model,
        jax.random.key(0),
        _make_observation(),
        guide_memory=guide_encoder_stub.memory,
        num_steps=1,
        noise=grouped_noise,
    )

    np.testing.assert_array_equal(actions, grouped_noise)
    assert guide_encoder_stub.calls == []
    assert len(prefix.calls) == 1
    assert len(suffix.calls) == 1
    assert len(llm.calls) == 2
    assert len(action_out_proj.calls) == 1


def test_raw_and_cached_guide_sampling_are_equivalent_with_fixed_noise(monkeypatch) -> None:
    model, guide_encoder_stub, *_ = _make_sampling_model(monkeypatch)
    guide = _make_guide()
    grouped_noise = jnp.arange(
        _GROUPS * _QUERIES * _ACTION_HORIZON * _ACTION_DIM,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _QUERIES, _ACTION_HORIZON, _ACTION_DIM)

    raw_actions = guide_pi0.GuidePi0.sample_guided_actions(
        model,
        jax.random.key(0),
        _make_observation(),
        guide=guide,
        num_steps=1,
        noise=grouped_noise,
    )
    cached_actions = guide_pi0.GuidePi0.sample_guided_actions_with_memory(
        model,
        jax.random.key(1),
        _make_observation(),
        guide_memory=guide_encoder_stub.memory,
        num_steps=1,
        noise=grouped_noise,
    )

    np.testing.assert_array_equal(cached_actions, raw_actions)
    assert guide_encoder_stub.calls == [guide]


@pytest.mark.parametrize(
    ("memory", "message"),
    [
        (
            guide_encoder.GuideMemory(
                tokens=jnp.zeros((1, _GUIDE_TOKENS, _PREFIX_WIDTH)),
                token_mask=jnp.ones((1, _GUIDE_TOKENS), dtype=jnp.bool_),
            ),
            "share G",
        ),
        (
            guide_encoder.GuideMemory(
                tokens=jnp.zeros((_GROUPS, _GUIDE_TOKENS, _PREFIX_WIDTH)),
                token_mask=jnp.ones((_GROUPS, _GUIDE_TOKENS + 1), dtype=jnp.bool_),
            ),
            "share \\[G, S\\]",
        ),
        (
            guide_encoder.GuideMemory(
                tokens=jnp.zeros((_GROUPS, _GUIDE_TOKENS, _PREFIX_WIDTH)),
                token_mask=jnp.ones((_GROUPS, _GUIDE_TOKENS), dtype=jnp.float32),
            ),
            "boolean dtype",
        ),
        (
            guide_encoder.GuideMemory(
                tokens=jnp.zeros((_GROUPS, _GUIDE_TOKENS)),
                token_mask=jnp.ones((_GROUPS, _GUIDE_TOKENS), dtype=jnp.bool_),
            ),
            "tokens must have shape",
        ),
        (
            guide_encoder.GuideMemory(
                tokens=jnp.zeros((_GROUPS, _GUIDE_TOKENS, _PREFIX_WIDTH)),
                token_mask=jnp.ones((_GROUPS, _GUIDE_TOKENS, 1), dtype=jnp.bool_),
            ),
            "token_mask must have shape",
        ),
    ],
)
def test_sample_guided_actions_with_memory_validates_memory(
    monkeypatch,
    memory,
    message,
) -> None:
    model, *_ = _make_sampling_model(monkeypatch)

    with pytest.raises(ValueError, match=message):
        guide_pi0.GuidePi0.sample_guided_actions_with_memory(
            model,
            jax.random.key(0),
            _make_observation(),
            guide_memory=memory,
            num_steps=1,
        )


def test_sample_guided_actions_rejects_flat_noise(monkeypatch) -> None:
    model, *_ = _make_sampling_model(monkeypatch)
    flat_noise = jnp.zeros(
        (_GROUPS * _QUERIES, _ACTION_HORIZON, _ACTION_DIM),
        dtype=jnp.float32,
    )

    with pytest.raises(ValueError, match="noise"):
        guide_pi0.GuidePi0.sample_guided_actions(
            model,
            jax.random.key(0),
            _make_observation(),
            guide=_make_guide(),
            num_steps=1,
            noise=flat_noise,
        )


def test_sample_guided_actions_generates_flat_noise_and_restores_group_shape(monkeypatch) -> None:
    model, *_ = _make_sampling_model(monkeypatch)

    actions = guide_pi0.GuidePi0.sample_guided_actions(
        model,
        jax.random.key(0),
        _make_observation(),
        guide=_make_guide(),
        num_steps=1,
    )

    assert actions.shape == (
        _GROUPS,
        _QUERIES,
        _ACTION_HORIZON,
        _ACTION_DIM,
    )
