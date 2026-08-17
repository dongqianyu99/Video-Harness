import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.guide_resampler import UnitResampler
from openpi.shared import nnx_utils

_GROUPS = 2
_UNITS = 3
_TOKENS = 4
_INPUT_DIM = 6
_OUTPUT_DIM = 10
_WIDTH = 8
_QUERIES = 2
_HEADS = 2


def _make_resampler(*, seed: int = 0) -> UnitResampler:
    return UnitResampler(
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        num_queries=_QUERIES,
        width=_WIDTH,
        num_heads=_HEADS,
        ffn_hidden_dim=16,
        rngs=nnx.Rngs(seed),
    )


def _make_inputs() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    unit_tokens = jax.random.normal(
        jax.random.key(1),
        (_GROUPS, _UNITS, _TOKENS, _INPUT_DIM),
    )
    token_mask = jnp.ones((_GROUPS, _UNITS, _TOKENS), dtype=jnp.bool_)
    role_ids = jnp.broadcast_to(
        jnp.arange(_TOKENS, dtype=jnp.int32) % 3,
        (_GROUPS, _UNITS, _TOKENS),
    )
    unit_mask = jnp.ones((_GROUPS, _UNITS), dtype=jnp.bool_)
    return unit_tokens, token_mask, role_ids, unit_mask


def test_unit_resampler_shape_and_jit() -> None:
    resampler = _make_resampler()
    inputs = _make_inputs()

    eager_output = resampler(*inputs)
    jitted_output = nnx_utils.module_jit(resampler.__call__)(*inputs)

    assert eager_output.shape == (_GROUPS, _UNITS, _QUERIES, _OUTPUT_DIM)
    np.testing.assert_allclose(jitted_output, eager_output, rtol=1e-5, atol=1e-6)


def test_padding_token_content_does_not_change_output() -> None:
    resampler = _make_resampler()
    unit_tokens, _, role_ids, unit_mask = _make_inputs()
    token_mask = jnp.broadcast_to(
        jnp.array([True, True, False, False]),
        (_GROUPS, _UNITS, _TOKENS),
    )
    changed_tokens = jnp.where(token_mask[..., None], unit_tokens, unit_tokens + 1000.0)

    original_output = resampler(unit_tokens, token_mask, role_ids, unit_mask)
    changed_output = resampler(changed_tokens, token_mask, role_ids, unit_mask)

    np.testing.assert_allclose(changed_output, original_output, rtol=1e-5, atol=1e-6)


def test_masked_unit_is_exactly_zero_and_all_token_masked_is_finite() -> None:
    resampler = _make_resampler()
    unit_tokens, token_mask, role_ids, _ = _make_inputs()
    token_mask = token_mask.at[0, 0].set(False)
    unit_mask = jnp.ones((_GROUPS, _UNITS), dtype=jnp.bool_).at[0, 1].set(False)

    output = resampler(unit_tokens, token_mask, role_ids, unit_mask)

    assert np.all(np.isfinite(output))
    np.testing.assert_array_equal(output[0, 1], np.zeros((_QUERIES, _OUTPUT_DIM), dtype=np.float32))
    assert np.all(np.isfinite(output[0, 0]))


def test_units_do_not_attend_to_each_other() -> None:
    resampler = _make_resampler()
    unit_tokens, token_mask, role_ids, unit_mask = _make_inputs()
    changed_tokens = unit_tokens.at[0, 1].add(10.0)

    original_output = resampler(unit_tokens, token_mask, role_ids, unit_mask)
    changed_output = resampler(changed_tokens, token_mask, role_ids, unit_mask)

    np.testing.assert_allclose(changed_output[0, 0], original_output[0, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(changed_output[0, 2], original_output[0, 2], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(changed_output[1], original_output[1], rtol=1e-5, atol=1e-6)
    assert not np.allclose(changed_output[0, 1], original_output[0, 1])


def test_valid_token_content_changes_output() -> None:
    resampler = _make_resampler()
    unit_tokens, token_mask, role_ids, unit_mask = _make_inputs()
    changed_tokens = unit_tokens.at[0, 0, 0].add(1.0)

    original_output = resampler(unit_tokens, token_mask, role_ids, unit_mask)
    changed_output = resampler(changed_tokens, token_mask, role_ids, unit_mask)

    assert not np.allclose(changed_output[0, 0], original_output[0, 0])


def test_role_ids_change_output() -> None:
    resampler = _make_resampler()
    unit_tokens, token_mask, role_ids, unit_mask = _make_inputs()
    changed_role_ids = role_ids.at[0, 0, 0].set((role_ids[0, 0, 0] + 1) % 3)

    original_output = resampler(unit_tokens, token_mask, role_ids, unit_mask)
    changed_output = resampler(unit_tokens, token_mask, changed_role_ids, unit_mask)

    assert not np.allclose(changed_output[0, 0], original_output[0, 0])


def test_gradients_reach_inputs_and_all_resampler_components() -> None:
    resampler = _make_resampler()
    unit_tokens, _, role_ids, unit_mask = _make_inputs()
    token_mask = jnp.broadcast_to(
        jnp.array([True, True, True, False]),
        (_GROUPS, _UNITS, _TOKENS),
    )

    def loss_fn(module: UnitResampler, tokens: jax.Array) -> jax.Array:
        output = module(tokens, token_mask, role_ids, unit_mask)
        weights = jnp.linspace(0.5, 1.5, output.size, dtype=output.dtype).reshape(output.shape)
        return jnp.mean(jnp.square(output) * weights)

    parameter_grads, input_grads = nnx.grad(loss_fn, argnums=(0, 1))(resampler, unit_tokens)
    flat_grads = {path: variable.value for path, variable in parameter_grads.flat_state().items()}

    required_parameter_paths = (
        ("learned_queries",),
        ("input_projection", "kernel"),
        ("output_projection", "kernel"),
        ("query_projection", "kernel"),
        ("key_projection", "kernel"),
        ("value_projection", "kernel"),
        ("attention_output_projection", "kernel"),
        ("ffn_in", "kernel"),
        ("ffn_out", "kernel"),
    )
    for path in required_parameter_paths:
        assert path in flat_grads
        assert np.any(np.abs(flat_grads[path]) > 1e-9), f"Expected nonzero gradient for {path}"

    input_grads = np.asarray(input_grads)
    token_mask_np = np.asarray(token_mask)
    assert np.any(np.abs(input_grads[token_mask_np]) > 1e-9)
    np.testing.assert_array_equal(input_grads[~token_mask_np], 0.0)


def test_output_projection_is_small_and_nonzero_at_initialization() -> None:
    resampler = _make_resampler()
    output_kernel = np.asarray(resampler.output_projection.kernel.value)

    assert np.any(output_kernel != 0.0)
    assert np.max(np.abs(output_kernel)) < 0.05


def test_invalid_shapes_and_role_ids_fail_clearly_outside_jit() -> None:
    resampler = _make_resampler()
    unit_tokens, token_mask, role_ids, unit_mask = _make_inputs()

    with pytest.raises(ValueError, match="token_mask"):
        resampler(unit_tokens, token_mask[..., :-1], role_ids, unit_mask)

    invalid_role_ids = role_ids.at[0, 0, 0].set(3)
    with pytest.raises(ValueError, match="role_ids"):
        resampler(unit_tokens, token_mask, invalid_role_ids, unit_mask)
