import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import guide_tokens

_GROUPS = 2
_FRAMES = 3
_PATCHES = 2
_TEXT_TOKENS = 2
_WIDTH = 3
_UNITS = 3


def _make_inputs(*, masked: bool = False):
    frame_tokens = jnp.arange(
        _GROUPS * _FRAMES * _PATCHES * _WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _FRAMES, _PATCHES, _WIDTH)
    text_embeddings = (
        jnp.arange(
            _GROUPS * _UNITS * _TEXT_TOKENS * _WIDTH,
            dtype=jnp.float32,
        ).reshape(_GROUPS, _UNITS, _TEXT_TOKENS, _WIDTH)
        + 1000.0
    )

    frame_mask = jnp.ones((_GROUPS, _FRAMES), dtype=jnp.bool_)
    text_mask = jnp.ones((_GROUPS, _UNITS, _TEXT_TOKENS), dtype=jnp.bool_)
    unit_mask = jnp.ones((_GROUPS, _UNITS), dtype=jnp.bool_)
    before_slot = jnp.array(
        [[0, 1, 2], [1, 2, 0]],
        dtype=jnp.int32,
    )
    after_slot = jnp.array(
        [[1, 2, 0], [2, 0, 1]],
        dtype=jnp.int32,
    )

    if masked:
        frame_mask = jnp.array(
            [[True, False, True], [False, True, True]],
            dtype=jnp.bool_,
        )
        text_mask = jnp.array(
            [
                [[True, False], [False, True], [True, True]],
                [[False, True], [True, False], [True, True]],
            ],
            dtype=jnp.bool_,
        )
        unit_mask = jnp.array(
            [[True, True, False], [True, False, True]],
            dtype=jnp.bool_,
        )
        before_slot = jnp.array(
            [[0, 1, -1], [2, -1, 0]],
            dtype=jnp.int32,
        )
        after_slot = jnp.array(
            [[1, 2, -1], [0, -1, 1]],
            dtype=jnp.int32,
        )

    return frame_tokens, frame_mask, text_embeddings, text_mask, unit_mask, before_slot, after_slot


def _assemble(inputs):
    return guide_tokens.assemble_unit_tokens(*inputs)


def test_unit_token_batch_is_a_jax_pytree() -> None:
    value = guide_tokens.UnitTokenBatch(
        tokens=jnp.zeros((_GROUPS, _UNITS, 2 * _PATCHES + _TEXT_TOKENS, _WIDTH)),
        token_mask=jnp.ones((_GROUPS, _UNITS, 2 * _PATCHES + _TEXT_TOKENS), dtype=jnp.bool_),
        role_ids=jnp.zeros((_GROUPS, _UNITS, 2 * _PATCHES + _TEXT_TOKENS), dtype=jnp.int32),
    )

    leaves = jax.tree_util.tree_leaves(value)
    assert len(leaves) == 3
    mapped = jax.tree_util.tree_map(lambda leaf: leaf, value)
    assert jax.tree_util.tree_structure(mapped) == jax.tree_util.tree_structure(value)
    jitted = jax.jit(lambda batch: batch)(value)
    np.testing.assert_array_equal(jitted.tokens, value.tokens)


def test_assembly_orders_before_text_after_and_assigns_roles() -> None:
    inputs = _make_inputs()
    frame_tokens, _, text_embeddings, _, _, before_slot, after_slot = inputs

    batch = _assemble(inputs)
    group_indices = jnp.arange(_GROUPS)[:, None]
    expected_before = frame_tokens[group_indices, before_slot]
    expected_after = frame_tokens[group_indices, after_slot]
    expected_tokens = jnp.concatenate(
        [expected_before, text_embeddings, expected_after],
        axis=2,
    )
    expected_roles = jnp.concatenate(
        [
            jnp.full((_GROUPS, _UNITS, _PATCHES), guide_tokens.BEFORE_ROLE, dtype=jnp.int32),
            jnp.full((_GROUPS, _UNITS, _TEXT_TOKENS), guide_tokens.TEXT_ROLE, dtype=jnp.int32),
            jnp.full((_GROUPS, _UNITS, _PATCHES), guide_tokens.AFTER_ROLE, dtype=jnp.int32),
        ],
        axis=2,
    )

    assert batch.tokens.shape == (_GROUPS, _UNITS, 2 * _PATCHES + _TEXT_TOKENS, _WIDTH)
    np.testing.assert_array_equal(batch.tokens, expected_tokens)
    np.testing.assert_array_equal(batch.token_mask, jnp.ones(batch.token_mask.shape, dtype=jnp.bool_))
    np.testing.assert_array_equal(batch.role_ids, expected_roles)


def test_adjacent_units_can_share_a_boundary_frame() -> None:
    inputs = _make_inputs()
    batch = _assemble(inputs)

    np.testing.assert_array_equal(batch.tokens[0, 0, -_PATCHES:], batch.tokens[0, 1, :_PATCHES])
    np.testing.assert_array_equal(batch.tokens[0, 1, -_PATCHES:], batch.tokens[0, 2, :_PATCHES])
    np.testing.assert_array_equal(batch.tokens[1, 0, -_PATCHES:], batch.tokens[1, 1, :_PATCHES])


def test_masks_are_combined_and_masked_values_are_exactly_zero() -> None:
    inputs = _make_inputs(masked=True)
    frame_tokens, frame_mask, text_embeddings, text_mask, unit_mask, before_slot, after_slot = inputs
    batch = _assemble(inputs)

    safe_before_slot = jnp.where(unit_mask, before_slot, 0)
    safe_after_slot = jnp.where(unit_mask, after_slot, 0)
    group_indices = jnp.arange(_GROUPS)[:, None]
    before_frame_mask = frame_mask[group_indices, safe_before_slot]
    after_frame_mask = frame_mask[group_indices, safe_after_slot]
    expected_mask = jnp.concatenate(
        [
            jnp.repeat(unit_mask[..., None] & before_frame_mask[..., None], _PATCHES, axis=2),
            unit_mask[..., None] & text_mask,
            jnp.repeat(unit_mask[..., None] & after_frame_mask[..., None], _PATCHES, axis=2),
        ],
        axis=2,
    )

    expected_tokens = jnp.concatenate(
        [
            frame_tokens[group_indices, safe_before_slot],
            text_embeddings,
            frame_tokens[group_indices, safe_after_slot],
        ],
        axis=2,
    )
    expected_tokens = jnp.where(expected_mask[..., None], expected_tokens, 0.0)

    np.testing.assert_array_equal(batch.token_mask, expected_mask)
    np.testing.assert_array_equal(batch.tokens, expected_tokens)
    assert np.all(np.isfinite(batch.tokens))
    np.testing.assert_array_equal(batch.tokens[0, 2], 0.0)
    np.testing.assert_array_equal(batch.tokens[1, 1], 0.0)
    assert set(np.asarray(batch.role_ids).tolist()[0][0]) <= {
        guide_tokens.BEFORE_ROLE,
        guide_tokens.TEXT_ROLE,
        guide_tokens.AFTER_ROLE,
    }


def test_padding_unit_negative_slots_do_not_read_last_frame() -> None:
    inputs = _make_inputs(masked=True)
    frame_tokens = inputs[0].at[:, -1].set(999_999.0)
    batch = _assemble((frame_tokens, *inputs[1:]))

    np.testing.assert_array_equal(batch.tokens[0, 2], 0.0)
    np.testing.assert_array_equal(batch.tokens[1, 1], 0.0)
    assert np.all(batch.token_mask[0, 2] == 0)
    assert np.all(batch.token_mask[1, 1] == 0)


def test_jit_assembly_matches_eager() -> None:
    inputs = _make_inputs(masked=True)
    eager = _assemble(inputs)
    jitted = jax.jit(_assemble)(inputs)

    np.testing.assert_array_equal(jitted.tokens, eager.tokens)
    np.testing.assert_array_equal(jitted.token_mask, eager.token_mask)
    np.testing.assert_array_equal(jitted.role_ids, eager.role_ids)


def test_valid_unit_out_of_range_slot_fails_in_eager_mode() -> None:
    inputs = _make_inputs()
    bad_before_slot = inputs[5].at[0, 0].set(_FRAMES)

    with pytest.raises(ValueError, match="before_slot"):
        _assemble((*inputs[:5], bad_before_slot, inputs[6]))


def test_valid_unit_negative_slot_fails_in_eager_mode() -> None:
    inputs = _make_inputs()
    bad_after_slot = inputs[6].at[0, 0].set(-1)

    with pytest.raises(ValueError, match="after_slot"):
        _assemble((*inputs[:6], bad_after_slot))


def test_shape_and_dtype_errors_fail_clearly_in_eager_mode() -> None:
    inputs = _make_inputs()
    frame_tokens, frame_mask, text_embeddings, text_mask, unit_mask, before_slot, after_slot = inputs

    with pytest.raises(ValueError, match="frame_mask"):
        _assemble((frame_tokens, frame_mask.astype(jnp.int32), text_embeddings, text_mask, unit_mask, before_slot, after_slot))

    with pytest.raises(ValueError, match="text_embeddings"):
        _assemble((frame_tokens, frame_mask, text_embeddings[..., :-1], text_mask, unit_mask, before_slot, after_slot))

    with pytest.raises(ValueError, match="text_mask"):
        _assemble((frame_tokens, frame_mask, text_embeddings, text_mask[..., :-1], unit_mask, before_slot, after_slot))
