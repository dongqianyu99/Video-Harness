import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import guide_tokens


def test_boundary_assembly_interleaves_each_views_image_and_text() -> None:
    images = jnp.arange(1 * 2 * 3 * 2 * 4, dtype=jnp.float32).reshape(1, 2, 3, 2, 4)
    text = (1000 + jnp.arange(1 * 2 * 3 * 3 * 4, dtype=jnp.float32)).reshape(1, 2, 3, 3, 4)
    image_mask = jnp.ones((1, 2, 3), dtype=jnp.bool_)
    text_mask = jnp.ones((1, 2, 3, 3), dtype=jnp.bool_)
    boundary_mask = jnp.ones((1, 2), dtype=jnp.bool_)

    batch = guide_tokens.assemble_boundary_tokens(
        images,
        image_mask,
        text,
        text_mask,
        boundary_mask,
    )

    expected = jnp.concatenate(
        [segment for view in range(3) for segment in (images[:, :, view], text[:, :, view])],
        axis=2,
    )
    assert batch.tokens.shape == (1, 2, 15, 4)
    np.testing.assert_array_equal(batch.tokens, expected)
    np.testing.assert_array_equal(
        batch.role_ids[0, 0],
        np.repeat(np.arange(6, dtype=np.int32), [2, 3, 2, 3, 2, 3]),
    )


def test_boundary_assembly_masks_padding_exactly_and_requires_three_views() -> None:
    images = jnp.ones((1, 2, 3, 2, 4), dtype=jnp.float32)
    text = jnp.ones((1, 2, 3, 3, 4), dtype=jnp.float32)
    image_mask = jnp.ones((1, 2, 3), dtype=jnp.bool_)
    text_mask = jnp.ones((1, 2, 3, 3), dtype=jnp.bool_)
    boundary_mask = jnp.array([[True, False]])

    batch = guide_tokens.assemble_boundary_tokens(
        images,
        image_mask,
        text,
        text_mask,
        boundary_mask,
    )

    np.testing.assert_array_equal(batch.tokens[0, 1], 0.0)
    assert not np.any(batch.token_mask[0, 1])
    with pytest.raises(ValueError, match="exactly 3 views"):
        guide_tokens.assemble_boundary_tokens(
            images[:, :, :2],
            image_mask[:, :, :2],
            text[:, :, :2],
            text_mask[:, :, :2],
            boundary_mask,
        )


def test_transition_assembly_contains_only_text() -> None:
    text = jnp.arange(1 * 2 * 3 * 4, dtype=jnp.float32).reshape(1, 2, 3, 4)
    text_mask = jnp.array([[[True, True, False], [True, True, True]]])
    unit_mask = jnp.array([[True, False]])

    batch = guide_tokens.assemble_transition_tokens(text, text_mask, unit_mask)

    assert batch.tokens.shape == text.shape
    np.testing.assert_array_equal(batch.tokens[0, 0, 2], 0.0)
    np.testing.assert_array_equal(batch.tokens[0, 1], 0.0)
    np.testing.assert_array_equal(batch.role_ids, 0)


def test_pack_guide_tokens_preserves_host_interleaving_and_type() -> None:
    boundary_memory = jnp.asarray([[[[10.0], [11.0]], [[20.0], [21.0]], [[30.0], [31.0]]]])
    transition_memory = jnp.asarray([[[[100.0]], [[200.0]]]])
    boundary_mask = jnp.array([[True, True, True]])
    unit_mask = jnp.array([[True, True]])
    # B0, T0, B1, T1, B2
    kind = jnp.array([[0, 0, 1, 0, 0, 1, 0, 0]], dtype=jnp.int32)
    index = jnp.array([[0, 0, 0, 1, 1, 1, 2, 2]], dtype=jnp.int32)
    offset = jnp.array([[0, 1, 0, 0, 1, 0, 0, 1]], dtype=jnp.int32)
    mask = jnp.ones((1, 8), dtype=jnp.bool_)
    type_embeddings = jnp.asarray([[1000.0], [2000.0]])

    packed = guide_tokens.pack_guide_tokens(
        boundary_memory,
        transition_memory,
        boundary_mask,
        unit_mask,
        kind,
        index,
        offset,
        mask,
        type_embeddings,
    )

    np.testing.assert_array_equal(
        packed.tokens[..., 0],
        [[1010, 1011, 2100, 1020, 1021, 2200, 1030, 1031]],
    )


def test_pack_guide_tokens_supports_boundary_boundary_gap_and_tail_padding() -> None:
    boundary_memory = jnp.arange(1 * 4 * 2 * 1, dtype=jnp.float32).reshape(1, 4, 2, 1)
    transition_memory = jnp.arange(1 * 2 * 1 * 1, dtype=jnp.float32).reshape(1, 2, 1, 1)
    # B0,T0,B1,B2,T1,B3, then two pads.
    kind = jnp.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=jnp.int32)
    index = jnp.array([[0, 0, 0, 1, 1, 2, 2, 1, 3, 3, 0, 0]], dtype=jnp.int32)
    offset = jnp.array([[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]], dtype=jnp.int32)
    mask = jnp.array([[True] * 10 + [False, False]])

    packed = guide_tokens.pack_guide_tokens(
        boundary_memory,
        transition_memory,
        jnp.ones((1, 4), dtype=jnp.bool_),
        jnp.ones((1, 2), dtype=jnp.bool_),
        kind,
        index,
        offset,
        mask,
        jnp.zeros((2, 1), dtype=jnp.float32),
    )

    assert packed.tokens.shape == (1, 12, 1)
    np.testing.assert_array_equal(packed.tokens[0, 10:], 0.0)
    np.testing.assert_array_equal(packed.token_mask, mask)


def test_pack_guide_tokens_rejects_duplicate_or_incomplete_map() -> None:
    boundary_memory = jnp.zeros((1, 1, 2, 4))
    transition_memory = jnp.zeros((1, 1, 1, 4))
    with pytest.raises(ValueError, match=r"duplicate|cover"):
        guide_tokens.pack_guide_tokens(
            boundary_memory,
            transition_memory,
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.array([[0, 0, 0]], dtype=jnp.int32),
            jnp.zeros((1, 3), dtype=jnp.int32),
            jnp.array([[0, 0, 1]], dtype=jnp.int32),
            jnp.ones((1, 3), dtype=jnp.bool_),
            jnp.zeros((2, 4)),
        )


def test_pack_guide_tokens_jits() -> None:
    args = (
        jnp.zeros((1, 1, 1, 2)),
        jnp.zeros((1, 1, 1, 2)),
        jnp.ones((1, 1), dtype=jnp.bool_),
        jnp.ones((1, 1), dtype=jnp.bool_),
        jnp.array([[0, 1]], dtype=jnp.int32),
        jnp.array([[0, 0]], dtype=jnp.int32),
        jnp.array([[0, 0]], dtype=jnp.int32),
        jnp.ones((1, 2), dtype=jnp.bool_),
        jnp.zeros((2, 2)),
    )
    eager = guide_tokens.pack_guide_tokens(*args)
    compiled = jax.jit(guide_tokens.pack_guide_tokens)(*args)
    np.testing.assert_array_equal(compiled.tokens, eager.tokens)
