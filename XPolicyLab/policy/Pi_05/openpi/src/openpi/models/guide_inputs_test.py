import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput


def _make_guide() -> GuideInput:
    return GuideInput(
        images=jnp.arange(2 * 3 * 2 * 2 * 3, dtype=jnp.uint8).reshape(2, 3, 2, 2, 3),
        image_mask=jnp.array([[True, True, False], [True, False, False]]),
        text_tokens=jnp.arange(2 * 2 * 4, dtype=jnp.int32).reshape(2, 2, 4),
        text_mask=jnp.array(
            [
                [[True, True, True, False], [True, True, False, False]],
                [[True, True, False, False], [False, False, False, False]],
            ]
        ),
        unit_mask=jnp.array([[True, True], [True, False]]),
        before_slot=jnp.array([[0, 1], [0, 1]], dtype=jnp.int32),
        after_slot=jnp.array([[1, 2], [1, 1]], dtype=jnp.int32),
    )


def _make_observation(*, include_prompt: bool = True) -> _model.Observation:
    group_ids = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
    images = {
        key: (
            jnp.arange(2 * 3 * 2 * 2 * 3, dtype=jnp.float32).reshape(2, 3, 2, 2, 3)
            + index * 1000
        )
        for index, key in enumerate(_model.IMAGE_KEYS)
    }
    image_masks = {
        key: (jnp.arange(2 * 3).reshape(2, 3) % (index + 2) != 0)
        for index, key in enumerate(_model.IMAGE_KEYS)
    }

    if include_prompt:
        tokenized_prompt = jnp.arange(2 * 3 * 5, dtype=jnp.int32).reshape(2, 3, 5)
        tokenized_prompt_mask = jnp.array(
            [
                [[True, True, True, False, False]] * 3,
                [[True, True, False, False, False]] * 3,
            ]
        )
        token_ar_mask = jnp.broadcast_to(jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32), (2, 3, 5))
        token_loss_mask = jnp.broadcast_to(jnp.array([False, False, True, True, True]), (2, 3, 5))
    else:
        tokenized_prompt = None
        tokenized_prompt_mask = None
        token_ar_mask = None
        token_loss_mask = None

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=group_ids[..., None] * 100 + jnp.arange(4, dtype=jnp.float32),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )


def _assert_same_pytree_values(expected, actual) -> None:
    assert jax.tree_util.tree_structure(expected) == jax.tree_util.tree_structure(actual)
    for expected_leaf, actual_leaf in zip(
        jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(expected_leaf, actual_leaf)


def test_guide_dataclasses_are_pytrees_and_jittable() -> None:
    guide = _make_guide()
    batch = GuideConditionedBatch(
        observation=_make_observation(),
        actions=jnp.arange(2 * 3 * 50 * 32, dtype=jnp.float32).reshape(2, 3, 50, 32),
        guide=guide,
    )

    for value in (guide, batch):
        leaves, treedef = jax.tree_util.tree_flatten(value)
        assert leaves
        assert all(hasattr(leaf, "shape") and hasattr(leaf, "dtype") for leaf in leaves)

        mapped = jax.tree_util.tree_map(lambda leaf: leaf, value)
        assert jax.tree_util.tree_structure(mapped) == treedef
        _assert_same_pytree_values(value, mapped)

        jitted = jax.jit(lambda argument: argument)(value)
        _assert_same_pytree_values(value, jitted)


def test_guide_conditioned_batch_does_not_change_stock_observation_definition() -> None:
    expected_fields = (
        "images",
        "image_masks",
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
    )
    assert tuple(field.name for field in dataclasses.fields(_model.Observation)) == expected_fields

    batch = GuideConditionedBatch(
        observation=_make_observation(),
        actions=jnp.zeros((2, 3, 50, 32), dtype=jnp.float32),
        guide=_make_guide(),
    )
    assert isinstance(batch.observation, _model.Observation)
    assert isinstance(batch.guide, GuideInput)
