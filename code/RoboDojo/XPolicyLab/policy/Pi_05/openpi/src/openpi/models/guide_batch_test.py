import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_inputs import broadcast_guide_memory
from openpi.models.guide_inputs import flatten_grouped_control
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.shared import array_typing as at


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
        tokenized_prompt_mask = jnp.ones((2, 3, 5), dtype=jnp.bool_)
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


def _make_guide() -> GuideInput:
    return GuideInput(
        boundary_images=jnp.zeros((2, 3, 3, 2, 2, 3), dtype=jnp.float32),
        boundary_image_mask=jnp.ones((2, 3, 3), dtype=jnp.bool_),
        boundary_text_tokens=jnp.zeros((2, 3, 3, 4), dtype=jnp.int32),
        boundary_text_mask=jnp.ones((2, 3, 3, 4), dtype=jnp.bool_),
        transition_text_tokens=jnp.zeros((2, 2, 4), dtype=jnp.int32),
        transition_text_mask=jnp.ones((2, 2, 4), dtype=jnp.bool_),
        boundary_mask=jnp.ones((2, 3), dtype=jnp.bool_),
        unit_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        memory_source_kind=jnp.zeros((2, 8), dtype=jnp.int32),
        memory_source_index=jnp.zeros((2, 8), dtype=jnp.int32),
        memory_source_offset=jnp.zeros((2, 8), dtype=jnp.int32),
        memory_mask=jnp.ones((2, 8), dtype=jnp.bool_),
    )


def _make_batch(*, include_prompt: bool = True) -> GuideConditionedBatch:
    return GuideConditionedBatch(
        observation=_make_observation(include_prompt=include_prompt),
        actions=jnp.zeros((2, 3, 50, 32), dtype=jnp.float32),
        guide=_make_guide(),
    )


def test_flatten_grouped_control_is_group_major_for_every_observation_field() -> None:
    groups, queries = 2, 3
    observation = _make_observation()
    actions = jnp.arange(groups * queries * 50 * 32, dtype=jnp.float32).reshape(groups, queries, 50, 32)

    flat_observation, flat_actions = flatten_grouped_control(observation, actions)

    assert isinstance(flat_observation, _model.Observation)
    assert flat_actions.shape == (groups * queries, 50, 32)
    np.testing.assert_array_equal(flat_actions, actions.reshape(groups * queries, 50, 32))

    for key in _model.IMAGE_KEYS:
        expected = observation.images[key].reshape(groups * queries, 2, 2, 3)
        np.testing.assert_array_equal(flat_observation.images[key], expected)
        expected_mask = observation.image_masks[key].reshape(groups * queries)
        np.testing.assert_array_equal(flat_observation.image_masks[key], expected_mask)

    np.testing.assert_array_equal(flat_observation.state, observation.state.reshape(groups * queries, 4))
    np.testing.assert_array_equal(
        flat_observation.tokenized_prompt,
        observation.tokenized_prompt.reshape(groups * queries, 5),
    )
    np.testing.assert_array_equal(
        flat_observation.tokenized_prompt_mask,
        observation.tokenized_prompt_mask.reshape(groups * queries, 5),
    )
    np.testing.assert_array_equal(
        flat_observation.token_ar_mask,
        observation.token_ar_mask.reshape(groups * queries, 5),
    )
    np.testing.assert_array_equal(
        flat_observation.token_loss_mask,
        observation.token_loss_mask.reshape(groups * queries, 5),
    )

    jitted_observation, jitted_actions = jax.jit(flatten_grouped_control)(observation, actions)
    np.testing.assert_array_equal(jitted_actions, flat_actions)
    np.testing.assert_array_equal(jitted_observation.state, flat_observation.state)


def test_flatten_grouped_control_preserves_optional_none_fields() -> None:
    observation = _make_observation(include_prompt=False)
    actions = jnp.zeros((2, 3, 50, 32), dtype=jnp.float32)

    flat_observation, flat_actions = flatten_grouped_control(observation, actions)

    assert flat_observation.tokenized_prompt is None
    assert flat_observation.tokenized_prompt_mask is None
    assert flat_observation.token_ar_mask is None
    assert flat_observation.token_loss_mask is None
    assert flat_actions.shape == (6, 50, 32)


def test_guide_memory_broadcast_is_group_major() -> None:
    groups, queries, slots, width = 2, 3, 4, 5
    guide_memory = jnp.arange(groups * slots * width, dtype=jnp.float32).reshape(groups, slots, width)

    actual = broadcast_guide_memory(guide_memory, queries_per_guide=queries)
    expected = jnp.repeat(guide_memory, queries, axis=0)

    assert actual.shape == (groups * queries, slots, width)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], guide_memory[0])
    np.testing.assert_array_equal(actual[queries], guide_memory[1])


def test_guide_memory_broadcast_supports_rank_two_masks() -> None:
    groups, queries, slots = 2, 3, 4
    guide_mask = jnp.array(
        [
            [True, True, False, False],
            [True, False, True, False],
        ],
        dtype=jnp.bool_,
    )

    actual = broadcast_guide_memory(guide_mask, queries_per_guide=queries)
    expected = jnp.repeat(guide_mask, queries, axis=0)

    assert actual.shape == (groups * queries, slots)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "guide_memory",
    [
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2, 4, 5, 6), dtype=jnp.float32),
    ],
)
def test_guide_memory_broadcast_rejects_unsupported_rank(guide_memory) -> None:
    with pytest.raises(ValueError, match=r"rank|shape"):
        broadcast_guide_memory(guide_memory, queries_per_guide=3)


def test_guide_memory_broadcast_rejects_nonpositive_query_count() -> None:
    guide_memory = jnp.zeros((2, 4, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="queries_per_guide"):
        broadcast_guide_memory(guide_memory, queries_per_guide=0)


def test_validate_guide_conditioned_batch_returns_group_and_query_counts() -> None:
    batch = _make_batch()

    assert validate_guide_conditioned_batch(batch) == (2, 3)


def test_validate_allows_optional_none_observation_fields() -> None:
    batch = _make_batch(include_prompt=False)

    assert validate_guide_conditioned_batch(batch) == (2, 3)


@pytest.mark.parametrize(
    "actions",
    [
        jnp.zeros((2, 3, 50), dtype=jnp.float32),
        jnp.zeros((2, 3, 50, 32, 1), dtype=jnp.float32),
        jnp.zeros((0, 3, 50, 32), dtype=jnp.float32),
        jnp.zeros((2, 0, 50, 32), dtype=jnp.float32),
    ],
)
def test_validate_rejects_invalid_actions_shape_or_empty_group_query(actions) -> None:
    batch = dataclasses.replace(_make_batch(), actions=actions)

    with pytest.raises(ValueError, match=r"actions|group|query"):
        validate_guide_conditioned_batch(batch)


def test_validate_rejects_observation_leaf_without_group_query_axes() -> None:
    batch = _make_batch()
    with at.disable_typechecking():
        invalid_observation = dataclasses.replace(
            batch.observation,
            state=jnp.zeros((2, 2, 4), dtype=jnp.float32),
        )
    batch = dataclasses.replace(batch, observation=invalid_observation)

    with pytest.raises(ValueError, match=r"observation|state"):
        validate_guide_conditioned_batch(batch)


def test_validate_rejects_guide_leaf_with_wrong_group_count() -> None:
    batch = _make_batch()
    invalid_guide = dataclasses.replace(
        batch.guide,
        boundary_images=batch.guide.boundary_images[:1],
    )
    batch = dataclasses.replace(batch, guide=invalid_guide)

    with pytest.raises(ValueError, match=r"guide|boundary_images"):
        validate_guide_conditioned_batch(batch)
