from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.training import sharding as stock_sharding
from openpi.training.guide_train_sharding import make_guided_batch_sharding
from openpi.training.guide_train_sharding import put_guided_batch


def _batch(*, groups: int = 1, queries: int = 4) -> GuideConditionedBatch:
    observation = _model.Observation(
        images={
            key: jnp.ones((groups, queries, 2, 2, 3), dtype=jnp.float32)
            for key in _model.IMAGE_KEYS
        },
        image_masks={
            key: jnp.ones((groups, queries), dtype=jnp.bool_)
            for key in _model.IMAGE_KEYS
        },
        state=jnp.ones((groups, queries, 3), dtype=jnp.float32),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )
    guide = GuideInput(
        images=jnp.ones((groups, 2, 2, 2, 3), dtype=jnp.float32),
        image_mask=jnp.ones((groups, 2), dtype=jnp.bool_),
        text_tokens=jnp.ones((groups, 1, 4), dtype=jnp.int32),
        text_mask=jnp.ones((groups, 1, 4), dtype=jnp.bool_),
        unit_mask=jnp.ones((groups, 1), dtype=jnp.bool_),
        before_slot=jnp.zeros((groups, 1), dtype=jnp.int32),
        after_slot=jnp.ones((groups, 1), dtype=jnp.int32),
    )
    return GuideConditionedBatch(
        observation=observation,
        actions=jnp.ones((groups, queries, 3, 2), dtype=jnp.float32),
        guide=guide,
    )


def test_sharding_replicates_g_and_guide_and_shards_q_when_mesh_has_data_axes() -> None:
    mesh = stock_sharding.make_mesh(1)
    sharding = make_guided_batch_sharding(_batch(queries=4), mesh)

    leaves = jax.tree_util.tree_leaves(sharding)
    assert leaves
    assert all(isinstance(leaf, jax.sharding.NamedSharding) for leaf in leaves)
    assert all(leaf.spec == jax.sharding.PartitionSpec() for leaf in leaves)

    placed = put_guided_batch(_batch(queries=4), sharding)
    assert placed.actions.shape == (1, 4, 3, 2)
    assert placed.guide.images.shape == (1, 2, 2, 2, 3)


def test_nontrivial_mesh_uses_none_for_g_and_data_axes_for_q(monkeypatch) -> None:
    class _FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(jax.sharding, "NamedSharding", _FakeNamedSharding)
    mesh = SimpleNamespace(
        shape={stock_sharding.BATCH_AXIS: 2, stock_sharding.FSDP_AXIS: 1},
    )
    sharding = make_guided_batch_sharding(_batch(queries=4), mesh)

    action_spec = sharding.actions.spec
    image_spec = sharding.observation.images["base_0_rgb"].spec
    state_spec = sharding.observation.state.spec
    guide_spec = sharding.guide.images.spec
    expected_query_spec = jax.sharding.PartitionSpec(None, stock_sharding.DATA_AXIS)
    assert action_spec == expected_query_spec
    assert image_spec == expected_query_spec
    assert state_spec == expected_query_spec
    assert guide_spec == jax.sharding.PartitionSpec()

    # Q is sharded over the mesh data axes as one logical axis.  Expanding
    # DATA_AXIS would incorrectly shard the next physical tensor dimension
    # (e.g. action horizon or image height).
    assert action_spec != jax.sharding.PartitionSpec(None, *stock_sharding.DATA_AXIS)


def test_query_count_must_be_divisible_by_data_mesh_size() -> None:
    mesh = SimpleNamespace(
        shape={stock_sharding.BATCH_AXIS: 2, stock_sharding.FSDP_AXIS: 1},
    )

    with pytest.raises(ValueError, match=r"Q=3.*data mesh size 2"):
        make_guided_batch_sharding(_batch(queries=3), mesh)  # type: ignore[arg-type]


def test_optional_observation_fields_remain_none() -> None:
    mesh = stock_sharding.make_mesh(1)
    sharding = make_guided_batch_sharding(_batch(), mesh)

    assert sharding.observation.tokenized_prompt is None
    assert sharding.observation.tokenized_prompt_mask is None


def test_invalid_leading_dimensions_fail_clearly() -> None:
    mesh = stock_sharding.make_mesh(1)
    batch = _batch()
    invalid_images = {key: value[:, :3] for key, value in batch.observation.images.items()}
    invalid_masks = {key: value[:, :3] for key, value in batch.observation.image_masks.items()}
    invalid_observation = dataclasses.replace(
        batch.observation,
        images=invalid_images,
        image_masks=invalid_masks,
        state=batch.observation.state[:, :3],
    )
    invalid_batch = GuideConditionedBatch(
        observation=invalid_observation,
        actions=batch.actions,
        guide=batch.guide,
    )

    with pytest.raises(ValueError, match="leading dimensions"):
        make_guided_batch_sharding(invalid_batch, mesh)
