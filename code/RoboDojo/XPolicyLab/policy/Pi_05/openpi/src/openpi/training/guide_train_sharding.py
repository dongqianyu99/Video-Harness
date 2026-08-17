from __future__ import annotations

from typing import Any

import jax

from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import validate_guide_conditioned_batch
import openpi.training.sharding as _sharding


def _data_axis_size(mesh: Any) -> int:
    try:
        return int(mesh.shape[_sharding.BATCH_AXIS]) * int(mesh.shape[_sharding.FSDP_AXIS])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "mesh must provide batch and fsdp axes for guided data sharding"
        ) from exc


def _replicated_sharding(mesh: Any) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _query_sharding(
    leaf: Any,
    *,
    groups: int,
    queries: int,
    mesh: Any,
    data_axis_size: int,
    name: str,
) -> Any:
    if leaf is None:
        return None
    if not hasattr(leaf, "shape"):
        raise ValueError(f"{name} must contain only array leaves, got {type(leaf).__name__}")

    shape = tuple(leaf.shape)
    if len(shape) < 2 or shape[:2] != (groups, queries):
        raise ValueError(
            f"{name} leaf must have leading dimensions {(groups, queries)}, got {shape}"
        )

    if data_axis_size == 1:
        return _replicated_sharding(mesh)

    if queries % data_axis_size != 0:
        raise ValueError(
            f"Q={queries} must be divisible by the data mesh size {data_axis_size}"
        )

    return jax.sharding.NamedSharding(
        mesh,
        # The grouped batch's Q axis is the only data axis.  Keep the mesh
        # axes nested in one PartitionSpec entry so subsequent tensor axes
        # (horizon, feature, image height, ...) remain replicated.
        jax.sharding.PartitionSpec(None, _sharding.DATA_AXIS),
    )


def _guide_sharding(leaf: Any, *, groups: int, mesh: Any, name: str) -> Any:
    if leaf is None:
        return None
    if not hasattr(leaf, "shape"):
        raise ValueError(f"{name} must contain only array leaves, got {type(leaf).__name__}")

    shape = tuple(leaf.shape)
    if not shape or shape[0] != groups:
        raise ValueError(
            f"{name} leaf must have leading group dimension {groups}, got {shape}"
        )
    return _replicated_sharding(mesh)


def _map_sharding(tree: Any, fn) -> Any:
    return jax.tree_util.tree_map(fn, tree)


def make_guided_batch_sharding(
    batch: GuideConditionedBatch,
    mesh: jax.sharding.Mesh,
) -> GuideConditionedBatch:
    """Create sharding leaves with replicated G/Guide and sharded Q semantics."""

    groups, queries = validate_guide_conditioned_batch(batch)
    data_axis_size = _data_axis_size(mesh)

    observation_sharding = _map_sharding(
        batch.observation,
        lambda leaf: _query_sharding(
            leaf,
            groups=groups,
            queries=queries,
            mesh=mesh,
            data_axis_size=data_axis_size,
            name="observation",
        ),
    )
    action_sharding = _query_sharding(
        batch.actions,
        groups=groups,
        queries=queries,
        mesh=mesh,
        data_axis_size=data_axis_size,
        name="actions",
    )
    guide_sharding = _map_sharding(
        batch.guide,
        lambda leaf: _guide_sharding(
            leaf,
            groups=groups,
            mesh=mesh,
            name="guide",
        ),
    )

    return GuideConditionedBatch(
        observation=observation_sharding,
        actions=action_sharding,
        guide=guide_sharding,
    )


def put_guided_batch(
    batch: GuideConditionedBatch,
    sharding: GuideConditionedBatch,
) -> GuideConditionedBatch:
    """Place a grouped batch according to its explicit sharding pytree."""

    return jax.device_put(batch, sharding)
