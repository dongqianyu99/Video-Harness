from __future__ import annotations

import hashlib

from flax import struct
import jax
import jax.numpy as jnp

from openpi.models import model as _model

GUIDE_REPRESENTATION_DIGEST = hashlib.sha256(

        b"boundary=[cam_high_image,cam_high_text,cam_left_wrist_image,"
        b"cam_left_wrist_text,cam_right_wrist_image,cam_right_wrist_text];"
        b"transition=text_only;memory=interleaved_boundary_transition;"
        b"causal_reason=host_only"

).hexdigest()


@struct.dataclass
class GuideInput:
    """Model-facing Behavior Document tensors for one grouped Guide bank."""

    boundary_images: jax.Array
    boundary_image_mask: jax.Array
    boundary_text_tokens: jax.Array
    boundary_text_mask: jax.Array
    transition_text_tokens: jax.Array
    transition_text_mask: jax.Array
    boundary_mask: jax.Array
    unit_mask: jax.Array
    memory_source_kind: jax.Array
    memory_source_index: jax.Array
    memory_source_offset: jax.Array
    memory_mask: jax.Array


@struct.dataclass
class GuideConditionedBatch:
    """Grouped Guide Batch before flattening into the stock Pi0 path."""

    observation: _model.Observation
    actions: _model.Actions
    guide: GuideInput
    query_mask: jax.Array | None = None


def _format_tree_path(path) -> str:
    """Format a JAX pytree path for validation errors."""

    parts = []

    for entry in path:
        if hasattr(entry, "name"):
            parts.append(f".{entry.name}")
        elif hasattr(entry, "key"):
            parts.append(f"[{entry.key!r}]")
        elif hasattr(entry, "idx"):
            parts.append(f"[{entry.idx}]")

    return "".join(parts) or "<root>"


def _validate_tree_leading_dims(
    tree,
    expected_leading_dims: tuple[int, ...],
    *,
    name: str,
) -> None:
    """Validate leading dimensions for every non-None pytree leaf."""

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)

    for path, leaf in leaves_with_paths:
        if leaf is None:
            continue

        if not hasattr(leaf, "shape"):
            raise ValueError(f"{name}{_format_tree_path(path)} must be an array leaf")

        shape = tuple(leaf.shape)
        expected_rank = len(expected_leading_dims)

        if len(shape) < expected_rank or shape[:expected_rank] != expected_leading_dims:
            raise ValueError(
                f"{name}{_format_tree_path(path)} must have leading dimensions {expected_leading_dims}, got {shape}"
            )


def validate_guide_conditioned_batch(
    batch: GuideConditionedBatch,
) -> tuple[int, int]:
    """Validate grouped Guide/control shapes and return G and Q."""

    if batch.actions.ndim != 4:
        raise ValueError(f"actions must have shape [G, Q, AH, AD], got {batch.actions.shape}")

    groups, queries = batch.actions.shape[:2]

    if groups <= 0:
        raise ValueError(f"group dimension G must be positive, got {groups}")

    if queries <= 0:
        raise ValueError(f"query dimension Q must be positive, got {queries}")

    _validate_tree_leading_dims(
        batch.observation,
        (groups, queries),
        name="observation",
    )

    _validate_tree_leading_dims(
        batch.guide,
        (groups,),
        name="guide",
    )

    if batch.query_mask is not None:
        if batch.query_mask.shape != (groups, queries):
            raise ValueError(
                "query_mask must have shape [G, Q], got "
                f"{batch.query_mask.shape} for G={groups}, Q={queries}"
            )
        if batch.query_mask.dtype != jnp.bool_:
            raise ValueError(
                f"query_mask must have boolean dtype, got {batch.query_mask.dtype}"
            )
    return groups, queries


def query_mask_or_ones(batch: GuideConditionedBatch) -> jax.Array:
    groups, queries = batch.actions.shape[:2]
    if batch.query_mask is None:
        return jnp.ones((groups, queries), dtype=jnp.bool_)
    return batch.query_mask


def validate_guide_conditioned_observation(
    observation: _model.Observation,
    guide: GuideInput,
) -> tuple[int, int]:
    """Validate grouped Guide/inference inputs and return G and Q."""

    if observation.state.ndim < 3:
        raise ValueError(f"observation.state must have shape [G, Q, S], got {observation.state.shape}")

    groups, queries = observation.state.shape[:2]

    if groups <= 0:
        raise ValueError(f"group dimension G must be positive, got {groups}")

    if queries <= 0:
        raise ValueError(f"query dimension Q must be positive, got {queries}")

    _validate_tree_leading_dims(
        observation,
        (groups, queries),
        name="observation",
    )

    _validate_tree_leading_dims(
        guide,
        (groups,),
        name="guide",
    )

    return groups, queries


def _flatten_group_query(value: jax.Array) -> jax.Array:
    """Flatten [G, Q, ...] into [G * Q, ...] in group-major order."""
    return value.reshape((value.shape[0] * value.shape[1], *value.shape[2:]))


def flatten_grouped_observation(
    observation: _model.Observation,
) -> _model.Observation:
    """Flatten [G, Q, ...] observation leaves into [G * Q, ...]."""

    return jax.tree_util.tree_map(_flatten_group_query, observation)


def flatten_grouped_control(
    observation: _model.Observation,
    actions: _model.Actions,
) -> tuple[_model.Observation, _model.Actions]:
    """Flatten grouped control inputs before calling the stock Pi0 path."""

    flat_observation = flatten_grouped_observation(observation)
    flat_actions = _flatten_group_query(actions)
    return flat_observation, flat_actions


def broadcast_guide_memory(
    guide_memory: jax.Array,
    *,
    queries_per_guide: int,
) -> jax.Array:
    """Broadcast Guide tokens [G, S, D] or masks [G, S] along Q."""

    if guide_memory.ndim not in (2, 3):
        raise ValueError(f"guide_memory must have rank 2 [G, S] or rank 3 [G, S, D], got shape {guide_memory.shape}")

    if queries_per_guide <= 0:
        raise ValueError(f"queries_per_guide must be positive, got {queries_per_guide}")

    return jnp.repeat(guide_memory, queries_per_guide, axis=0)
