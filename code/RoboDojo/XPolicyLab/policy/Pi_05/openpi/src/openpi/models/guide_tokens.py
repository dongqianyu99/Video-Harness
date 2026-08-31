from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

CAM_HIGH_IMAGE_ROLE = 0
CAM_HIGH_TEXT_ROLE = 1
CAM_LEFT_WRIST_IMAGE_ROLE = 2
CAM_LEFT_WRIST_TEXT_ROLE = 3
CAM_RIGHT_WRIST_IMAGE_ROLE = 4
CAM_RIGHT_WRIST_TEXT_ROLE = 5
NUM_BOUNDARY_ROLES = 6

TRANSITION_TEXT_ROLE = 0
NUM_TRANSITION_ROLES = 1

BOUNDARY_MEMORY_KIND = 0
TRANSITION_MEMORY_KIND = 1
NUM_MEMORY_KINDS = 2


@struct.dataclass
class ResamplerTokenBatch:
    tokens: jax.Array
    token_mask: jax.Array
    role_ids: jax.Array


@struct.dataclass
class PackedGuideTokens:
    tokens: jax.Array
    token_mask: jax.Array


def validate_materialized_guide_map(
    guide,
    *,
    boundary_num_queries: int,
    transition_num_queries: int,
) -> None:
    """Eagerly validate a host-materialized Guide map before any JIT gather."""

    for name, value in (
        ("boundary_num_queries", boundary_num_queries),
        ("transition_num_queries", transition_num_queries),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    boundary_mask = np.asarray(guide.boundary_mask)
    unit_mask = np.asarray(guide.unit_mask)
    source_kind = np.asarray(guide.memory_source_kind)
    source_index = np.asarray(guide.memory_source_index)
    source_offset = np.asarray(guide.memory_source_offset)
    memory_mask = np.asarray(guide.memory_mask)
    if boundary_mask.ndim != 2 or unit_mask.ndim != 2:
        raise ValueError("Guide boundary/unit masks must have shapes [G,F] and [G,U]")
    if boundary_mask.dtype != np.bool_ or unit_mask.dtype != np.bool_:
        raise ValueError("Guide boundary/unit masks must have bool dtype")
    if memory_mask.ndim != 2:
        raise ValueError("Guide memory_mask must have shape [G,S]")
    expected_map_shape = (boundary_mask.shape[0], memory_mask.shape[1])
    if any(
        value.shape != expected_map_shape
        for value in (source_kind, source_index, source_offset)
    ):
        raise ValueError("Guide memory map arrays must share shape [G,S]")
    if memory_mask.dtype != np.bool_:
        raise ValueError("Guide memory_mask must have bool dtype")
    if any(
        not np.issubdtype(value.dtype, np.integer)
        for value in (source_kind, source_index, source_offset)
    ):
        raise ValueError("Guide memory map values must have integer dtype")
    if np.any(memory_mask[:, 1:] & ~memory_mask[:, :-1]):
        raise ValueError("Guide memory_mask must contain only tail padding")

    for group in range(boundary_mask.shape[0]):
        expected = {
            (BOUNDARY_MEMORY_KIND, index, offset)
            for index in np.flatnonzero(boundary_mask[group])
            for offset in range(boundary_num_queries)
        }
        expected.update(
            (TRANSITION_MEMORY_KIND, index, offset)
            for index in np.flatnonzero(unit_mask[group])
            for offset in range(transition_num_queries)
        )
        actual = [
            (int(kind), int(index), int(offset))
            for kind, index, offset in zip(
                source_kind[group, memory_mask[group]],
                source_index[group, memory_mask[group]],
                source_offset[group, memory_mask[group]],
                strict=True,
            )
        ]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(
                "Guide memory map must cover every valid Boundary/Transition token exactly once"
            )


def _require_bool(value: jax.Array, name: str) -> None:
    if value.dtype != jnp.bool_:
        raise ValueError(f"{name} must have bool dtype, got {value.dtype}")


def _require_integer(value: jax.Array, name: str) -> None:
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise ValueError(f"{name} must have an integer dtype, got {value.dtype}")


def assemble_boundary_tokens(
    image_tokens: jax.Array,
    image_mask: jax.Array,
    text_embeddings: jax.Array,
    text_mask: jax.Array,
    boundary_mask: jax.Array,
) -> ResamplerTokenBatch:
    """Assemble three image/text view pairs for each Boundary."""

    if image_tokens.ndim != 5:
        raise ValueError(
            "image_tokens must have shape [G, F, V, P, D], got "
            f"{image_tokens.shape}"
        )
    if text_embeddings.ndim != 5:
        raise ValueError(
            "text_embeddings must have shape [G, F, V, T, D], got "
            f"{text_embeddings.shape}"
        )
    groups, boundaries, views, patches, width = image_tokens.shape
    text_groups, text_boundaries, text_views, text_tokens, text_width = (
        text_embeddings.shape
    )
    if views != 3:
        raise ValueError(f"Boundary image tokens require exactly 3 views, got {views}")
    if (text_groups, text_boundaries, text_views, text_width) != (
        groups,
        boundaries,
        views,
        width,
    ):
        raise ValueError(
            "Boundary image/text features must share G, F, V, and D: "
            f"got {image_tokens.shape} and {text_embeddings.shape}"
        )
    if image_mask.shape != (groups, boundaries, views):
        raise ValueError(
            f"image_mask must have shape {(groups, boundaries, views)}, got "
            f"{image_mask.shape}"
        )
    if text_mask.shape != (groups, boundaries, views, text_tokens):
        raise ValueError(
            "text_mask must have shape "
            f"{(groups, boundaries, views, text_tokens)}, got {text_mask.shape}"
        )
    if boundary_mask.shape != (groups, boundaries):
        raise ValueError(
            f"boundary_mask must have shape {(groups, boundaries)}, got "
            f"{boundary_mask.shape}"
        )
    if not jnp.issubdtype(image_tokens.dtype, jnp.floating):
        raise ValueError(f"image_tokens must have a floating dtype, got {image_tokens.dtype}")
    if not jnp.issubdtype(text_embeddings.dtype, jnp.floating):
        raise ValueError(
            f"text_embeddings must have a floating dtype, got {text_embeddings.dtype}"
        )
    _require_bool(image_mask, "image_mask")
    _require_bool(text_mask, "text_mask")
    _require_bool(boundary_mask, "boundary_mask")

    if not any(
        isinstance(value, jax.core.Tracer)
        for value in (image_mask, text_mask, boundary_mask)
    ):
        valid = np.asarray(boundary_mask)
        valid_images = np.asarray(image_mask)
        valid_text = np.asarray(text_mask).any(axis=-1)
        if np.any(valid[..., None] & ~valid_images):
            raise ValueError("Every valid Boundary must provide all three image views")
        if np.any(valid[..., None] & ~valid_text):
            raise ValueError("Every valid Boundary must provide text for all three views")

    segments: list[jax.Array] = []
    masks: list[jax.Array] = []
    roles: list[jax.Array] = []
    role_pairs = (
        (CAM_HIGH_IMAGE_ROLE, CAM_HIGH_TEXT_ROLE),
        (CAM_LEFT_WRIST_IMAGE_ROLE, CAM_LEFT_WRIST_TEXT_ROLE),
        (CAM_RIGHT_WRIST_IMAGE_ROLE, CAM_RIGHT_WRIST_TEXT_ROLE),
    )
    for view, (image_role, text_role) in enumerate(role_pairs):
        view_image_mask = boundary_mask[..., None] & image_mask[..., view, None]
        view_image_mask = jnp.broadcast_to(
            view_image_mask,
            (groups, boundaries, patches),
        )
        view_text_mask = boundary_mask[..., None] & text_mask[..., view, :]
        view_images = jnp.where(
            view_image_mask[..., None],
            image_tokens[..., view, :, :],
            jnp.zeros_like(image_tokens[..., view, :, :]),
        )
        view_text = jnp.where(
            view_text_mask[..., None],
            text_embeddings[..., view, :, :],
            jnp.zeros_like(text_embeddings[..., view, :, :]),
        )
        segments.extend((view_images, view_text))
        masks.extend((view_image_mask, view_text_mask))
        roles.extend(
            (
                jnp.full(view_image_mask.shape, image_role, dtype=jnp.int32),
                jnp.full(view_text_mask.shape, text_role, dtype=jnp.int32),
            )
        )

    return ResamplerTokenBatch(
        tokens=jnp.concatenate(segments, axis=2),
        token_mask=jnp.concatenate(masks, axis=2),
        role_ids=jnp.concatenate(roles, axis=2),
    )


def assemble_transition_tokens(
    text_embeddings: jax.Array,
    text_mask: jax.Array,
    unit_mask: jax.Array,
) -> ResamplerTokenBatch:
    """Mask transition text without mixing Boundary memory into the resampler."""

    if text_embeddings.ndim != 4:
        raise ValueError(
            "text_embeddings must have shape [G, U, T, D], got "
            f"{text_embeddings.shape}"
        )
    groups, units, tokens, _ = text_embeddings.shape
    if text_mask.shape != (groups, units, tokens):
        raise ValueError(
            f"text_mask must have shape {(groups, units, tokens)}, got {text_mask.shape}"
        )
    if unit_mask.shape != (groups, units):
        raise ValueError(
            f"unit_mask must have shape {(groups, units)}, got {unit_mask.shape}"
        )
    if not jnp.issubdtype(text_embeddings.dtype, jnp.floating):
        raise ValueError(
            f"text_embeddings must have a floating dtype, got {text_embeddings.dtype}"
        )
    _require_bool(text_mask, "text_mask")
    _require_bool(unit_mask, "unit_mask")

    combined_mask = unit_mask[..., None] & text_mask
    if not any(
        isinstance(value, jax.core.Tracer) for value in (text_mask, unit_mask)
    ) and np.any(np.asarray(unit_mask) & ~np.asarray(text_mask).any(axis=-1)):
        raise ValueError("Every valid transition must provide text")
    return ResamplerTokenBatch(
        tokens=jnp.where(
            combined_mask[..., None],
            text_embeddings,
            jnp.zeros_like(text_embeddings),
        ),
        token_mask=combined_mask,
        role_ids=jnp.full(combined_mask.shape, TRANSITION_TEXT_ROLE, dtype=jnp.int32),
    )


def _validate_memory_map(
    boundary_memory: jax.Array,
    transition_memory: jax.Array,
    boundary_mask: jax.Array,
    unit_mask: jax.Array,
    source_kind: jax.Array,
    source_index: jax.Array,
    source_offset: jax.Array,
    memory_mask: jax.Array,
) -> tuple[int, int, int, int, int, int]:
    if boundary_memory.ndim != 4:
        raise ValueError(
            "boundary_memory must have shape [G, F, K_B, D], got "
            f"{boundary_memory.shape}"
        )
    if transition_memory.ndim != 4:
        raise ValueError(
            "transition_memory must have shape [G, U, K_T, D], got "
            f"{transition_memory.shape}"
        )
    groups, boundaries, boundary_tokens, width = boundary_memory.shape
    transition_groups, units, transition_tokens, transition_width = (
        transition_memory.shape
    )
    if transition_groups != groups or transition_width != width:
        raise ValueError("Boundary and transition memory must share G and D")
    if boundary_mask.shape != (groups, boundaries):
        raise ValueError(
            f"boundary_mask must have shape {(groups, boundaries)}, got "
            f"{boundary_mask.shape}"
        )
    if unit_mask.shape != (groups, units):
        raise ValueError(
            f"unit_mask must have shape {(groups, units)}, got {unit_mask.shape}"
        )
    if source_kind.ndim != 2:
        raise ValueError(
            f"memory_source_kind must have shape [G, S], got {source_kind.shape}"
        )
    sequence = source_kind.shape[1]
    expected_map_shape = (groups, sequence)
    for name, value in (
        ("memory_source_index", source_index),
        ("memory_source_offset", source_offset),
        ("memory_mask", memory_mask),
    ):
        if value.shape != expected_map_shape:
            raise ValueError(
                f"{name} must have shape {expected_map_shape}, got {value.shape}"
            )
    _require_bool(boundary_mask, "boundary_mask")
    _require_bool(unit_mask, "unit_mask")
    _require_bool(memory_mask, "memory_mask")
    _require_integer(source_kind, "memory_source_kind")
    _require_integer(source_index, "memory_source_index")
    _require_integer(source_offset, "memory_source_offset")
    return groups, boundaries, units, boundary_tokens, transition_tokens, width


def pack_guide_tokens(
    boundary_memory: jax.Array,
    transition_memory: jax.Array,
    boundary_mask: jax.Array,
    unit_mask: jax.Array,
    source_kind: jax.Array,
    source_index: jax.Array,
    source_offset: jax.Array,
    memory_mask: jax.Array,
    memory_type_embeddings: jax.Array,
) -> PackedGuideTokens:
    """Pack Boundary and transition memory in host-provided canonical order."""

    (
        groups,
        _,
        _,
        boundary_tokens,
        transition_tokens,
        width,
    ) = _validate_memory_map(
        boundary_memory,
        transition_memory,
        boundary_mask,
        unit_mask,
        source_kind,
        source_index,
        source_offset,
        memory_mask,
    )
    if memory_type_embeddings.shape != (NUM_MEMORY_KINDS, width):
        raise ValueError(
            "memory_type_embeddings must have shape "
            f"{(NUM_MEMORY_KINDS, width)}, got {memory_type_embeddings.shape}"
        )

    if not any(
        isinstance(value, jax.core.Tracer)
        for value in (
            boundary_mask,
            unit_mask,
            source_kind,
            source_index,
            source_offset,
            memory_mask,
        )
    ):
        kinds = np.asarray(source_kind)
        indices = np.asarray(source_index)
        offsets = np.asarray(source_offset)
        valid = np.asarray(memory_mask)
        boundary_valid = np.asarray(boundary_mask)
        transition_valid = np.asarray(unit_mask)
        if np.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("memory_mask must contain valid tokens followed only by tail padding")
        for group in range(groups):
            expected = {
                (BOUNDARY_MEMORY_KIND, index, offset)
                for index in np.flatnonzero(boundary_valid[group])
                for offset in range(boundary_tokens)
            }
            expected.update(
                (TRANSITION_MEMORY_KIND, index, offset)
                for index in np.flatnonzero(transition_valid[group])
                for offset in range(transition_tokens)
            )
            actual = [
                (int(kind), int(index), int(offset))
                for kind, index, offset in zip(
                    kinds[group, valid[group]],
                    indices[group, valid[group]],
                    offsets[group, valid[group]],
                    strict=True,
                )
            ]
            if len(actual) != len(set(actual)):
                raise ValueError("Guide memory map contains duplicate source tokens")
            if set(actual) != expected:
                raise ValueError("Guide memory map must cover every valid source token exactly once")

    is_boundary = memory_mask & (source_kind == BOUNDARY_MEMORY_KIND)
    is_transition = memory_mask & (source_kind == TRANSITION_MEMORY_KIND)
    safe_boundary_index = jnp.where(is_boundary, source_index, 0)
    safe_boundary_offset = jnp.where(is_boundary, source_offset, 0)
    safe_transition_index = jnp.where(is_transition, source_index, 0)
    safe_transition_offset = jnp.where(is_transition, source_offset, 0)
    group_index = jnp.arange(groups, dtype=source_index.dtype)[:, None]

    boundary_tokens_value = boundary_memory[
        group_index,
        safe_boundary_index,
        safe_boundary_offset,
    ]
    transition_tokens_value = transition_memory[
        group_index,
        safe_transition_index,
        safe_transition_offset,
    ]
    safe_kind = jnp.where(memory_mask, source_kind, 0)
    tokens = jnp.where(
        is_boundary[..., None],
        boundary_tokens_value,
        transition_tokens_value,
    )
    tokens = tokens + memory_type_embeddings[safe_kind]
    tokens = jnp.where(memory_mask[..., None], tokens, jnp.zeros_like(tokens))
    return PackedGuideTokens(tokens=tokens, token_mask=memory_mask)
