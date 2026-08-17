from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

BEFORE_ROLE = 0
TEXT_ROLE = 1
AFTER_ROLE = 2
NUM_ROLES = 3

@struct.dataclass
class UnitTokenBatch:
    """Assembled [Before, Text, After] tokens for every Guide unit."""

    tokens: jax.Array
    token_mask: jax.Array
    role_ids: jax.Array


def _validate_inputs(
    frame_tokens: jax.Array,
    frame_mask: jax.Array,
    text_embeddings: jax.Array,
    text_mask: jax.Array,
    unit_mask: jax.Array,
    before_slot: jax.Array,
    after_slot: jax.Array,
) -> tuple[int, int, int, int, int, int]:
    """Validate assembly inputs and return their static dimensions."""

    if frame_tokens.ndim != 4:
        raise ValueError(f"frame_tokens must have shape [G, F, P, D], got {frame_tokens.shape}")
    if text_embeddings.ndim != 4:
        raise ValueError(f"text_embeddings must have shape [G, U, T, D], got {text_embeddings.shape}")

    groups, frames, patches, width = frame_tokens.shape
    text_groups, units, text_tokens, text_width = text_embeddings.shape

    if text_groups != groups or text_width != width:
        raise ValueError(
            "text_embeddings must share G and D with frame_tokens: "
            f"expected G={groups}, D={width}, got {text_embeddings.shape}"
        )

    if not jnp.issubdtype(frame_tokens.dtype, jnp.floating):
        raise ValueError(f"frame_tokens must have a floating dtype, got {frame_tokens.dtype}")
    if not jnp.issubdtype(text_embeddings.dtype, jnp.floating):
        raise ValueError(f"text_embeddings must have a floating dtype, got {text_embeddings.dtype}")

    if frame_mask.shape != (groups, frames):
        raise ValueError(f"frame_mask must have shape {(groups, frames)}, got {frame_mask.shape}")
    if text_mask.shape != (groups, units, text_tokens):
        raise ValueError(f"text_mask must have shape {(groups, units, text_tokens)}, got {text_mask.shape}")
    if unit_mask.shape != (groups, units):
        raise ValueError(f"unit_mask must have shape {(groups, units)}, got {unit_mask.shape}")
    if before_slot.shape != (groups, units):
        raise ValueError(f"before_slot must have shape {(groups, units)}, got {before_slot.shape}")
    if after_slot.shape != (groups, units):
        raise ValueError(f"after_slot must have shape {(groups, units)}, got {after_slot.shape}")

    if frame_mask.dtype != jnp.bool_:
        raise ValueError(f"frame_mask must have bool dtype, got {frame_mask.dtype}")
    if text_mask.dtype != jnp.bool_:
        raise ValueError(f"text_mask must have bool dtype, got {text_mask.dtype}")
    if unit_mask.dtype != jnp.bool_:
        raise ValueError(f"unit_mask must have bool dtype, got {unit_mask.dtype}")

    for name, slot in (("before_slot", before_slot), ("after_slot", after_slot)):
        if not jnp.issubdtype(slot.dtype, jnp.integer):
            raise ValueError(f"{name} must have an integer dtype, got {slot.dtype}")

    # Concrete slot values can be checked eagerly. During jit tracing, values
    # are tracers and the static shape/dtype checks above still remain valid.
    if not any(isinstance(value, jax.core.Tracer) for value in (unit_mask, before_slot, after_slot)):
        valid_units = np.asarray(unit_mask)
        for name, slot in (("before_slot", before_slot), ("after_slot", after_slot)):
            slot_values = np.asarray(slot)
            invalid = valid_units & ((slot_values < 0) | (slot_values >= frames))
            if np.any(invalid):
                raise ValueError(f"{name} must be in [0, {frames}) for valid units")

    return groups, frames, patches, units, text_tokens, width


def _safe_slot(
    slot: jax.Array,
    unit_mask: jax.Array,
) -> jax.Array:
    """Replace slots of padding units with frame index zero."""

    return jnp.where(unit_mask, slot, jnp.zeros_like(slot))


def _gather_frame_segment(
    frame_tokens: jax.Array,
    frame_mask: jax.Array,
    slots: jax.Array,
    unit_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Gather one frame segment for every unit and apply its masks."""

    safe_slots = _safe_slot(slots, unit_mask)

    group_indices = jnp.arange(
        frame_tokens.shape[0],
        dtype=safe_slots.dtype,
    )[:, None]

    gathered_tokens = frame_tokens[group_indices, safe_slots]  # [G, U, P, D]
    gathered_frame_mask = frame_mask[group_indices, safe_slots]  # [G, U]

    segment_mask = jnp.broadcast_to(
        unit_mask[..., None] & gathered_frame_mask[..., None],  # [G, U, 1]
        gathered_tokens.shape[:-1],
    )  # [G, U, P]

    gathered_tokens = jnp.where(
        segment_mask[..., None],  # [G, U, P, 1]
        gathered_tokens,  # [G, U, P, D]
        jnp.zeros_like(gathered_tokens),
    )

    return gathered_tokens, segment_mask


def _mask_text_segment(
    text_embeddings: jax.Array,
    text_mask: jax.Array,
    unit_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply unit and token masks to text embeddings."""

    segment_mask = unit_mask[..., None] & text_mask

    text_embeddings = jnp.where(
        segment_mask[..., None],
        text_embeddings,
        jnp.zeros_like(text_embeddings),
    )

    return text_embeddings, segment_mask


def _make_role_ids(
    before_tokens: jax.Array,
    text_tokens: jax.Array,
    after_tokens: jax.Array,
) -> jax.Array:
    """Create role ids matching the [Before, Text, After] sequence."""

    before_roles = jnp.full(
        before_tokens.shape[:-1],  # [G, U, P]
        BEFORE_ROLE,
        dtype=jnp.int32,
    )
    text_roles = jnp.full(
        text_tokens.shape[:-1],  # [G, U, T]
        TEXT_ROLE,
        dtype=jnp.int32,
    )
    after_roles = jnp.full(
        after_tokens.shape[:-1],  # [G, U, P]
        AFTER_ROLE,
        dtype=jnp.int32,
    )

    return jnp.concatenate(
        [before_roles, text_roles, after_roles],
        axis=2,
    )


def _build_unit_token_batch(
    before_tokens: jax.Array,
    before_mask: jax.Array,
    text_tokens: jax.Array,
    text_mask: jax.Array,
    after_tokens: jax.Array,
    after_mask: jax.Array,
) -> UnitTokenBatch:
    """Concatenate the three masked segments into one UnitTokenBatch."""

    tokens = jnp.concatenate(
        [before_tokens, text_tokens, after_tokens],
        axis=2,
    )
    token_mask = jnp.concatenate(
        [before_mask, text_mask, after_mask],
        axis=2,
    )
    role_ids = _make_role_ids(
        before_tokens,
        text_tokens,
        after_tokens,
    )

    return UnitTokenBatch(
        tokens=tokens,
        token_mask=token_mask,
        role_ids=role_ids,
    )


def assemble_unit_tokens(
    frame_tokens: jax.Array,
    frame_mask: jax.Array,
    text_embeddings: jax.Array,
    text_mask: jax.Array,
    unit_mask: jax.Array,
    before_slot: jax.Array,
    after_slot: jax.Array,
) -> UnitTokenBatch:
    """Assemble [Before, Text, After] tokens for every guidance unit."""

    _validate_inputs(
        frame_tokens,
        frame_mask,
        text_embeddings,
        text_mask,
        unit_mask,
        before_slot,
        after_slot,
    )

    before_tokens, before_mask = _gather_frame_segment(
        frame_tokens,
        frame_mask,
        before_slot,
        unit_mask,
    )
    after_tokens, after_mask = _gather_frame_segment(
        frame_tokens,
        frame_mask,
        after_slot,
        unit_mask,
    )
    text_tokens, text_mask = _mask_text_segment(
        text_embeddings,
        text_mask,
        unit_mask,
    )

    return _build_unit_token_batch(
        before_tokens,
        before_mask,
        text_tokens,
        text_mask,
        after_tokens,
        after_mask,
    )
