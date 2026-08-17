from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models import pi0 as _pi0


def _make_block_ar_mask(token_counts: int) -> jax.Array:
    """
    Create [True, False, ...] for one attention block.

    e.g.
        _make_block_ar_mask(3) = [True, False, False]
    """
    return jnp.arange(token_counts, dtype=jnp.int32) == 0


def make_gca_ar_mask(
    *,
    guide_tokens: int,
    control_tokens: int,
    action_tokens: int,
) -> jax.Array:
    """Create the block boundaries for the [Guide, Control, Action] sequence."""
    return jnp.concatenate(
        [
            jnp.zeros((guide_tokens,), dtype=jnp.bool_),
            _make_block_ar_mask(control_tokens),
            _make_block_ar_mask(action_tokens),
        ]
    )


def make_gca_attn_mask(
    guide_mask: jax.Array,
    control_mask: jax.Array,
    action_mask: jax.Array,
) -> jax.Array:
    """Create the [Guide, Control, Action] attention mask."""
    input_mask = jnp.concatenate([guide_mask, control_mask, action_mask], axis=1)
    ar_mask = make_gca_ar_mask(
        guide_tokens=guide_mask.shape[1],
        control_tokens=control_mask.shape[1],
        action_tokens=action_mask.shape[1],
    )
    return _pi0.make_attn_mask(input_mask, ar_mask)


def make_gc_ar_mask(
    *,
    guide_tokens: int,
    control_tokens: int,
) -> jax.Array:
    """Create the block boundaries for the [Guide, Control] sequence."""

    return jnp.concatenate(
        [
            jnp.zeros((guide_tokens,), dtype=jnp.bool_),
            _make_block_ar_mask(control_tokens),
        ]
    )


def make_gc_attn_mask(
    guide_mask: jax.Array,
    control_mask: jax.Array,
) -> jax.Array:
    """Create the [Guide, Control] attention mask."""

    input_mask = jnp.concatenate([guide_mask, control_mask], axis=1)

    ar_mask = make_gc_ar_mask(
        guide_tokens=guide_mask.shape[1],
        control_tokens=control_mask.shape[1],
    )

    return _pi0.make_attn_mask(input_mask, ar_mask)
