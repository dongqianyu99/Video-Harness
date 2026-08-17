from __future__ import annotations

from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.guide_resampler import UnitResampler
from openpi.models.guide_tokens import assemble_unit_tokens


@struct.dataclass
class GuideMemory:
    """Flattened Guide memory for one grouped Guide bank."""

    tokens: jax.Array  # [G, U * K, D]
    token_mask: jax.Array  # [G, U * K]


@struct.dataclass
class GuideBackboneFeatures:
    """shared-backbone features for Guide images and text."""

    frame_tokens: jax.Array  # [G, F, P, D]
    text_embeddings: jax.Array  # [G, U, T, D]


def _validate_inputs(
    unit_memory: jax.Array,
    unit_mask: jax.Array,
) -> tuple[int, int, int, int]:
    """Validate unit memory inputs and return G, U, K, and D."""

    if unit_memory.ndim != 4:
        raise ValueError(f"unit_memory must have shape [G, U, K, D], got {unit_memory.shape}")

    if unit_mask.ndim != 2:
        raise ValueError(f"unit_mask must have shape [G, U], got {unit_mask.shape}")

    groups, units, queries, width = unit_memory.shape

    if unit_mask.shape != (groups, units):
        raise ValueError(f"unit_mask must have shape {(groups, units)}, got {unit_mask.shape}")

    if not jnp.issubdtype(unit_memory.dtype, jnp.floating):
        raise ValueError(f"unit_memory must have a floating dtype, got {unit_memory.dtype}")

    if unit_mask.dtype != jnp.bool_:
        raise ValueError(f"unit_mask must have bool dtype, got {unit_mask.dtype}")

    return groups, units, queries, width


def _flatten_memory_and_mask(
    unit_memory: jax.Array,
    unit_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Flatten unit/query axes and apply the repeated unit masks."""

    groups, units, queries, width = unit_memory.shape

    tokens = unit_memory.reshape(
        groups,
        units * queries,
        width,
    )

    token_mask = jnp.repeat(
        unit_mask,
        queries,
        axis=1,
    )

    tokens = jnp.where(
        token_mask[..., None],
        tokens,
        jnp.zeros_like(tokens),
    )

    return tokens, token_mask


def flatten_unit_memory(
    unit_memory: jax.Array,
    unit_mask: jax.Array,
) -> GuideMemory:
    """Flatten per-unit memory into ordered GuideMemory tokens."""

    _validate_inputs(
        unit_memory,
        unit_mask,
    )

    tokens, token_mask = _flatten_memory_and_mask(
        unit_memory,
        unit_mask,
    )

    return GuideMemory(
        tokens=tokens,
        token_mask=token_mask,
    )


class GuideFeatureEncoder(nnx.Module):
    """Encode pre-embedded Guide features with a shared per-unit resampler."""

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 2048,
        *,
        num_queries: int = 8,
        width: int = 1024,
        num_heads: int = 8,
        ffn_hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ):
        self.unit_resampler = UnitResampler(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=num_queries,
            width=width,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            rngs=rngs,
        )


    def __call__(
        self,
        frame_tokens: jax.Array,
        frame_mask: jax.Array,
        text_embeddings: jax.Array,
        text_mask: jax.Array,
        unit_mask: jax.Array,
        before_slot: jax.Array,
        after_slot: jax.Array,
    ) -> GuideMemory:
        _unit_tokens = assemble_unit_tokens(
            frame_tokens,  # [G, F, P, D_in]
            frame_mask,
            text_embeddings,  # [G, U, T, D_in]
            text_mask,
            unit_mask,
            before_slot,
            after_slot,
        )

        _unit_memory = self.unit_resampler(
            _unit_tokens.tokens,  # [G, U, 2P+T, D_in]
            _unit_tokens.token_mask,
            _unit_tokens.role_ids,
            unit_mask,
        )  # [G, U, K, D_out]

        return flatten_unit_memory(_unit_memory, unit_mask)


def _validate_shared_feature_inputs(
    images: jax.Array,
    text_token_ids: jax.Array,
) -> tuple[int, int, int, int, int, int]:
    """Validate Guide image and text inputs before shared encoding."""

    if images.ndim != 5:
        raise ValueError(f"images must have shape [G, F, H, W, 3], got {images.shape}")

    if text_token_ids.ndim != 3:
        raise ValueError(f"text_token_ids must have shape [G, U, T], got {text_token_ids.shape}")

    groups, frames, height, image_width, channels = images.shape
    text_groups, units, text_tokens = text_token_ids.shape

    if channels != 3:
        raise ValueError(f"images must have 3 channels, got {channels}")

    if text_groups != groups:
        raise ValueError(
            "images and text_token_ids must have the same group count: "
            f"images G={groups}, text_token_ids G={text_groups}"
        )

    if not jnp.issubdtype(images.dtype, jnp.floating):
        raise ValueError(f"images must have a floating dtype, got {images.dtype}")

    if not jnp.issubdtype(text_token_ids.dtype, jnp.integer):
        raise ValueError(f"text_token_ids must have an integer dtype, got {text_token_ids.dtype}")

    return groups, frames, height, image_width, units, text_tokens


def encode_shared_guide_features(
    images: jax.Array,
    text_token_ids: jax.Array,
    *,
    image_encoder,
    text_embedder,
) -> GuideBackboneFeatures:
    """Encode Guide images and text with externally owned shared backbones."""

    groups, frames, height, image_width, units, text_tokens = (
        _validate_shared_feature_inputs(images, text_token_ids)
    )

    flat_images = images.reshape(groups * frames, height, image_width, 3)

    flat_text_token_ids = text_token_ids.reshape(groups * units, text_tokens)

    _flat_frame_tokens, _ = image_encoder(flat_images, train=False)

    _flat_text_embeddings = text_embedder(flat_text_token_ids, method="embed")

    if _flat_frame_tokens.ndim != 3:
        raise ValueError(f"image_encoder output must have shape [G*F, P, D], got {_flat_frame_tokens.shape}")

    expected_frame_batch = groups * frames
    frame_batch, patches, image_width = _flat_frame_tokens.shape

    if frame_batch != expected_frame_batch:
        raise ValueError(
            f"image_encoder output leading dimension must equal G*F: expected {expected_frame_batch}, got {frame_batch}"
        )

    if _flat_text_embeddings.ndim != 3:
        raise ValueError(f"text_embedder output must have shape [G*U, T, D], got {_flat_text_embeddings.shape}")

    expected_text_batch = groups * units
    text_batch, output_text_tokens, text_width = _flat_text_embeddings.shape

    if text_batch != expected_text_batch:
        raise ValueError(
            f"text_embedder output leading dimension must equal G*U: expected {expected_text_batch}, got {text_batch}"
        )

    if output_text_tokens != text_tokens:
        raise ValueError(
            f"text_embedder output token length must match input T: expected {text_tokens}, got {output_text_tokens}"
        )

    if text_width != image_width:
        raise ValueError(
            "image and text encoder outputs must have the same width: "
            f"image width={image_width}, text width={text_width}"
        )

    frame_tokens = _flat_frame_tokens.reshape(groups, frames, patches, image_width)
    text_embeddings = _flat_text_embeddings.reshape(groups, units, text_tokens, text_width)

    return GuideBackboneFeatures(
        frame_tokens=frame_tokens,  # [G, F, P, D]
        text_embeddings=text_embeddings,  # [G, U, T, D]
    )
