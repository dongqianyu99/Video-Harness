from __future__ import annotations

from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.guide_resampler import PerceiverResampler
from openpi.models.guide_tokens import NUM_BOUNDARY_ROLES
from openpi.models.guide_tokens import NUM_MEMORY_KINDS
from openpi.models.guide_tokens import NUM_TRANSITION_ROLES
from openpi.models.guide_tokens import assemble_boundary_tokens
from openpi.models.guide_tokens import assemble_transition_tokens
from openpi.models.guide_tokens import pack_guide_tokens


@struct.dataclass
class GuideMemory:
    tokens: jax.Array  # [G, S, D]
    token_mask: jax.Array  # [G, S]


@struct.dataclass
class GuideBackboneFeatures:
    boundary_image_tokens: jax.Array  # [G, F, V, P, D]
    boundary_text_embeddings: jax.Array  # [G, F, V, T_B, D]
    transition_text_embeddings: jax.Array  # [G, U, T_T, D]


class GuideFeatureEncoder(nnx.Module):
    """Encode Boundary evidence and transition text into interleaved Guide memory."""

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 2048,
        *,
        boundary_num_queries: int = 8,
        transition_num_queries: int = 4,
        width: int = 1024,
        num_heads: int = 8,
        ffn_hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ):
        self.boundary_resampler = PerceiverResampler(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=boundary_num_queries,
            width=width,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            num_roles=NUM_BOUNDARY_ROLES,
            rngs=rngs,
        )
        self.transition_resampler = PerceiverResampler(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=transition_num_queries,
            width=width,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            num_roles=NUM_TRANSITION_ROLES,
            rngs=rngs,
        )
        self.memory_type_embeddings = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                rngs.params(),
                (NUM_MEMORY_KINDS, output_dim),
                jnp.float32,
            )
        )

    def __call__(
        self,
        boundary_image_tokens: jax.Array,
        boundary_image_mask: jax.Array,
        boundary_text_embeddings: jax.Array,
        boundary_text_mask: jax.Array,
        transition_text_embeddings: jax.Array,
        transition_text_mask: jax.Array,
        boundary_mask: jax.Array,
        unit_mask: jax.Array,
        memory_source_kind: jax.Array,
        memory_source_index: jax.Array,
        memory_source_offset: jax.Array,
        memory_mask: jax.Array,
    ) -> GuideMemory:
        boundary_tokens = assemble_boundary_tokens(
            boundary_image_tokens,
            boundary_image_mask,
            boundary_text_embeddings,
            boundary_text_mask,
            boundary_mask,
        )
        transition_tokens = assemble_transition_tokens(
            transition_text_embeddings,
            transition_text_mask,
            unit_mask,
        )
        boundary_memory = self.boundary_resampler(
            boundary_tokens.tokens,
            boundary_tokens.token_mask,
            boundary_tokens.role_ids,
            boundary_mask,
        )
        transition_memory = self.transition_resampler(
            transition_tokens.tokens,
            transition_tokens.token_mask,
            transition_tokens.role_ids,
            unit_mask,
        )
        packed = pack_guide_tokens(
            boundary_memory,
            transition_memory,
            boundary_mask,
            unit_mask,
            memory_source_kind,
            memory_source_index,
            memory_source_offset,
            memory_mask,
            self.memory_type_embeddings.value,
        )
        return GuideMemory(tokens=packed.tokens, token_mask=packed.token_mask)


def _validate_shared_feature_inputs(
    boundary_images: jax.Array,
    boundary_text_token_ids: jax.Array,
    transition_text_token_ids: jax.Array,
) -> tuple[int, int, int, int, int, int, int, int]:
    if boundary_images.ndim != 6:
        raise ValueError(
            "boundary_images must have shape [G, F, V, H, W, 3], got "
            f"{boundary_images.shape}"
        )
    if boundary_text_token_ids.ndim != 4:
        raise ValueError(
            "boundary_text_token_ids must have shape [G, F, V, T_B], got "
            f"{boundary_text_token_ids.shape}"
        )
    if transition_text_token_ids.ndim != 3:
        raise ValueError(
            "transition_text_token_ids must have shape [G, U, T_T], got "
            f"{transition_text_token_ids.shape}"
        )
    groups, boundaries, views, height, image_width, channels = boundary_images.shape
    text_groups, text_boundaries, text_views, boundary_text_tokens = (
        boundary_text_token_ids.shape
    )
    transition_groups, units, transition_text_tokens = transition_text_token_ids.shape
    if views != 3 or channels != 3:
        raise ValueError(
            f"boundary_images must contain exactly 3 RGB views, got {boundary_images.shape}"
        )
    if (text_groups, text_boundaries, text_views) != (groups, boundaries, views):
        raise ValueError("Boundary images and Boundary text must share G, F, and V")
    if transition_groups != groups:
        raise ValueError("Boundary and transition inputs must share G")
    if not jnp.issubdtype(boundary_images.dtype, jnp.floating):
        raise ValueError(
            f"boundary_images must have a floating dtype, got {boundary_images.dtype}"
        )
    if not jnp.issubdtype(boundary_text_token_ids.dtype, jnp.integer):
        raise ValueError("boundary_text_token_ids must have an integer dtype")
    if not jnp.issubdtype(transition_text_token_ids.dtype, jnp.integer):
        raise ValueError("transition_text_token_ids must have an integer dtype")
    return (
        groups,
        boundaries,
        views,
        height,
        image_width,
        units,
        boundary_text_tokens,
        transition_text_tokens,
    )


def encode_shared_guide_features(
    boundary_images: jax.Array,
    boundary_text_token_ids: jax.Array,
    transition_text_token_ids: jax.Array,
    *,
    image_encoder,
    text_embedder,
) -> GuideBackboneFeatures:
    """Encode three-view Boundary evidence and transition text with Pi0.5 backbones."""

    (
        groups,
        boundaries,
        views,
        height,
        image_width,
        units,
        boundary_text_tokens,
        transition_text_tokens,
    ) = _validate_shared_feature_inputs(
        boundary_images,
        boundary_text_token_ids,
        transition_text_token_ids,
    )
    flat_images = boundary_images.reshape(
        groups * boundaries * views,
        height,
        image_width,
        3,
    )
    flat_boundary_text = boundary_text_token_ids.reshape(
        groups * boundaries * views,
        boundary_text_tokens,
    )
    flat_transition_text = transition_text_token_ids.reshape(
        groups * units,
        transition_text_tokens,
    )
    flat_image_tokens, _ = image_encoder(flat_images, train=False)
    flat_boundary_text_embeddings = text_embedder(flat_boundary_text, method="embed")
    flat_transition_text_embeddings = text_embedder(flat_transition_text, method="embed")

    if flat_image_tokens.ndim != 3:
        raise ValueError(
            "image_encoder output must have shape [G*F*V, P, D], got "
            f"{flat_image_tokens.shape}"
        )
    image_batch, patches, image_dim = flat_image_tokens.shape
    if image_batch != groups * boundaries * views:
        raise ValueError("image_encoder output leading dimension must equal G*F*V")
    if flat_boundary_text_embeddings.shape[:2] != (
        groups * boundaries * views,
        boundary_text_tokens,
    ):
        raise ValueError("Boundary text embedding shape does not match token ids")
    if flat_transition_text_embeddings.shape[:2] != (
        groups * units,
        transition_text_tokens,
    ):
        raise ValueError("Transition text embedding shape does not match token ids")
    boundary_text_dim = flat_boundary_text_embeddings.shape[-1]
    transition_text_dim = flat_transition_text_embeddings.shape[-1]
    if boundary_text_dim != image_dim or transition_text_dim != image_dim:
        raise ValueError("Image, Boundary text, and transition text widths must match")

    return GuideBackboneFeatures(
        boundary_image_tokens=flat_image_tokens.reshape(
            groups,
            boundaries,
            views,
            patches,
            image_dim,
        ),
        boundary_text_embeddings=flat_boundary_text_embeddings.reshape(
            groups,
            boundaries,
            views,
            boundary_text_tokens,
            image_dim,
        ),
        transition_text_embeddings=flat_transition_text_embeddings.reshape(
            groups,
            units,
            transition_text_tokens,
            image_dim,
        ),
    )
