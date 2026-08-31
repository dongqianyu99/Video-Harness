from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def _local_sinusoidal_position_encoding(length: int, width: int, dtype: jnp.dtype) -> jax.Array:
    positions = jnp.arange(length, dtype=jnp.float32)[:, None]  # [L] -> [L, 1]
    frequency_indices = jnp.arange(0, width, 2, dtype=jnp.float32)  # [width // 2, ]
    inverse_frequencies = jnp.exp(-math.log(10_000.0) * frequency_indices / width)  # [width // 2, ]
    angles = positions * inverse_frequencies[None, :]  # [L, width // 2]
    encoding = jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(length, -1)
    return encoding[:, :width].astype(dtype)


class PerceiverResampler(nnx.Module):
    """Shared Perceiver-style resampler applied independently over one bank axis."""

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 2048,
        *,
        num_queries: int = 8,
        width: int = 1024,
        num_heads: int = 8,
        ffn_hidden_dim: int | None = None,
        num_roles: int,
        rngs: nnx.Rngs,
    ):
        if (
            input_dim <= 0
            or output_dim <= 0
            or num_queries <= 0
            or width <= 0
            or num_heads <= 0
            or num_roles <= 0
        ):
            raise ValueError(
                "input_dim, output_dim, num_queries, width, num_heads, and num_roles "
                "must all be positive"
            )
        if width % num_heads != 0:
            raise ValueError(f"width ({width}) must be divisible by num_heads ({num_heads})")
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * width
        if ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.num_roles = num_roles

        query_init = nnx.initializers.normal(stddev=0.02)
        self.learned_queries = nnx.Param(
            query_init(rngs.params(), (num_queries, width), jnp.float32)
        )
        self.role_embeddings = nnx.Param(
            query_init(rngs.params(), (num_roles, width), jnp.float32)
        )

        kernel_init = nnx.initializers.xavier_uniform()
        self.input_projection = nnx.Linear(input_dim, width, kernel_init=kernel_init, rngs=rngs)
        self.query_norm = nnx.LayerNorm(width, rngs=rngs)
        self.token_norm = nnx.LayerNorm(width, rngs=rngs)
        self.query_projection = nnx.Linear(
            width,
            width,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.key_projection = nnx.Linear(
            width,
            width,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.value_projection = nnx.Linear(
            width,
            width,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.attention_output_projection = nnx.Linear(
            width,
            width,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.ffn_norm = nnx.LayerNorm(width, rngs=rngs)
        self.ffn_in = nnx.Linear(width, ffn_hidden_dim, kernel_init=kernel_init, rngs=rngs)
        self.ffn_out = nnx.Linear(ffn_hidden_dim, width, kernel_init=kernel_init, rngs=rngs)

        self.output_projection = nnx.Linear(
            width,
            output_dim,
            kernel_init=nnx.initializers.normal(stddev=1e-3),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )


    def _validate_inputs(
        self,
        bank_tokens: jax.Array,
        token_mask: jax.Array,
        role_ids: jax.Array,
        item_mask: jax.Array,
    ) -> None:
        if bank_tokens.ndim != 4:
            raise ValueError(
                "bank_tokens must have shape [G, N, L, D_in], got "
                f"{bank_tokens.shape}"
            )

        groups, items, tokens, input_dim = bank_tokens.shape
        if input_dim != self.input_dim:
            raise ValueError(
                f"bank_tokens last dimension must be input_dim={self.input_dim}, "
                f"got {input_dim}"
            )
        if token_mask.shape != (groups, items, tokens):
            raise ValueError(
                f"token_mask must have shape {(groups, items, tokens)}, got "
                f"{token_mask.shape}"
            )
        if role_ids.shape != (groups, items, tokens):
            raise ValueError(
                f"role_ids must have shape {(groups, items, tokens)}, got {role_ids.shape}"
            )
        if item_mask.shape != (groups, items):
            raise ValueError(
                f"item_mask must have shape {(groups, items)}, got {item_mask.shape}"
            )
        if token_mask.dtype != jnp.bool_:
            raise ValueError(f"token_mask must have bool dtype, got {token_mask.dtype}")
        if item_mask.dtype != jnp.bool_:
            raise ValueError(f"item_mask must have bool dtype, got {item_mask.dtype}")
        if not jnp.issubdtype(role_ids.dtype, jnp.integer):
            raise ValueError(f"role_ids must have an integer dtype, got {role_ids.dtype}")

        if not isinstance(role_ids, jax.core.Tracer):
            role_ids_array = np.asarray(role_ids)
            if np.any((role_ids_array < 0) | (role_ids_array >= self.num_roles)):
                raise ValueError(
                    f"role_ids must contain only values in [0, {self.num_roles})"
                )


    def _split_heads(self, value: jax.Array) -> jax.Array:
        batch, tokens, _ = value.shape
        value = value.reshape(batch, tokens, self.num_heads, self.head_dim)
        return jnp.transpose(value, (0, 2, 1, 3))

    def _cross_attention(
        self,
        queries: jax.Array,
        tokens: jax.Array,
        token_mask: jax.Array,
    ) -> jax.Array:
        queries = self._split_heads(self.query_projection(queries))  # [B, Q, W] -> [B, H, Q, D]
        keys = self._split_heads(self.key_projection(tokens))  # [B, L, W] -> [B, H, L, D]
        values = self._split_heads(self.value_projection(tokens))

        logits = jnp.einsum("bhqd,bhkd->bhqk", queries, keys) / math.sqrt(self.head_dim)  # similarity between queries and keys
        key_mask = token_mask[:, None, None, :]  # [B, 1, 1, L]
        masked_logits = jnp.where(key_mask, logits, jnp.finfo(logits.dtype).min)  # block padding keys

        attention_weights = jax.nn.softmax(masked_logits, axis=-1)  # softmax to get attention weights
        attention_weights = jnp.where(key_mask, attention_weights, 0.0)  # set attention weights on padding tokens to 0
        normalizer = jnp.maximum(
            jnp.sum(attention_weights, axis=-1, keepdims=True),
            jnp.finfo(attention_weights.dtype).tiny,  # all tokens masked
        )
        attention_weights = attention_weights / normalizer

        attended = jnp.einsum("bhqk,bhkd->bhqd", attention_weights, values)
        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(
            attended.shape[0],
            attended.shape[2],
            self.width,
        )  # [B, H, Q, D] -> [B, Q, W]
        return self.attention_output_projection(attended)

    def __call__(
        self,
        bank_tokens: jax.Array,
        token_mask: jax.Array,
        role_ids: jax.Array,
        item_mask: jax.Array,
    ) -> jax.Array:
        self._validate_inputs(bank_tokens, token_mask, role_ids, item_mask)
        groups, items, tokens, _ = bank_tokens.shape
        flat_items = groups * items

        x = self.input_projection(bank_tokens)
        x = x + self.role_embeddings.value[role_ids]
        x = x + _local_sinusoidal_position_encoding(tokens, self.width, x.dtype)[None, None, :, :]
        x = x.reshape(flat_items, tokens, self.width)
        flat_token_mask = token_mask.reshape(flat_items, tokens)

        queries = jnp.broadcast_to(
            self.learned_queries.value,
            (flat_items, self.num_queries, self.width),
        )
        queries = queries + self._cross_attention(
            self.query_norm(queries),
            self.token_norm(x),
            flat_token_mask,
        )
        queries = queries + self.ffn_out(
            jax.nn.gelu(self.ffn_in(self.ffn_norm(queries)))
        )

        output = self.output_projection(queries)
        output = output.reshape(groups, items, self.num_queries, self.output_dim)
        return jnp.where(item_mask[..., None, None], output, jnp.zeros_like(output))
