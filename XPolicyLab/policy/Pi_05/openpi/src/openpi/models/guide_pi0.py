from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import guide_pi0_config
from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.models.guide_attention import make_gc_attn_mask
from openpi.models.guide_attention import make_gca_attn_mask
from openpi.models.guide_encoder import GuideFeatureEncoder
from openpi.models.guide_encoder import GuideMemory
from openpi.models.guide_encoder import encode_shared_guide_features
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_inputs import broadcast_guide_memory
from openpi.models.guide_inputs import flatten_grouped_control
from openpi.models.guide_inputs import flatten_grouped_observation
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.models.guide_inputs import validate_guide_conditioned_observation
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at


class GuidePi0(Pi0):
    """Pi05 model structure extended with a Guide encoder."""

    def __init__(
        self,
        config: guide_pi0_config.GuidePi0Config,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)

        paligemma_config = _gemma.get_config(config.paligemma_variant)

        self.guide_encoder = GuideFeatureEncoder(
            input_dim=paligemma_config.width,
            output_dim=paligemma_config.width,
            num_queries=config.guide_num_queries,
            width=config.guide_resampler_width,
            num_heads=config.guide_resampler_num_heads,
            ffn_hidden_dim=config.guide_resampler_ffn_hidden_dim,
            rngs=rngs,
        )

    def encode_guide(self, guide: GuideInput) -> GuideMemory:
        """Encode one grouped Guide bank with the shared Pi05 backbones."""

        features = encode_shared_guide_features(
            guide.images,
            guide.text_tokens,
            image_encoder=self.PaliGemma.img,
            text_embedder=self.PaliGemma.llm,
        )

        return self.guide_encoder(
            features.frame_tokens,
            guide.image_mask,
            features.text_embeddings,
            guide.text_mask,
            guide.unit_mask,
            guide.before_slot,
            guide.after_slot,
        )

    def _embed_guide_control_prefix(
        self,
        guide_memory: GuideMemory,
        flat_observation: _model.Observation,
        *,
        queries_per_guide: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Combine broadcast Guide tokens with the native Control prefix."""

        guide_tokens = broadcast_guide_memory(
            guide_memory.tokens,
            queries_per_guide=queries_per_guide,
        )
        guide_mask = broadcast_guide_memory(
            guide_memory.token_mask,
            queries_per_guide=queries_per_guide,
        )

        control_tokens, control_mask, _ = self.embed_prefix(flat_observation)

        combined_prefix_tokens = jnp.concatenate(
            [guide_tokens, control_tokens],
            axis=1,
        )

        return combined_prefix_tokens, guide_mask, control_mask


    def _prefill_guided_prefix(
        self,
        observation: _model.Observation,
        guide: GuideInput,
    ) -> tuple[_model.Observation, jax.Array, object, int, int]:
        """Encode Guide and prefill the combined Guide-Control prefix."""

        groups, queries = validate_guide_conditioned_observation(
            observation,
            guide,
        )

        guide_memory = self.encode_guide(guide)

        flat_observation = flatten_grouped_observation(observation)
        flat_observation = _model.preprocess_observation(
            None,
            flat_observation,
            train=False,
        )

        prefix_tokens, guide_mask, control_mask = (
            self._embed_guide_control_prefix(
                guide_memory,
                flat_observation,
                queries_per_guide=queries,
            )
        )

        prefix_mask = jnp.concatenate(
            [guide_mask, control_mask],
            axis=1,
        )

        prefix_attn_mask = make_gc_attn_mask(
            guide_mask,
            control_mask,
        )

        prefix_positions = jnp.cumsum(
            prefix_mask,
            axis=1,
        ) - 1

        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        return (
            flat_observation,
            prefix_mask,
            kv_cache,
            groups,
            queries,
        )

    def sample_guided_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        guide: GuideInput,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: _model.Actions | None = None,
    ) -> _model.Actions:
        """Sample actions conditioned on one grouped Guide bank."""

        groups, queries = validate_guide_conditioned_observation(
            observation,
            guide,
        )

        batch_size = groups * queries
        expected_noise_shape = (
            groups,
            queries,
            self.action_horizon,
            self.action_dim,
        )

        if noise is None:
            flat_noise = jax.random.normal(
                rng,
                (
                    batch_size,
                    self.action_horizon,
                    self.action_dim,
                ),
            )
        else:
            if tuple(noise.shape) != expected_noise_shape:
                raise ValueError(
                    "noise must have grouped shape "
                    f"{expected_noise_shape}, got {noise.shape}"
                )

            flat_noise = noise.reshape(
                batch_size,
                self.action_horizon,
                self.action_dim,
            )

        flat_observation, prefix_mask, kv_cache, _, _ = (
            self._prefill_guided_prefix(
                observation,
                guide,
            )
        )

        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
                self.embed_suffix(
                    flat_observation,
                    x_t,
                    jnp.broadcast_to(time, batch_size),
                )
            )

            suffix_attn_mask = make_attn_mask(
                suffix_mask,
                suffix_ar_mask,
            )

            prefix_to_suffix_mask = jnp.broadcast_to(
                prefix_mask[:, None, :],
                (
                    batch_size,
                    suffix_tokens.shape[1],
                    prefix_mask.shape[1],
                ),
            )

            full_attn_mask = jnp.concatenate(
                [
                    prefix_to_suffix_mask,
                    suffix_attn_mask,
                ],
                axis=-1,
            )

            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )

            assert prefix_out is None

            velocity = self.action_out_proj(
                suffix_out[:, -self.action_horizon :]
            )

            return x_t + dt * velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        flat_actions, _ = jax.lax.while_loop(
            cond,
            step,
            (flat_noise, 1.0),
        )

        return flat_actions.reshape(expected_noise_shape)

    def _prepare_guided_loss_inputs(
        self,
        rng: at.KeyArrayLike,
        batch: GuideConditionedBatch,
        *,
        train: bool,
    ) -> tuple[
        _model.Observation,
        jax.Array,
        jax.Array,
        jax.Array,
        int,
    ]:
        """Prepare flattened Control inputs and stock Pi05 flow targets."""

        _, queries = validate_guide_conditioned_batch(batch)

        flat_observation, flat_actions = flatten_grouped_control(
            batch.observation,
            batch.actions,
        )

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        flat_observation = _model.preprocess_observation(
            preprocess_rng,
            flat_observation,
            train=train,
        )

        noise = jax.random.normal(noise_rng, flat_actions.shape)

        batch_shape = flat_actions.shape[:-2]
        time = jax.random.beta(
            time_rng,
            1.5,
            1,
            batch_shape,
        )
        time = time * 0.999 + 0.001

        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * flat_actions
        u_t = noise - flat_actions

        return flat_observation, x_t, u_t, time, queries

    def _run_guided_joint_forward(
        self,
        guide_memory: GuideMemory,
        flat_observation: _model.Observation,
        x_t: jax.Array,
        time: jax.Array,
        *,
        queries_per_guide: int,
    ) -> jax.Array:
        """Run one Guide/Control/Action forward pass and predict action velocity."""

        combined_prefix_tokens, guide_mask, control_mask = (
            self._embed_guide_control_prefix(
                guide_memory,
                flat_observation,
                queries_per_guide=queries_per_guide,
            )
        )

        suffix_tokens, action_mask, _, adarms_cond = self.embed_suffix(
            flat_observation,
            x_t,
            time,
        )

        input_mask = jnp.concatenate(
            [guide_mask, control_mask, action_mask],
            axis=1,
        )

        attn_mask = make_gca_attn_mask(
            guide_mask,
            control_mask,
            action_mask,
        )

        positions = jnp.cumsum(input_mask, axis=1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [combined_prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        return self.action_out_proj(
            suffix_out[:, -self.action_horizon :]
        )

    @at.typecheck
    def compute_guided_loss(
        self,
        rng: at.KeyArrayLike,
        batch: GuideConditionedBatch,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute the Guide-conditioned Pi05 action flow-matching loss."""

        flat_observation, x_t, u_t, time, queries = self._prepare_guided_loss_inputs(
            rng,
            batch,
            train=train,
        )

        guide_memory = self.encode_guide(batch.guide)

        predicted_velocity = self._run_guided_joint_forward(
            guide_memory,
            flat_observation,
            x_t,
            time,
            queries_per_guide=queries,
        )

        return jnp.mean(
            jnp.square(predicted_velocity - u_t),
            axis=-1,
        )
