from __future__ import annotations

import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import query_mask_or_ones
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.shared import array_typing as at
import openpi.training.utils as training_utils


def _state_values(state: Any) -> list[jax.Array]:
    if state is None:
        return []

    if hasattr(state, "flat_state"):
        variables = state.flat_state().values()
        return [variable.value for variable in variables if hasattr(variable, "value")]

    return [leaf for leaf in jax.tree_util.tree_leaves(state) if hasattr(leaf, "shape")]


def _global_norm(state: Any) -> jax.Array:
    values = _state_values(state)
    if not values:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return optax.global_norm(values)


def _prefix_norm(state: Any, prefix: tuple[str, ...], *, complement: bool = False) -> jax.Array:
    if state is None or not hasattr(state, "flat_state"):
        return jnp.asarray(0.0, dtype=jnp.float32)

    values = []
    for path, variable in state.flat_state().items():
        matches = path[: len(prefix)] == prefix
        if matches != complement and hasattr(variable, "value"):
            values.append(variable.value)

    if not values:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return optax.global_norm(values)


def guided_loss_and_grad(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: GuideConditionedBatch,
) -> tuple[jax.Array, nnx.State, int, int, jax.Array]:
    """Compute one guided loss and dense gradient without applying an update."""

    groups, queries = validate_guide_conditioned_batch(batch)
    model = nnx.merge(state.model_def, state.params)

    if not callable(getattr(model, "compute_guided_loss", None)):
        raise ValueError("state.model_def must provide compute_guided_loss")

    model.train()

    def loss_fn(model_instance: Any, loss_rng: at.KeyArrayLike, guided_batch: GuideConditionedBatch):
        chunked_loss = model_instance.compute_guided_loss(
            loss_rng,
            guided_batch,
            train=True,
        )
        flat_query_mask = query_mask_or_ones(guided_batch).reshape(-1)
        per_query_loss = jnp.mean(
            chunked_loss.reshape((flat_query_mask.shape[0], -1)), axis=-1
        )
        valid_queries = jnp.sum(flat_query_mask, dtype=jnp.float32)
        return jnp.sum(
            per_query_loss * flat_query_mask.astype(per_query_loss.dtype)
        ) / jnp.maximum(valid_queries, 1.0)

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, batch)
    valid_queries = jnp.sum(query_mask_or_ones(batch), dtype=jnp.int32)
    return loss, grads, groups, queries, valid_queries


def _apply_guided_gradients(
    state: training_utils.TrainState,
    grads: nnx.State,
    *,
    trainable_filter: Any,
) -> training_utils.TrainState:
    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, new_params)
    full_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=full_params,
        opt_state=new_opt_state,
    )

    if state.ema_decay is not None:
        if state.ema_params is None:
            raise ValueError("ema_decay is set but ema_params is missing")
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                full_params,
            ),
        )
    return new_state


def _training_info(
    *,
    loss: jax.Array,
    grads: nnx.State,
    params: nnx.State,
    groups: int,
    queries: int,
    valid_queries: jax.Array,
    microbatches: int,
) -> dict[str, at.Array]:
    return {
        "loss": loss,
        "grad_norm": _global_norm(grads),
        "param_norm": _global_norm(params),
        "guide_encoder_grad_norm": _prefix_norm(grads, ("guide_encoder",)),
        "native_backbone_grad_norm": _prefix_norm(
            grads, ("guide_encoder",), complement=True
        ),
        "G": jnp.asarray(groups, dtype=jnp.int32),
        "Q": jnp.asarray(queries, dtype=jnp.int32),
        "valid_queries": jnp.asarray(valid_queries, dtype=jnp.int32),
        "microbatches": jnp.asarray(microbatches, dtype=jnp.int32),
    }


@at.typecheck
def guided_train_step(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: GuideConditionedBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Run one dense Guide-conditioned Pi05 optimization step."""

    loss, grads, groups, queries, valid_queries = guided_loss_and_grad(
        config, rng, state, batch
    )
    new_state = _apply_guided_gradients(
        state, grads, trainable_filter=config.trainable_filter
    )
    info = _training_info(
        loss=loss,
        grads=grads,
        params=new_state.params,
        groups=groups,
        queries=queries,
        valid_queries=valid_queries,
        microbatches=1,
    )
    return new_state, info


@at.typecheck
def guided_accumulated_train_step(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batches: tuple[GuideConditionedBatch, ...],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Accumulate sample-weighted microbatch gradients and update once."""

    if not batches:
        raise ValueError("gradient accumulation requires at least one microbatch")

    weighted_grads = None
    weighted_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_valid = jnp.asarray(0, dtype=jnp.int32)
    first_groups = first_queries = None

    for microbatch_index, batch in enumerate(batches):
        loss, grads, groups, queries, valid_queries = guided_loss_and_grad(
            config,
            jax.random.fold_in(rng, microbatch_index),
            state,
            batch,
        )
        if first_groups is None:
            first_groups, first_queries = groups, queries
        elif (groups, queries) != (first_groups, first_queries):
            raise ValueError("all accumulated microbatches must share G and Q")
        weight = valid_queries.astype(jnp.float32)
        current = jax.tree.map(
            lambda value, weight=weight: value * weight, grads
        )
        weighted_grads = (
            current
            if weighted_grads is None
            else jax.tree.map(lambda left, right: left + right, weighted_grads, current)
        )
        weighted_loss = weighted_loss + loss * weight
        total_valid = total_valid + valid_queries

    assert weighted_grads is not None
    assert first_groups is not None
    assert first_queries is not None
    divisor = jnp.maximum(total_valid.astype(jnp.float32), 1.0)
    accumulated_grads = jax.tree.map(
        lambda value: value / divisor, weighted_grads
    )
    combined_loss = weighted_loss / divisor
    new_state = _apply_guided_gradients(
        state,
        accumulated_grads,
        trainable_filter=config.trainable_filter,
    )
    info = _training_info(
        loss=combined_loss,
        grads=accumulated_grads,
        params=new_state.params,
        groups=first_groups,
        queries=first_queries,
        valid_queries=total_valid,
        microbatches=len(batches),
    )
    return new_state, info
