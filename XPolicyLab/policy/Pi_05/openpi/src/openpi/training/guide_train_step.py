from __future__ import annotations

import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models.guide_inputs import GuideConditionedBatch
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
) -> tuple[jax.Array, nnx.State, int, int]:
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
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, batch)
    return loss, grads, groups, queries


@at.typecheck
def guided_train_step(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: GuideConditionedBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Run one dense Guide-conditioned Pi05 optimization step."""

    loss, grads, groups, queries = guided_loss_and_grad(config, rng, state, batch)

    params = state.params.filter(config.trainable_filter)
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

    info = {
        "loss": loss,
        "grad_norm": _global_norm(grads),
        "param_norm": _global_norm(full_params),
        "guide_encoder_grad_norm": _prefix_norm(
            grads,
            ("guide_encoder",),
        ),
        "native_backbone_grad_norm": _prefix_norm(
            grads,
            ("guide_encoder",),
            complement=True,
        ),
        "G": jnp.asarray(groups, dtype=jnp.int32),
        "Q": jnp.asarray(queries, dtype=jnp.int32),
    }
    return new_state, info
