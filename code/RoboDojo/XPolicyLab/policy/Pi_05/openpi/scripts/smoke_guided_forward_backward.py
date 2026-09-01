from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_pi0_config import GuidePi0Config
from openpi.training.guide_train_config import GuidedTrainRunConfig
from openpi.training.guide_train_config import resolve_guided_train_config
from openpi.training.guide_train_sharding import make_guided_batch_sharding
from openpi.training.guide_train_sharding import put_guided_batch
from openpi.training.guide_train_step import _prefix_norm
from openpi.training.guide_train_step import guided_loss_and_grad
from openpi.training.guide_train_step import guided_train_step
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one real GuidePi05 forward/backward smoke.")
    parser.add_argument("--native-config-name", required=True)
    parser.add_argument("--base-params-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--documents-root", type=Path, required=True)
    parser.add_argument("--guide-materialization-cache-root", type=Path, required=True)
    parser.add_argument("--guides-per-batch", type=int, required=True)
    parser.add_argument("--queries-per-guide", type=int, required=True)
    parser.add_argument("--max-boundaries", type=int, required=True)
    parser.add_argument("--max-units", type=int, required=True)
    parser.add_argument("--max-boundary-text-tokens", type=int, required=True)
    parser.add_argument("--max-transition-text-tokens", type=int, required=True)
    parser.add_argument("--guide-boundary-num-queries", type=int, default=8)
    parser.add_argument("--guide-transition-num-queries", type=int, default=4)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--no-optimizer-update", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _shape(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def _finite_tree(tree: Any) -> bool:
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree) if hasattr(leaf, "shape")]
    return all(bool(np.isfinite(leaf).all()) for leaf in leaves)


def _parameter_count(state: Any) -> int:
    if not hasattr(state, "flat_state"):
        return 0
    return sum(int(np.prod(variable.value.shape)) for variable in state.flat_state().values())


def _memory_stats() -> dict[str, int] | None:
    try:
        stats = jax.devices()[0].memory_stats()
    except (AttributeError, RuntimeError):
        return None
    if stats is None:
        return None
    return {key: int(value) for key, value in stats.items() if isinstance(value, (int, np.integer))}


def _gradient_sharding(train_state_sharding: Any, trainable_filter: Any) -> Any:
    """Mirror ``nnx.DiffState``: gradients only contain trainable params."""

    params_sharding = getattr(train_state_sharding, "params", None)
    if params_sharding is None or not callable(getattr(params_sharding, "filter", None)):
        raise ValueError("train state sharding does not expose filtered params sharding")
    return params_sharding.filter(trainable_filter)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_module = importlib.import_module("openpi.training.config")
    native_config = config_module.get_config(args.native_config_name)
    guided_data = RoboDojoGuidedDataConfig(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        documents_root=args.documents_root,
        guide_materialization_cache_root=args.guide_materialization_cache_root,
        guides_per_batch=args.guides_per_batch,
        queries_per_guide=args.queries_per_guide,
        seed=args.seed,
        max_boundaries=args.max_boundaries,
        max_units=args.max_units,
        max_boundary_text_tokens=args.max_boundary_text_tokens,
        max_transition_text_tokens=args.max_transition_text_tokens,
        guide_boundary_num_queries=args.guide_boundary_num_queries,
        guide_transition_num_queries=args.guide_transition_num_queries,
    )
    run_config = GuidedTrainRunConfig(
        native_config_name=args.native_config_name,
        base_params_path=args.base_params_path,
        guided_data=guided_data,
        experiment_name="forward-backward-smoke",
        checkpoint_dir=args.base_params_path.parent / "guided-forward-backward-smoke",
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        fsdp_devices=args.fsdp_devices,
    )
    resolved_config = resolve_guided_train_config(run_config)
    if not isinstance(resolved_config.model, GuidePi0Config):
        raise ValueError("resolved config is not GuidePi0Config")

    data_module = importlib.import_module("openpi.training.robodojo_guide_data")
    data_loader = data_module.create_robodojo_guided_data_loader(
        native_config,
        guided_data,
        num_batches=1,
    )
    batch = next(iter(data_loader))
    if not isinstance(batch, GuideConditionedBatch):
        raise ValueError("guided data loader did not return GuideConditionedBatch")

    mesh_module = importlib.import_module("openpi.training.sharding")
    mesh = mesh_module.make_mesh(args.fsdp_devices)
    batch_sharding = make_guided_batch_sharding(batch, mesh)
    batch = put_guided_batch(batch, batch_sharding)

    train_script = importlib.import_module("scripts.train")
    rng = jax.random.key(args.seed)
    _, init_rng = jax.random.split(rng)
    train_state, train_state_sharding = train_script.init_train_state(
        resolved_config,
        init_rng,
        mesh,
        resume=False,
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    grad_sharding = _gradient_sharding(train_state_sharding, resolved_config.trainable_filter)
    with mesh_module.set_mesh(mesh):
        if args.no_optimizer_update:
            pforward = jax.jit(
                lambda step_rng, state, guided_batch: guided_loss_and_grad(
                    resolved_config, step_rng, state, guided_batch
                ),
                in_shardings=(replicated_sharding, train_state_sharding, batch_sharding),
                out_shardings=(
                    replicated_sharding,
                    grad_sharding,
                    replicated_sharding,
                    replicated_sharding,
                    replicated_sharding,
                ),
            )
            loss, grads, groups, queries, valid_queries = pforward(rng, train_state, batch)
            jax.block_until_ready((loss, grads))
            info = {
                "loss": loss,
                "guide_encoder_grad_norm": _prefix_norm(grads, ("guide_encoder",)),
                "native_backbone_grad_norm": _prefix_norm(grads, ("guide_encoder",), complement=True),
                "G": groups,
                "Q": queries,
                "valid_queries": valid_queries,
            }
        else:
            ptrain_step = jax.jit(
                lambda step_rng, state, guided_batch: guided_train_step(resolved_config, step_rng, state, guided_batch),
                in_shardings=(replicated_sharding, train_state_sharding, batch_sharding),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )
            train_state, info = ptrain_step(rng, train_state, batch)
            jax.block_until_ready(info)

    loss = info["loss"]
    groups = info["G"]
    queries = info["Q"]

    output = {
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(resolved_config.model.dtype),
        "G": groups,
        "Q": queries,
        "observation_image_shapes": {key: _shape(value) for key, value in batch.observation.images.items()},
        "state_shape": _shape(batch.observation.state),
        "action_shape": _shape(batch.actions),
        "guide_boundary_image_shape": _shape(batch.guide.boundary_images),
        "guide_boundary_text_shape": _shape(batch.guide.boundary_text_tokens),
        "guide_transition_text_shape": _shape(batch.guide.transition_text_tokens),
        "guide_memory_mask_shape": _shape(batch.guide.memory_mask),
        "loss": float(loss),
        "finite_loss": bool(np.isfinite(float(loss))),
        "finite_gradients": _finite_tree(info),
        "guide_encoder_grad_norm": float(info["guide_encoder_grad_norm"]),
        "native_backbone_grad_norm": float(info["native_backbone_grad_norm"]),
        "parameter_count": _parameter_count(train_state.params),
        "peak_device_memory": _memory_stats(),
        "optimizer_updated": not args.no_optimizer_update,
    }

    if not output["finite_loss"] or not output["finite_gradients"]:
        raise ValueError("guided forward/backward produced non-finite loss or gradients")

    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(_run(args), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
