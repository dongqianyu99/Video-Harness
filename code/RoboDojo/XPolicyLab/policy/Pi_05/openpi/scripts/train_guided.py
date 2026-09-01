from __future__ import annotations

import argparse
import dataclasses
import importlib
import itertools
import logging
from pathlib import Path
import time
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.models.guide_pi0 import GuidePi0
from openpi.models.guide_pi0_config import GuidePi0Config
from openpi.training import robodojo_defaults as _defaults
from openpi.training.guide_data_loader import prefetch_guided_batches
from openpi.training.guide_run import GuidedResumeContractError
from openpi.training.guide_run import GuidedRunLayout
from openpi.training.guide_run import configure_guided_run_logging
from openpi.training.guide_run import finish_guided_wandb
from openpi.training.guide_run import init_guided_wandb
from openpi.training.guide_run import prepare_guided_run_layout
from openpi.training.guide_run import validate_guided_run_resume
from openpi.training.guide_run import write_guided_run_manifest
from openpi.training.guide_train_config import GuidedTrainRunConfig
from openpi.training.guide_train_config import resolve_guided_train_config
from openpi.training.guide_train_sharding import make_guided_batch_sharding
from openpi.training.guide_train_step import guided_accumulated_train_step
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig

logger = logging.getLogger(__name__)


def _load_native_config(name: str) -> Any:
    return importlib.import_module("openpi.training.config").get_config(name)


def _load_checkpoints() -> Any:
    return importlib.import_module("openpi.training.checkpoints")


def _validate_runtime_config(run_config: GuidedTrainRunConfig, resolved_config: Any) -> None:
    if jax.process_count() != 1:
        raise ValueError("M5 guided training currently requires jax.process_count() == 1")
    if not isinstance(resolved_config.model, GuidePi0Config):
        raise ValueError("resolved guided training config must use GuidePi0Config")
    if not resolved_config.model.pi05:
        raise ValueError("resolved guided training config must use GuidePi0 Pi05")
    if (
        resolved_config.model.guide_boundary_num_queries
        != run_config.guided_data.guide_boundary_num_queries
        or resolved_config.model.guide_transition_num_queries
        != run_config.guided_data.guide_transition_num_queries
    ):
        raise ValueError("resolved Guide capacity does not match guided data config")
    if resolved_config.batch_size != run_config.guided_data.batch_size:
        raise ValueError(
            "resolved TrainConfig.batch_size does not match guided_data.batch_size: "
            f"{resolved_config.batch_size} != {run_config.guided_data.batch_size}"
        )
    if resolved_config.num_workers != run_config.guided_data.num_workers:
        raise ValueError(
            "resolved num_workers does not match guided data config: "
            f"{resolved_config.num_workers} != {run_config.guided_data.num_workers}"
        )
    if not isinstance(resolved_config.freeze_filter, nnx.Nothing):
        raise ValueError("M5 guided training requires full dense fine-tuning with freeze_filter=nnx.Nothing()")
    effective_batch_size = run_config.effective_global_batch_size
    if run_config.enforce_reference_batch_size:
        if run_config.guided_data.remainder_strategy != "drop":
            raise ValueError("strict reference batch alignment requires remainder_strategy='drop'")
        if effective_batch_size != run_config.reference_global_batch_size:
            raise ValueError(
                "effective global batch does not match the reference: "
                f"microbatch={run_config.guided_data.batch_size} * "
                f"accumulation={run_config.gradient_accumulation_steps} = "
                f"{effective_batch_size}, expected "
                f"{run_config.reference_global_batch_size}"
            )
    if run_config.base_params_path.resolve() == run_config.checkpoint_dir.resolve():
        raise ValueError("base_params_path and guided resume checkpoint_dir must be different")


def _require_checkpoint_data_config(data_loader: Any) -> Any:
    data_config = getattr(data_loader, "data_config", None)
    if not callable(data_config):
        raise ValueError("GuidedDataLoader must provide data_config() before checkpoint save/restore")
    try:
        return data_config()
    except Exception as exc:
        raise ValueError("guided checkpoint operation requires a native data_config") from exc


def save_guided_state(checkpoint_manager: Any, state: Any, data_loader: Any, step: int) -> Any:
    """Delegate to stock checkpoint format after validating native assets are available."""

    _require_checkpoint_data_config(data_loader)
    return _load_checkpoints().save_state(checkpoint_manager, state, data_loader, step)


def restore_guided_state(
    checkpoint_manager: Any,
    state: Any,
    data_loader: Any,
    step: int | None = None,
) -> Any:
    """Restore a guided resume checkpoint using the stock checkpoint format."""

    _require_checkpoint_data_config(data_loader)
    return _load_checkpoints().restore_state(checkpoint_manager, state, data_loader, step)


def _catalog_log(data_loader: Any) -> dict[str, Any]:
    catalog = getattr(data_loader, "guide_catalog", None)
    if catalog is None:
        return {"count": 0, "tasks": 0, "examples": []}
    records = tuple(catalog.records)
    examples = [
        {
            "guide_index": record.guide_index,
            "source_episode_index": record.source_episode_index,
            "document_id": record.document_id,
            "task_index": record.task_index,
        }
        for record in records[:10]
    ]
    return {
        "count": len(records),
        "tasks": len({record.task_index for record in records}),
        "catalog_digest": catalog.catalog_digest,
        "examples": examples,
    }


def _parameter_count(state: Any) -> int:
    if not hasattr(state, "flat_state"):
        return 0
    return sum(int(np.prod(variable.value.shape)) for variable in state.flat_state().values())


def _detach_ema_buffers(state: Any) -> Any:
    """Give donated train state and EMA state distinct device buffers."""

    if state.ema_params is None:
        return state

    def _copy_leaf(value):
        return value.copy()

    return dataclasses.replace(
        state,
        ema_params=jax.tree.map(_copy_leaf, state.ema_params),
    )


def _run_guided_training_impl(
    run_config: GuidedTrainRunConfig,
    layout: GuidedRunLayout,
    session: dict[str, Any],
) -> Any:
    """Run the guided loop; real model/data execution belongs on the server."""

    resolved_config = resolve_guided_train_config(run_config)
    _validate_runtime_config(run_config, resolved_config)
    native_config = _load_native_config(run_config.native_config_name)

    if resolved_config.fsdp_devices <= 0:
        raise ValueError("fsdp_devices must be positive")

    data_module = importlib.import_module("openpi.training.robodojo_guide_data")
    data_loader = data_module.create_robodojo_guided_data_loader(
        native_config,
        run_config.guided_data,
        num_batches=None,
    )
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    if not isinstance(first_batch, GuideConditionedBatch):
        raise ValueError("guided data loader did not return GuideConditionedBatch")
    groups, queries = validate_guide_conditioned_batch(first_batch)
    runtime_metadata = {
        "groups_per_microbatch": groups,
        "queries_per_guide": queries,
        "effective_global_batch_size": run_config.effective_global_batch_size,
        "data": getattr(data_loader, "host_metadata", {}),
    }
    validate_guided_run_resume(
        layout,
        run_config=run_config,
        runtime_data=runtime_metadata["data"],
    )
    write_guided_run_manifest(
        layout,
        run_config=run_config,
        status="running",
        runtime=runtime_metadata,
    )
    configure_guided_run_logging(layout, resume=run_config.resume)
    checkpoints = _load_checkpoints()
    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        resolved_config.checkpoint_dir,
        keep_period=resolved_config.keep_period,
        overwrite=resolved_config.overwrite,
        resume=resolved_config.resume,
    )
    session["wandb_run"] = init_guided_wandb(
        layout=layout,
        run_config=run_config,
        resolved_config=resolved_config,
        resuming=resuming,
    )
    train_script = importlib.import_module("scripts.train")
    wandb_run = session.get("wandb_run")
    wandb_config = getattr(wandb_run, "config", None)
    update_wandb_config = getattr(wandb_config, "update", None)
    if callable(update_wandb_config):
        update_wandb_config({"runtime": runtime_metadata}, allow_val_change=True)

    mesh_module = importlib.import_module("openpi.training.sharding")
    mesh = mesh_module.make_mesh(resolved_config.fsdp_devices)
    batch_sharding = make_guided_batch_sharding(first_batch, mesh)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rng = jax.random.key(resolved_config.seed)
    train_rng, init_rng = jax.random.split(rng)
    train_state, train_state_sharding = train_script.init_train_state(
        resolved_config,
        init_rng,
        mesh,
        resume=resuming,
    )
    jax.block_until_ready(train_state)

    model = nnx.merge(train_state.model_def, train_state.params)
    if not isinstance(model, GuidePi0):
        raise ValueError(f"initialized guided state uses {type(model).__name__}, expected GuidePi0")

    if resuming:
        train_state = restore_guided_state(checkpoint_manager, train_state, data_loader)

    train_state = _detach_ema_buffers(train_state)

    logger.info(
        "Initialized GuidePi0: G=%d Q=%d devices=%d parameters=%d base_params_path=%s "
        "resume_checkpoint_dir=%s catalog=%s metadata=%s",
        groups,
        queries,
        jax.device_count(),
        _parameter_count(train_state.params),
        run_config.base_params_path,
        run_config.checkpoint_dir,
        _catalog_log(data_loader),
        getattr(data_loader, "host_metadata", {}),
    )

    accumulation_steps = run_config.gradient_accumulation_steps
    ptrain_step = jax.jit(
        lambda step_rng, state, batches: guided_accumulated_train_step(resolved_config, step_rng, state, batches),
        in_shardings=(
            replicated_sharding,
            train_state_sharding,
            tuple(batch_sharding for _ in range(accumulation_steps)),
        ),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    device_batches = prefetch_guided_batches(
        iter(itertools.chain((first_batch,), data_iter)),
        sharding=batch_sharding,
        size=run_config.guided_data.device_prefetch_size,
    )
    batches = tuple(next(device_batches) for _ in range(accumulation_steps))
    start_step = int(train_state.step)
    lr_schedule = resolved_config.lr_schedule.create()
    last_log_time = time.perf_counter()
    last_log_step = start_step
    accumulated_data_wait_s = 0.0

    for step in range(start_step, resolved_config.num_train_steps):
        with mesh_module.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batches)

        if step < resolved_config.num_train_steps - 1:
            data_wait_start = time.perf_counter()
            batches = tuple(next(device_batches) for _ in range(accumulation_steps))
            accumulated_data_wait_s += time.perf_counter() - data_wait_start

        if step % resolved_config.log_interval == 0:
            host_info = {key: float(value) for key, value in jax.device_get(info).items() if key not in {"G", "Q"}}
            now = time.perf_counter()
            interval_steps = step - last_log_step + 1
            interval_s = max(now - last_log_time, 1e-9)
            metrics = {
                "train/optimizer_step": step,
                "train/loss": host_info["loss"],
                "train/grad_norm": host_info["grad_norm"],
                "train/param_norm": host_info["param_norm"],
                "train/guide_encoder_grad_norm": host_info["guide_encoder_grad_norm"],
                "train/native_backbone_grad_norm": host_info["native_backbone_grad_norm"],
                "train/learning_rate": float(jax.device_get(lr_schedule(step))),
                "batch/groups": int(info["G"]),
                "batch/queries_per_guide": int(info["Q"]),
                "batch/valid_queries": int(info["valid_queries"]),
                "batch/microbatches": int(info["microbatches"]),
                "batch/effective_capacity": run_config.effective_global_batch_size,
                "batch/reference_global_batch": run_config.reference_global_batch_size,
                "performance/optimizer_steps_per_s": interval_steps / interval_s,
                "performance/data_wait_ms_per_step": (accumulated_data_wait_s / interval_steps * 1000.0),
                "system/device_count": jax.device_count(),
            }
            logger.info(
                "step=%d loss=%.6f grad_norm=%.6f guide_grad_norm=%.6f native_grad_norm=%.6f "
                "G=%s Q=%s valid_queries=%s microbatches=%s effective_capacity=%d reference_batch=%d",
                step,
                host_info["loss"],
                host_info["grad_norm"],
                host_info["guide_encoder_grad_norm"],
                host_info["native_backbone_grad_norm"],
                int(info["G"]),
                int(info["Q"]),
                int(info["valid_queries"]),
                int(info["microbatches"]),
                run_config.guided_data.batch_size * accumulation_steps,
                run_config.reference_global_batch_size,
            )
            wandb_run = session.get("wandb_run")
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            last_log_time = now
            last_log_step = step + 1
            accumulated_data_wait_s = 0.0

        if (
            step % resolved_config.save_interval == 0 and step > start_step
        ) or step == resolved_config.num_train_steps - 1:
            save_guided_state(checkpoint_manager, train_state, data_loader, step)

    checkpoint_manager.wait_until_finished()
    return train_state


def run_guided_training(run_config: GuidedTrainRunConfig) -> Any:
    """Run one tracked Guide-conditioned training experiment."""

    layout = GuidedRunLayout.from_paths(
        run_dir=run_config.run_dir,
        checkpoint_dir=run_config.checkpoint_dir,
    )
    prepare_guided_run_layout(
        layout,
        resume=run_config.resume,
        overwrite=run_config.overwrite,
    )
    validate_guided_run_resume(layout, run_config=run_config)
    if not run_config.resume:
        write_guided_run_manifest(
            layout,
            run_config=run_config,
            status="running",
        )
    session: dict[str, Any] = {}
    try:
        result = _run_guided_training_impl(run_config, layout, session)
    except BaseException as exc:
        logger.exception("Guided training failed")
        if not isinstance(exc, GuidedResumeContractError):
            write_guided_run_manifest(
                layout,
                run_config=run_config,
                status="failed",
                error=exc,
            )
        finish_guided_wandb(session.get("wandb_run"), exit_code=1)
        raise
    write_guided_run_manifest(
        layout,
        run_config=run_config,
        status="completed",
        runtime={"final_optimizer_step": int(result.step)},
    )
    finish_guided_wandb(session.get("wandb_run"), exit_code=0)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Guide-conditioned Pi05 with the RoboDojo data path.")
    parser.add_argument("--native-config-name", required=True)
    parser.add_argument("--base-params-path", type=Path, required=True)
    parser.add_argument("--repo-id", default=_defaults.ROBODOJO_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=_defaults.ROBODOJO_DATASET_ROOT)
    parser.add_argument("--documents-root", type=Path, default=_defaults.GUIDE_DOCUMENTS_ROOT)
    parser.add_argument(
        "--guide-materialization-cache-root",
        type=Path,
        default=_defaults.GUIDE_MATERIALIZATION_CACHE_ROOT,
    )
    parser.add_argument("--guides-per-batch", type=int, default=_defaults.GUIDES_PER_BATCH)
    parser.add_argument("--queries-per-guide", type=int, default=_defaults.QUERIES_PER_GUIDE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device-prefetch-size", type=int, default=2)
    parser.add_argument("--guide-cache-entries", type=int, default=2)
    parser.add_argument("--guide-cache-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--worker-timeout-s", type=float, default=0.0)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=_defaults.GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--reference-global-batch-size", type=int, default=256)
    parser.add_argument("--allow-effective-batch-mismatch", action="store_true")
    parser.add_argument(
        "--remainder-strategy",
        choices=("drop", "pad_mask"),
        default="drop",
    )
    parser.add_argument("--max-boundaries", type=int, default=_defaults.MAX_BOUNDARIES)
    parser.add_argument("--max-units", type=int, default=_defaults.MAX_UNITS)
    parser.add_argument(
        "--max-boundary-text-tokens",
        type=int,
        default=_defaults.MAX_BOUNDARY_TEXT_TOKENS,
    )
    parser.add_argument(
        "--max-transition-text-tokens",
        type=int,
        default=_defaults.MAX_TRANSITION_TEXT_TOKENS,
    )
    parser.add_argument("--guide-boundary-num-queries", type=int, default=8)
    parser.add_argument("--guide-transition-num-queries", type=int, default=4)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="legacy path; new runs should use --run-dir",
    )
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    return parser


def _run_from_args(args: argparse.Namespace) -> Any:
    accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    run_dir = getattr(args, "run_dir", None)
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if run_dir is not None:
        expected_checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir is not None and checkpoint_dir.resolve() != expected_checkpoint_dir.resolve():
            raise ValueError("--checkpoint-dir must equal --run-dir/checkpoints when both are set")
        checkpoint_dir = expected_checkpoint_dir
    elif checkpoint_dir is None:
        raise ValueError("provide --run-dir for a tracked run")

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
        guide_boundary_num_queries=getattr(args, "guide_boundary_num_queries", 8),
        guide_transition_num_queries=getattr(args, "guide_transition_num_queries", 4),
        num_workers=getattr(args, "num_workers", 0),
        prefetch_factor=getattr(args, "prefetch_factor", 2),
        persistent_workers=not getattr(args, "no_persistent_workers", False),
        worker_timeout_s=getattr(args, "worker_timeout_s", 0.0),
        worker_torch_threads=getattr(args, "worker_torch_threads", 1),
        guide_cache_entries=getattr(args, "guide_cache_entries", 2),
        guide_cache_max_bytes=getattr(args, "guide_cache_max_bytes", 256 * 1024 * 1024),
        device_prefetch_size=getattr(args, "device_prefetch_size", 2),
        require_all_tasks=True,
        remainder_strategy=getattr(args, "remainder_strategy", "drop"),
        gradient_accumulation_steps=accumulation_steps,
    )
    run_config = GuidedTrainRunConfig(
        native_config_name=args.native_config_name,
        base_params_path=args.base_params_path,
        guided_data=guided_data,
        experiment_name=args.experiment_name,
        checkpoint_dir=checkpoint_dir,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        fsdp_devices=args.fsdp_devices,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        gradient_accumulation_steps=accumulation_steps,
        reference_global_batch_size=getattr(args, "reference_global_batch_size", 256),
        enforce_reference_batch_size=not getattr(args, "allow_effective_batch_mismatch", False),
        run_dir=run_dir,
    )
    return run_guided_training(run_config)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args(argv)
    _run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
