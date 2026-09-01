from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import importlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = "openpi.guided-run.v0"


class GuidedResumeContractError(ValueError):
    """Raised before mutating a run whose resume contract drifted."""


@dataclass(frozen=True, slots=True)
class GuidedRunLayout:
    root: Path
    checkpoints: Path
    logs: Path
    wandb: Path
    eval: Path
    manifest: Path
    train_log: Path
    wandb_id: Path
    legacy_flat_checkpoints: bool

    @classmethod
    def from_paths(
        cls,
        *,
        run_dir: Path | None,
        checkpoint_dir: Path,
    ) -> GuidedRunLayout:
        checkpoint_dir = checkpoint_dir.resolve()
        if run_dir is None:
            root = checkpoint_dir.parent / f"{checkpoint_dir.name}.run"
            legacy_flat = True
        else:
            root = run_dir.resolve()
            expected_checkpoint_dir = root / "checkpoints"
            if checkpoint_dir != expected_checkpoint_dir:
                raise ValueError(
                    "an explicit run_dir requires checkpoint_dir=run_dir/checkpoints: "
                    f"expected {expected_checkpoint_dir}, got {checkpoint_dir}"
                )
            legacy_flat = False
        return cls(
            root=root,
            checkpoints=checkpoint_dir,
            logs=root / "logs",
            wandb=root / "wandb",
            eval=root / "eval",
            manifest=root / "run.json",
            train_log=root / "logs" / "train.log",
            wandb_id=root / "wandb_id.txt",
            legacy_flat_checkpoints=legacy_flat,
        )


def prepare_guided_run_layout(
    layout: GuidedRunLayout,
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be true")
    if layout.root.exists() and overwrite:
        shutil.rmtree(layout.root)
    if layout.root.exists() and any(layout.root.iterdir()) and not resume:
        raise FileExistsError(
            f"run directory already contains artifacts: {layout.root}; "
            "use --resume or --overwrite"
        )
    if resume:
        if not layout.root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {layout.root}")
        if not layout.manifest.is_file():
            raise FileNotFoundError(
                f"resume requires the run manifest: {layout.manifest}"
            )
        if not layout.checkpoints.is_dir():
            raise FileNotFoundError(
                f"resume requires the checkpoint directory: {layout.checkpoints}"
            )

    layout.root.mkdir(parents=True, exist_ok=True)
    layout.logs.mkdir(parents=True, exist_ok=True)
    layout.wandb.mkdir(parents=True, exist_ok=True)
    layout.eval.mkdir(parents=True, exist_ok=True)


def configure_guided_run_logging(layout: GuidedRunLayout, *, resume: bool) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in tuple(root_logger.handlers):
        if getattr(handler, "guided_run_file_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(
        layout.train_log,
        mode="a" if resume else "w",
        encoding="utf-8",
    )
    handler.guided_run_file_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s "
            "(%(process)d:%(filename)s:%(lineno)d)"
        )
    )
    root_logger.addHandler(handler)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalized_run_config(value: Any) -> dict[str, Any]:
    config = _jsonable(value)
    if not isinstance(config, dict):
        raise ValueError("guided run config must serialize to an object")
    normalized = dict(config)
    normalized.pop("resume", None)
    normalized.pop("overwrite", None)
    return normalized


def validate_guided_run_resume(
    layout: GuidedRunLayout,
    *,
    run_config: Any,
    runtime_data: dict[str, Any] | None = None,
) -> None:
    """Fail closed when a resume request changes its data/model contract."""

    if not getattr(run_config, "resume", False):
        return
    try:
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise GuidedResumeContractError(
            f"cannot validate guided resume manifest: {layout.manifest}"
        ) from exc
    previous_config = manifest.get("config")
    if not isinstance(previous_config, dict):
        raise GuidedResumeContractError("guided resume manifest has no config object")
    if _normalized_run_config(previous_config) != _normalized_run_config(run_config):
        raise GuidedResumeContractError(
            "guided resume config does not match the original run"
        )
    if runtime_data is None:
        return
    previous_runtime = manifest.get("runtime")
    previous_data = (
        previous_runtime.get("data") if isinstance(previous_runtime, dict) else None
    )
    if not isinstance(previous_data, dict):
        raise GuidedResumeContractError(
            "guided resume manifest has no runtime data contract"
        )
    contract_keys = (
        "catalog_build_id",
        "catalog_digest",
        "guide_max_units",
        "guide_max_boundaries",
        "guide_boundary_num_queries",
        "guide_transition_num_queries",
        "guides_per_batch",
        "queries_per_guide",
        "remainder_strategy",
        "gradient_accumulation_steps",
    )
    expected = _jsonable(runtime_data)
    for key in contract_keys:
        if previous_data.get(key) != expected.get(key):
            raise GuidedResumeContractError(
                f"guided resume runtime data contract changed for {key!r}: "
                f"previous={previous_data.get(key)!r}, current={expected.get(key)!r}"
            )


def write_guided_run_manifest(
    layout: GuidedRunLayout,
    *,
    run_config: Any,
    status: str,
    error: BaseException | None = None,
    runtime: dict[str, Any] | None = None,
) -> None:
    existing: dict[str, Any] = {}
    if layout.manifest.is_file():
        try:
            loaded = json.loads(layout.manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}

    now = datetime.now(UTC).isoformat()
    manifest = {
        **existing,
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_name": run_config.experiment_name,
        "status": status,
        "started_at": existing.get("started_at", now),
        "updated_at": now,
        "completed_at": now if status in {"completed", "failed"} else None,
        "layout": {
            "root": str(layout.root),
            "checkpoints": str(layout.checkpoints),
            "logs": str(layout.logs),
            "wandb": str(layout.wandb),
            "eval": str(layout.eval),
        },
        "config": _jsonable(run_config),
        "runtime": {
            **existing.get("runtime", {}),
            **({} if runtime is None else _jsonable(runtime)),
        },
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    temporary = layout.manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, layout.manifest)


def init_guided_wandb(
    *,
    layout: GuidedRunLayout,
    run_config: Any,
    resolved_config: Any,
    resuming: bool,
    wandb_module: Any | None = None,
) -> Any | None:
    if not resolved_config.wandb_enabled:
        return None
    wandb_module = (
        importlib.import_module("wandb") if wandb_module is None else wandb_module
    )
    common = {
        "project": resolved_config.project_name,
        "dir": str(layout.wandb),
    }
    if resuming:
        if not layout.wandb_id.is_file():
            raise FileNotFoundError(
                f"W&B resume requires {layout.wandb_id}"
            )
        run = wandb_module.init(
            id=layout.wandb_id.read_text(encoding="utf-8").strip(),
            resume="must",
            **common,
        )
    else:
        run = wandb_module.init(
            name=resolved_config.exp_name,
            config=_jsonable(run_config),
            **common,
        )
        layout.wandb_id.write_text(str(run.id) + "\n", encoding="utf-8")

    define_metric = getattr(run, "define_metric", None)
    if callable(define_metric):
        define_metric("train/optimizer_step")
        define_metric("train/*", step_metric="train/optimizer_step")
        define_metric("batch/*", step_metric="train/optimizer_step")
        define_metric("performance/*", step_metric="train/optimizer_step")
    return run


def finish_guided_wandb(wandb_run: Any | None, *, exit_code: int) -> None:
    if wandb_run is None:
        return
    finish = getattr(wandb_run, "finish", None)
    if callable(finish):
        finish(exit_code=exit_code)
