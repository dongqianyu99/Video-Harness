from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import Any

from openpi.models.guide_pi0_config import GuidePi0Config
from openpi.models.pi0_config import Pi0Config
from openpi.training.guide_weight_loaders import GuidePi0BaseWeightLoader
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig


@dataclasses.dataclass(frozen=True)
class GuidedTrainRunConfig:
    """Standalone configuration for one Guide-conditioned training run."""

    native_config_name: str
    base_params_path: Path
    guided_data: RoboDojoGuidedDataConfig

    experiment_name: str
    checkpoint_dir: Path

    overwrite: bool = False
    resume: bool = False
    gradient_accumulation_steps: int = 1
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.native_config_name, str) or not self.native_config_name.strip():
            raise ValueError("native_config_name must be a non-empty string")
        if not isinstance(self.base_params_path, Path):
            raise ValueError("base_params_path must be a pathlib.Path")
        if not isinstance(self.checkpoint_dir, Path):
            raise ValueError("checkpoint_dir must be a pathlib.Path")
        if self.run_dir is not None and not isinstance(self.run_dir, Path):
            raise ValueError("run_dir must be pathlib.Path or None")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")

        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")

        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume cannot both be true")
        if (
            self.guided_data.gradient_accumulation_steps
            != self.gradient_accumulation_steps
        ):
            raise ValueError(
                "guided_data.gradient_accumulation_steps must match the run config"
            )

    @property
    def effective_global_batch_size(self) -> int:
        """Valid-query capacity per optimizer step in strict drop mode."""

        return self.guided_data.batch_size * self.gradient_accumulation_steps


def _native_config() -> Any:
    return importlib.import_module("openpi.training.config")


def _load_native_config(name: str) -> Any:
    return _native_config().get_config(name)


def make_guide_pi0_config(
    native_model: Pi0Config,
    *,
    guide_boundary_num_queries: int = 8,
    guide_transition_num_queries: int = 4,
) -> GuidePi0Config:
    """Copy every native Pi0Config init field before adding Guide settings."""

    if not isinstance(native_model, Pi0Config):
        raise ValueError(
            "guided training requires a native openpi.models.pi0_config.Pi0Config, "
            f"got {type(native_model).__name__}"
        )
    if not native_model.pi05:
        raise ValueError("guided training requires a native Pi05 configuration")

    lora_variants = {
        field_name: getattr(native_model, field_name)
        for field_name in ("paligemma_variant", "action_expert_variant")
        if isinstance(getattr(native_model, field_name, None), str)
        and "lora" in getattr(native_model, field_name).lower()
    }
    if lora_variants:
        details = ", ".join(f"{name}={value!r}" for name, value in lora_variants.items())
        raise ValueError(
            "guided training requires full-dense Pi0.5 variants; LoRA variants are not supported: "
            + details
        )

    guide_fields = {field.name for field in dataclasses.fields(GuidePi0Config) if field.init}
    native_fields = {field.name for field in dataclasses.fields(native_model) if field.init}
    missing_guide_fields = native_fields - guide_fields
    if missing_guide_fields:
        raise ValueError(
            "GuidePi0Config cannot represent native Pi0Config fields: "
            + ", ".join(sorted(missing_guide_fields))
        )

    kwargs = {name: getattr(native_model, name) for name in native_fields}
    kwargs.update(
        guide_boundary_num_queries=guide_boundary_num_queries,
        guide_transition_num_queries=guide_transition_num_queries,
        guide_resampler_width=1024,
        guide_resampler_num_heads=8,
        guide_resampler_ffn_hidden_dim=None,
    )
    return GuidePi0Config(**kwargs)


def resolve_guided_train_config(run_config: GuidedTrainRunConfig) -> Any:
    """Resolve a standalone guided run into the stock TrainConfig shape."""

    native_config = _load_native_config(run_config.native_config_name)
    if not isinstance(native_config.model, Pi0Config):
        raise ValueError(
            "native config model must be Pi0Config for guided training, "
            f"got {type(native_config.model).__name__}"
        )
    train_config_type = type(native_config)
    if train_config_type.__name__ != "TrainConfig":
        raise ValueError(
            "native config loader must return openpi.training.config.TrainConfig, "
            f"got {train_config_type.__name__}"
        )

    guide_model = make_guide_pi0_config(
        native_config.model,
        guide_boundary_num_queries=(
            run_config.guided_data.guide_boundary_num_queries
        ),
        guide_transition_num_queries=(
            run_config.guided_data.guide_transition_num_queries
        ),
    )
    config_fields = {field.name for field in dataclasses.fields(train_config_type) if field.init}
    kwargs = {name: getattr(native_config, name) for name in config_fields}
    kwargs.update(
        name=f"{native_config.name}_guided",
        exp_name=run_config.experiment_name,
        model=guide_model,
        weight_loader=GuidePi0BaseWeightLoader(str(run_config.base_params_path)),
        checkpoint_dir_override=str(run_config.checkpoint_dir),
        batch_size=run_config.guided_data.batch_size,
        num_workers=run_config.guided_data.num_workers,
        overwrite=run_config.overwrite,
        resume=run_config.resume,
    )
    return train_config_type(**kwargs)
