"""Inference policy for a trained JAX GuidePi0 checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.guide_inputs import GuideInput
from openpi.policies.policy import Policy
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training.guide_train_config import make_guide_pi0_config


def _require_single_guide(guide: GuideInput) -> None:
    for leaf in jax.tree_util.tree_leaves(guide):
        if not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != 1:
            raise ValueError("Eval GuideInput leaves must have leading G=1")


class GuidedPolicy(Policy):
    """Stock OpenPI transforms with one persistent task-level GuideInput."""

    def __init__(
        self,
        model: Any,
        *,
        rng: Any = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model,
            rng=rng,
            transforms=transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=metadata,
            is_pytorch=False,
        )
        self._sample_guided_actions = nnx_utils.module_jit(model.sample_guided_actions)
        self._guide: GuideInput | None = None

    @property
    def guide(self) -> GuideInput | None:
        return self._guide

    def set_guide(self, guide: GuideInput) -> None:
        if not isinstance(guide, GuideInput):
            raise TypeError(f"guide must be GuideInput, got {type(guide).__name__}")
        _require_single_guide(guide)
        self._guide = guide

    def clear_guide(self) -> None:
        self._guide = None

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        if self._guide is None:
            raise RuntimeError("GuidedPolicy requires set_guide() before infer()")

        inputs = jax.tree.map(lambda value: value, obs)
        inputs = self._input_transform(inputs)
        if "state" not in inputs:
            raise ValueError("GuidedPolicy input is missing state")
        if np.asarray(inputs["state"]).ndim != 1:
            raise ValueError("GuidedPolicy.infer supports exactly one unbatched query")

        grouped_inputs = jax.tree.map(
            lambda value: jnp.asarray(value)[None, None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(grouped_inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_array = jnp.asarray(noise)
            if noise_array.ndim == 2:
                noise_array = noise_array[None, None, ...]
            sample_kwargs["noise"] = noise_array

        start = time.monotonic()
        actions = self._sample_guided_actions(
            sample_rng,
            observation,
            guide=self._guide,
            **sample_kwargs,
        )
        if actions.ndim != 4 or actions.shape[:2] != (1, 1):
            raise ValueError(f"sample_guided_actions must return [1,1,AH,AD], got {actions.shape}")

        outputs = {
            "state": np.asarray(grouped_inputs["state"][0, 0]),
            "actions": np.asarray(actions[0, 0]),
        }
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
        return outputs


def create_trained_guided_policy(
    native_train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
) -> GuidedPolicy:
    """Restore a full GuidePi0 tree while retaining native policy transforms."""

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if (checkpoint_dir / "model.safetensors").exists():
        raise ValueError("GuidedPolicy only supports JAX checkpoints, not model.safetensors")
    params_path = checkpoint_dir / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Guided checkpoint params do not exist: {params_path}")

    guide_config = make_guide_pi0_config(native_train_config.model)
    model = guide_config.load(_model.restore_params(params_path, dtype=jnp.bfloat16))
    data_config = native_train_config.data.create(native_train_config.assets_dirs, guide_config)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    repack = repack_transforms or _transforms.Group()
    return GuidedPolicy(
        model,
        transforms=[
            *repack.inputs,
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=native_train_config.policy_metadata,
    )
