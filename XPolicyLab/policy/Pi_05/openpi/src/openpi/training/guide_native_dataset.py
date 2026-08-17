from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import importlib
from typing import Any

import numpy as np


def _copy_identity_scalar(sample: Mapping[str, Any], field: str) -> np.ndarray:
    if field not in sample:
        raise ValueError(f"native sample is missing identity field {field!r}")

    value = np.asarray(sample[field])
    if value.ndim != 0:
        raise ValueError(
            f"native sample identity field {field!r} must be scalar, got shape {value.shape}"
        )
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(
            f"native sample identity field {field!r} must have an integer dtype, got {value.dtype}"
        )

    integer_value = int(value.item())
    if integer_value < 0:
        raise ValueError(
            f"native sample identity field {field!r} must be non-negative, got {integer_value}"
        )

    return np.array(value, copy=True)


def _compose(transforms: Sequence[Any]):
    def apply(data: Any) -> Any:
        for transform in transforms:
            data = transform(data)
        return data

    return apply


def _load_transforms_module() -> Any:
    return importlib.import_module("openpi.transforms")


class IdentityPreservingTransformedDataset:
    """Apply stock input transforms while preserving query identity metadata."""

    def __init__(
        self,
        dataset: Any,
        transforms: Sequence[Any],
        *,
        identity_fields: Sequence[str] = ("episode_index", "task_index"),
    ):
        if not identity_fields:
            raise ValueError("identity_fields must not be empty")
        if len(set(identity_fields)) != len(identity_fields):
            raise ValueError("identity_fields must be unique")
        if any(not isinstance(field, str) or not field for field in identity_fields):
            raise ValueError("identity_fields must contain non-empty strings")

        self._dataset = dataset
        self._transform = _compose(tuple(transforms))
        self._identity_fields = tuple(identity_fields)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: Any) -> dict[str, Any]:
        raw_sample = self._dataset[index]
        if not isinstance(raw_sample, Mapping):
            raise ValueError(
                f"native dataset sample must be a mapping, got {type(raw_sample).__name__}"
            )

        identity = {
            field: _copy_identity_scalar(raw_sample, field)
            for field in self._identity_fields
        }

        transformed_input = copy.deepcopy(dict(raw_sample))
        for field in self._identity_fields:
            transformed_input.pop(field, None)

        transformed = self._transform(transformed_input)
        if not isinstance(transformed, Mapping):
            raise ValueError(
                "native transforms must return a mapping, "
                f"got {type(transformed).__name__}"
            )

        result = dict(transformed)
        for field, value in identity.items():
            result.pop(field, None)
            result[field] = value
        return result


def transform_dataset_preserving_identity(
    dataset: Any,
    data_config: Any,
    *,
    skip_norm_stats: bool = False,
    transforms_module: Any | None = None,
) -> IdentityPreservingTransformedDataset:
    """Build the stock transform sequence with two host-side identity leaves."""

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found for guided dataset. "
                f"asset_id={data_config.asset_id!r}; "
                "provide the native RoboDojo norm-stats asset or use "
                "skip_norm_stats=True only for fake/unit tests."
            )
        norm_stats = data_config.norm_stats

    transforms_module = _load_transforms_module() if transforms_module is None else transforms_module
    transforms = (
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        transforms_module.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    )
    return IdentityPreservingTransformedDataset(dataset, transforms)
