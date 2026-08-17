from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openpi.training.guide_dataset import GuideBoundDataset
from openpi.training.guide_native_dataset import IdentityPreservingTransformedDataset
from openpi.training.guide_native_dataset import transform_dataset_preserving_identity


class _CountingDataset:
    def __init__(self, sample: dict[str, Any]):
        self.sample = sample
        self.calls: list[int] = []

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.calls.append(index)
        return copy.deepcopy(self.sample)


@dataclasses.dataclass(frozen=True)
class _Group:
    inputs: tuple[Any, ...] = ()


@dataclasses.dataclass(frozen=True)
class _DataConfig:
    repo_id: str
    norm_stats: dict[str, Any] | None
    repack_transforms: _Group
    data_transforms: _Group
    model_transforms: _Group
    use_quantile_norm: bool = False
    asset_id: str | None = "test-asset"


@dataclasses.dataclass(frozen=True)
class _ProjectValue:
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"value": data["value"]}


@dataclasses.dataclass(frozen=True)
class _AddTrace:
    name: str

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        output["trace"] = (*output.get("trace", ()), self.name)
        return output


class _FakeNormalize:
    def __init__(self, norm_stats: Any, *, use_quantiles: bool):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        output["value"] = output["value"] * 2.0
        output["normalized_with"] = self.norm_stats
        return output


_FAKE_TRANSFORMS = SimpleNamespace(Normalize=_FakeNormalize)


def _raw_sample() -> dict[str, Any]:
    return {
        "value": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "episode_index": np.asarray(7, dtype=np.int64),
        "task_index": np.asarray(3, dtype=np.int64),
    }


def _data_config(*, real_repo: bool = False) -> _DataConfig:
    return _DataConfig(
        repo_id="real" if real_repo else "fake",
        norm_stats={"value": "native-stats"} if real_repo else None,
        repack_transforms=_Group(inputs=(_ProjectValue(),)),
        data_transforms=_Group(inputs=(_AddTrace("data"),)),
        model_transforms=_Group(inputs=(_AddTrace("model"),)),
    )


def _compose(transforms: tuple[Any, ...]):
    def apply(data: dict[str, Any]) -> dict[str, Any]:
        for transform in transforms:
            data = transform(data)
        return data

    return apply


def _stock_transform(raw_sample: dict[str, Any], config: _DataConfig) -> dict[str, Any]:
    norm_stats = {} if config.norm_stats is None else config.norm_stats
    transforms = (
        *config.repack_transforms.inputs,
        *config.data_transforms.inputs,
        _FakeNormalize(norm_stats, use_quantiles=config.use_quantile_norm),
        *config.model_transforms.inputs,
    )
    return _compose(transforms)(copy.deepcopy(raw_sample))


def _assert_tree_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected) == set(actual)
        for key in expected:
            _assert_tree_equal(expected[key], actual[key])
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(expected) == len(actual)
        for expected_child, actual_child in zip(expected, actual, strict=True):
            _assert_tree_equal(expected_child, actual_child)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    else:
        assert expected == actual


def test_identity_preserving_transform_matches_stock_transform_and_keeps_identity():
    config = _data_config()
    raw = _raw_sample()
    guided_dataset = _CountingDataset(raw)

    stock = _stock_transform(raw, config)
    guided = transform_dataset_preserving_identity(
        guided_dataset,
        config,
        skip_norm_stats=True,
        transforms_module=_FAKE_TRANSFORMS,
    )[0]

    for key in stock:
        _assert_tree_equal(stock[key], guided[key])
    assert guided["episode_index"].item() == 7
    assert guided["task_index"].item() == 3
    assert guided_dataset.calls == [0]


def test_identity_is_removed_by_guide_bound_dataset_after_transforms():
    config = _data_config()
    transformed = transform_dataset_preserving_identity(
        _CountingDataset(_raw_sample()),
        config,
        skip_norm_stats=True,
        transforms_module=_FAKE_TRANSFORMS,
    )

    class _BindingIndex:
        def by_query_episode(self, episode_index: int):
            assert episode_index == 7
            return type(
                "Record",
                (),
                {
                    "binding_index": 0,
                    "task_index": 3,
                },
            )()

    item = GuideBoundDataset(transformed, _BindingIndex())[0]

    assert "episode_index" not in item["query"]
    assert "task_index" not in item["query"]
    assert item["guide_binding_index"].item() == 0


def test_identity_metadata_is_not_normalized_and_norm_stats_are_required():
    config = _data_config(real_repo=True)
    transformed = transform_dataset_preserving_identity(
        _CountingDataset(_raw_sample()),
        config,
        transforms_module=_FAKE_TRANSFORMS,
    )[0]

    np.testing.assert_array_equal(
        transformed["value"],
        np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
    )
    assert transformed["episode_index"].item() == 7

    missing_stats = dataclasses.replace(_data_config(real_repo=True), norm_stats=None)
    with pytest.raises(ValueError, match=r"asset_id|Normalization stats"):
        transform_dataset_preserving_identity(
            _CountingDataset(_raw_sample()),
            missing_stats,
            transforms_module=_FAKE_TRANSFORMS,
        )


@pytest.mark.parametrize(
    "field_value",
    [
        np.asarray([7], dtype=np.int64),
        np.asarray(np.bool_(1), dtype=np.bool_),
        np.asarray(7.0, dtype=np.float32),
    ],
)
def test_identity_fields_must_be_nonnegative_integer_scalars(field_value):
    sample = _raw_sample()
    sample["episode_index"] = field_value

    with pytest.raises(ValueError, match="episode_index"):
        transform_dataset_preserving_identity(
            _CountingDataset(sample),
            _data_config(),
            skip_norm_stats=True,
            transforms_module=_FAKE_TRANSFORMS,
        )[0]


def test_identity_field_missing_fails_before_transform():
    sample = _raw_sample()
    sample.pop("task_index")
    dataset = _CountingDataset(sample)

    with pytest.raises(ValueError, match="task_index"):
        transform_dataset_preserving_identity(
            dataset,
            _data_config(),
            skip_norm_stats=True,
            transforms_module=_FAKE_TRANSFORMS,
        )[0]

    assert dataset.calls == [0]


def test_constructor_rejects_duplicate_identity_fields():
    with pytest.raises(ValueError, match="unique"):
        IdentityPreservingTransformedDataset(
            _CountingDataset(_raw_sample()),
            (),
            identity_fields=("episode_index", "episode_index"),
        )
