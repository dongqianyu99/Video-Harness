from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.guide_materializer import materialize_guide
from openpi.training import guide_data_loader as _guide_data_loader
from openpi.training.guide_cache import ConstantResolverFactory
from openpi.training.guide_cache import ProcessLocalGuideResolver
from openpi.training.guide_collator import SingleGuideBatchCollator
from openpi.training.guide_data_loader import GuidedDataLoader
from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.guide_dataset import GuideBoundDataset
from openpi.training.guide_sampler import HomogeneousBindingBatchSampler
from openpi.training.guide_sampler import QueryEpisodeRange
from openpi.training.guide_sampler import build_binding_to_sample_indices


@dataclass(frozen=True)
class _SupportBinding:
    query_episode_index: int
    support_episode_index: int
    task_index: int
    support_document_id: str


@dataclass(frozen=True)
class _FrameRef:
    binding_index: int
    frame_index: int


@dataclass(frozen=True)
class _PlanUnit:
    before_slot: int
    after_slot: int
    transition_text: str


@dataclass(frozen=True)
class _GuidePlan:
    frames: tuple[_FrameRef, ...]
    units: tuple[_PlanUnit, ...]


def _make_bindings() -> tuple[_SupportBinding, ...]:
    return (
        _SupportBinding(10, 11, 4, "document-0"),
        _SupportBinding(20, 21, 5, "document-1"),
    )


def _make_binding_index() -> GuideBindingIndex:
    return GuideBindingIndex.from_bindings(list(_make_bindings()))


def _make_ranges() -> tuple[QueryEpisodeRange, ...]:
    return (
        QueryEpisodeRange(10, 0, 4),
        QueryEpisodeRange(20, 4, 8),
    )


def _make_plans() -> dict[int, _GuidePlan]:
    return {
        binding_index: _GuidePlan(
            frames=tuple(
                _FrameRef(binding_index, frame_index)
                for frame_index in range(2)
            ),
            units=(
                _PlanUnit(
                    before_slot=0,
                    after_slot=1,
                    transition_text=f"guide-{binding_index}",
                ),
            ),
        )
        for binding_index in (0, 1)
    }


def _make_native_sample(
    sample_index: int,
    *,
    episode_index: int,
    task_index: int,
    include_prompt: bool = True,
) -> dict[str, Any]:
    sample = {
        "image": {
            key: np.full(
                (2, 2, 3),
                sample_index + image_index,
                dtype=np.float32,
            )
            for image_index, key in enumerate(_model.IMAGE_KEYS)
        },
        "image_mask": {
            key: np.asarray(image_index % 2 == 0, dtype=np.bool_)
            for image_index, key in enumerate(_model.IMAGE_KEYS)
        },
        "state": np.asarray(
            [sample_index, sample_index + 1, sample_index + 2, sample_index + 3],
            dtype=np.float32,
        ),
        "actions": np.full((50, 32), sample_index, dtype=np.float32),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "task_index": np.asarray(task_index, dtype=np.int64),
    }

    if include_prompt:
        sample.update(
            {
                "tokenized_prompt": np.asarray([1, 2, 3, 0], dtype=np.int32),
                "tokenized_prompt_mask": np.asarray(
                    [True, True, True, False],
                    dtype=np.bool_,
                ),
            }
        )

    return sample


class _RecordingNativeDataset:
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples
        self.accessed_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accessed_indices.append(index)
        return copy.deepcopy(self.samples[index])


class _RecordingFrameDecoder:
    def __init__(self):
        self.calls: list[_FrameRef] = []

    def __call__(self, frame_ref: _FrameRef) -> np.ndarray:
        self.calls.append(frame_ref)
        return np.full(
            (2, 4, 3),
            30 + frame_ref.binding_index * 10 + frame_ref.frame_index,
            dtype=np.uint8,
        )


class _RecordingTokenizer:
    def __init__(self):
        self.calls: list[str] = []

    def tokenize_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(text)
        binding_index = int(text.rsplit("-", maxsplit=1)[-1])
        return (
            np.asarray([100 + binding_index, 7, 0, 0], dtype=np.int32),
            np.asarray([True, True, False, False], dtype=np.bool_),
        )


class _MaterializingResolver:
    def __init__(self, plans: dict[int, _GuidePlan]):
        self.plans = plans
        self.frame_decoder = _RecordingFrameDecoder()
        self.tokenizer = _RecordingTokenizer()
        self.calls: list[int] = []

    def __call__(self, record) -> Any:
        binding_index = record.binding_index
        self.calls.append(binding_index)
        return materialize_guide(
            self.plans[binding_index],
            frame_decoder=self.frame_decoder,
            tokenizer=self.tokenizer,
            config=GuideMaterializerConfig(
                max_frames=2,
                max_units=1,
                max_text_tokens=4,
            ),
        )


def _make_samples(*, include_prompt: bool = True) -> list[dict[str, Any]]:
    return [
        _make_native_sample(
            sample_index=index,
            episode_index=10 if index < 4 else 20,
            task_index=4 if index < 4 else 5,
            include_prompt=include_prompt,
        )
        for index in range(8)
    ]


def _make_loader(
    *,
    num_batches: int | None = 6,
    include_prompt: bool = True,
) -> tuple[
    GuidedDataLoader,
    _RecordingNativeDataset,
    _MaterializingResolver,
    dict[int, _GuidePlan],
]:
    samples = _make_samples(include_prompt=include_prompt)
    native_dataset = _RecordingNativeDataset(samples)
    binding_index = _make_binding_index()
    binding_to_samples = build_binding_to_sample_indices(
        _make_ranges(),
        binding_index=binding_index,
    )
    sampler = HomogeneousBindingBatchSampler(
        binding_to_samples,
        binding_index=binding_index,
        batch_size=2,
        seed=23,
    )
    bound_dataset = GuideBoundDataset(native_dataset, binding_index)
    plans = _make_plans()
    resolver = _MaterializingResolver(plans)
    collator = SingleGuideBatchCollator(
        binding_index=binding_index,
        guide_input_resolver=resolver,
    )
    return (
        GuidedDataLoader(
            bound_dataset,
            batch_sampler=sampler,
            collator=collator,
            num_batches=num_batches,
        ),
        native_dataset,
        resolver,
        plans,
    )


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected) == set(actual)
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_child, actual_child in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_child, actual_child)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    else:
        assert expected == actual


def test_guided_loader_runs_full_single_guide_pipeline_across_epochs():
    loader, native_dataset, resolver, plans = _make_loader(num_batches=6)
    samples_before = copy.deepcopy(native_dataset.samples)
    plans_before = copy.deepcopy(plans)

    batches = list(loader)

    assert len(loader) == 6
    assert len(batches) == 6
    assert all(isinstance(batch, GuideConditionedBatch) for batch in batches)
    assert loader.epoch == 1
    assert len(resolver.calls) == 6
    assert len(resolver.frame_decoder.calls) == 12
    assert len(resolver.tokenizer.calls) == 6

    seen_binding_indices = set()
    for batch in batches:
        groups, queries = validate_guide_conditioned_batch(batch)
        assert (groups, queries) == (1, 2)
        assert batch.actions.shape == (1, 2, 50, 32)
        assert batch.observation.state.shape == (1, 2, 4)

        guide_binding_marker = int(np.asarray(batch.guide.text_tokens[0, 0, 0]))
        binding_index = guide_binding_marker - 100
        assert binding_index in (0, 1)
        seen_binding_indices.add(binding_index)

        state_indices = np.asarray(batch.observation.state[0, :, 0], dtype=np.int32)
        action_indices = np.asarray(batch.actions[0, :, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(state_indices, action_indices)
        if binding_index == 0:
            assert np.all((state_indices >= 0) & (state_indices < 4))
        else:
            assert np.all((state_indices >= 4) & (state_indices < 8))

        assert all(not isinstance(leaf, str) for leaf in jax.tree_util.tree_leaves(batch))

    assert seen_binding_indices == {0, 1}
    assert native_dataset.accessed_indices
    assert all(index in range(8) for index in native_dataset.accessed_indices)
    _assert_nested_equal(samples_before, native_dataset.samples)
    assert plans == plans_before

    for offset in range(0, len(resolver.frame_decoder.calls), 2):
        first, second = resolver.frame_decoder.calls[offset : offset + 2]
        assert first.frame_index == 0
        assert second.frame_index == 1
        assert first.binding_index == second.binding_index


def test_guided_loader_keeps_optional_prompt_fields_as_none():
    loader, _, _, _ = _make_loader(num_batches=1, include_prompt=False)

    batch = next(iter(loader))

    assert batch.observation.tokenized_prompt is None
    assert batch.observation.tokenized_prompt_mask is None
    assert batch.observation.token_ar_mask is None
    assert batch.observation.token_loss_mask is None


def test_guided_loader_num_batches_stops_exactly():
    loader, _, resolver, _ = _make_loader(num_batches=3)

    assert len(loader) == 3
    assert len(list(loader)) == 3
    assert len(resolver.calls) == 3


def test_guided_loader_configures_workers_and_rejects_negative_workers_or_empty_sampler():
    loader, _, _, _ = _make_loader(num_batches=1)

    worker_loader = GuidedDataLoader(
        loader._data_loader.dataset,  # noqa: SLF001
        batch_sampler=loader._data_loader.batch_sampler,  # noqa: SLF001
        collator=loader._data_loader.collate_fn,  # noqa: SLF001
        num_workers=1,
        prefetch_factor=3,
    )
    assert worker_loader.torch_loader.num_workers == 1
    assert worker_loader.torch_loader.prefetch_factor == 3

    with pytest.raises(ValueError, match="num_workers"):
        GuidedDataLoader(
            loader._data_loader.dataset,  # noqa: SLF001
            batch_sampler=loader._data_loader.batch_sampler,  # noqa: SLF001
            collator=loader._data_loader.collate_fn,  # noqa: SLF001
            num_workers=-1,
        )

    class _EmptySampler:
        batch_size = 2

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def set_epoch(self, _epoch):
            pass

    with pytest.raises(ValueError, match="empty"):
        GuidedDataLoader(
            loader._data_loader.dataset,  # noqa: SLF001
            batch_sampler=_EmptySampler(),
            collator=loader._data_loader.collate_fn,  # noqa: SLF001
        )


def test_guided_loader_spawn_worker_returns_a_complete_batch():
    loader, _, _, _ = _make_loader(num_batches=1)
    process_local_resolver = ProcessLocalGuideResolver(
        resolver_factory=ConstantResolverFactory(_MaterializingResolver(_make_plans())),
        max_entries=1,
    )
    worker_collator = SingleGuideBatchCollator(
        binding_index=_make_binding_index(),
        guide_input_resolver=process_local_resolver,
    )
    worker_loader = GuidedDataLoader(
        loader._data_loader.dataset,  # noqa: SLF001
        batch_sampler=loader._data_loader.batch_sampler,  # noqa: SLF001
        collator=worker_collator,
        num_batches=1,
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=False,
    )

    batch = next(iter(worker_loader))

    assert validate_guide_conditioned_batch(batch) == (1, 2)


def test_guided_loader_rejects_mixed_binding_batch_from_sampler():
    samples = _make_samples()
    native_dataset = _RecordingNativeDataset(samples)
    binding_index = _make_binding_index()
    bound_dataset = GuideBoundDataset(native_dataset, binding_index)
    resolver = _MaterializingResolver(_make_plans())
    collator = SingleGuideBatchCollator(
        binding_index=binding_index,
        guide_input_resolver=resolver,
    )

    class _MixedBindingSampler:
        batch_size = 2

        def __iter__(self):
            yield [0, 4]

        def __len__(self):
            return 1

        def set_epoch(self, _epoch):
            pass

    loader = GuidedDataLoader(
        bound_dataset,
        batch_sampler=_MixedBindingSampler(),
        collator=collator,
        num_batches=1,
    )

    with pytest.raises(ValueError, match=r"single-guide|binding"):
        next(iter(loader))


def test_guided_loader_rejects_non_positive_num_batches():
    loader, _, _, _ = _make_loader(num_batches=0)

    assert len(loader) == 0
    assert list(loader) == []

    with pytest.raises(ValueError, match="num_batches"):
        _make_loader(num_batches=-1)


def test_guided_loader_does_not_mutate_plan_objects_when_resolving():
    loader, _, resolver, plans = _make_loader(num_batches=1)
    before = copy.deepcopy(plans)

    next(iter(loader))

    assert plans == before
    assert resolver.calls in ([0], [1])


def test_guided_loader_output_is_not_a_stock_observation_tuple():
    loader, _, _, _ = _make_loader(num_batches=1)

    batch = next(iter(loader))

    assert isinstance(batch, GuideConditionedBatch)
    assert not isinstance(batch, tuple)


def test_guided_loader_requires_native_data_config_for_checkpoint_assets():
    loader, _, _, _ = _make_loader(num_batches=1)

    with pytest.raises(ValueError, match="no native data_config"):
        loader.data_config()

    native_config = object()
    loader._data_config = native_config  # noqa: SLF001
    assert loader.data_config() is native_config


def test_device_prefetch_is_bounded_and_preserves_order(monkeypatch):
    batches = [object(), object(), object()]
    calls = []

    def fake_device_put(batch, sharding):
        calls.append((batch, sharding))
        return ("device", batch)

    monkeypatch.setattr(_guide_data_loader.jax, "device_put", fake_device_put)
    iterator = _guide_data_loader.prefetch_guided_batches(
        iter(batches),
        sharding="sharding",
        size=2,
    )

    first = next(iterator)
    assert first == ("device", batches[0])
    assert len(calls) == 2
    assert list(iterator) == [("device", batches[1]), ("device", batches[2])]
    assert [batch for batch, _ in calls] == batches
