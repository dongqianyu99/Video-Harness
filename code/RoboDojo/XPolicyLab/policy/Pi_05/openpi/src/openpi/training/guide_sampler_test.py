from dataclasses import FrozenInstanceError
from dataclasses import dataclass

import pytest

from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.guide_sampler import BindingBatchStats
from openpi.training.guide_sampler import GroupedBindingBatchSampler
from openpi.training.guide_sampler import HomogeneousBindingBatchSampler
from openpi.training.guide_sampler import QueryEpisodeRange
from openpi.training.guide_sampler import build_binding_to_sample_indices


@dataclass(frozen=True)
class _SupportBinding:
    query_episode_index: int
    support_episode_index: int
    task_index: int
    support_document_id: str


def _make_index() -> GuideBindingIndex:
    return GuideBindingIndex.from_bindings(
        [
            _SupportBinding(20, 21, 5, "document-1"),
            _SupportBinding(10, 11, 4, "document-0"),
        ]
    )


def _make_ranges() -> tuple[QueryEpisodeRange, ...]:
    return (
        QueryEpisodeRange(20, 5, 9),
        QueryEpisodeRange(10, 0, 5),
        QueryEpisodeRange(999, 9, 11),
    )


def _make_sample_indices() -> dict[int, tuple[int, ...]]:
    return {
        1: (5, 6, 7, 8),
        0: (0, 1, 2, 3, 4),
    }


def test_build_binding_to_sample_indices_uses_half_open_ranges_and_stable_binding_keys():
    result = build_binding_to_sample_indices(
        _make_ranges(),
        binding_index=_make_index(),
    )

    assert dict(result) == {
        0: (0, 1, 2, 3, 4),
        1: (5, 6, 7, 8),
    }


def test_range_builder_is_independent_of_input_order_and_ignores_extra_episode():
    ranges = list(reversed(_make_ranges()))
    original = list(ranges)

    result = build_binding_to_sample_indices(
        ranges,
        binding_index=_make_index(),
    )

    assert set(result) == {0, 1}
    assert ranges == original


def test_range_builder_returns_immutable_groups():
    result = build_binding_to_sample_indices(
        _make_ranges(),
        binding_index=_make_index(),
    )

    with pytest.raises(TypeError):
        result[0] = (99,)  # type: ignore[index]


def test_range_builder_rejects_missing_query_episode():
    ranges = (QueryEpisodeRange(10, 0, 2),)

    with pytest.raises(ValueError, match=r"missing|query_episode_index"):
        build_binding_to_sample_indices(ranges, binding_index=_make_index())


def test_range_builder_rejects_duplicate_episode_ranges():
    ranges = (
        QueryEpisodeRange(10, 0, 2),
        QueryEpisodeRange(10, 2, 4),
        QueryEpisodeRange(20, 4, 6),
    )

    with pytest.raises(ValueError, match=r"duplicate|episode_index"):
        build_binding_to_sample_indices(ranges, binding_index=_make_index())


def test_range_builder_rejects_overlapping_ranges():
    ranges = (
        QueryEpisodeRange(10, 0, 3),
        QueryEpisodeRange(20, 2, 5),
    )

    with pytest.raises(ValueError, match="overlap"):
        build_binding_to_sample_indices(ranges, binding_index=_make_index())


@pytest.mark.parametrize(
    "entry",
    [
        QueryEpisodeRange(10, 0, 0),
        QueryEpisodeRange(10, 3, 2),
        QueryEpisodeRange(10, -1, 2),
        QueryEpisodeRange(-1, 0, 2),
    ],
)
def test_range_builder_rejects_empty_negative_or_reversed_ranges(entry):
    with pytest.raises(ValueError, match=r"range|non-negative|episode"):
        build_binding_to_sample_indices((entry,), binding_index=_make_index())


def test_sampler_yields_homogeneous_fixed_size_batches_and_reports_tail():
    sampler = HomogeneousBindingBatchSampler(
        _make_sample_indices(),
        binding_index=_make_index(),
        batch_size=2,
        seed=7,
    )

    batches = list(sampler)

    assert len(sampler) == 4
    assert len(batches) == 4
    assert all(len(batch) == 2 for batch in batches)
    assert all(set(batch) <= set(range(5)) or set(batch) <= set(range(5, 9)) for batch in batches)
    used_values = {value for batch in batches for value in batch}
    assert len(used_values) == 8
    assert used_values <= set(range(5)) | set(range(5, 9))
    assert sampler.stats == (
        BindingBatchStats(0, 5, 4, 1, 2),
        BindingBatchStats(1, 4, 4, 0, 2),
    )
    assert sampler.total_samples == 9
    assert sampler.used_samples == 8
    assert sampler.dropped_samples == 1


def test_sampler_mapping_order_does_not_change_output():
    first = HomogeneousBindingBatchSampler(
        {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)},
        binding_index=_make_index(),
        batch_size=2,
        seed=11,
    )
    second = HomogeneousBindingBatchSampler(
        {1: (4, 5, 6, 7), 0: (0, 1, 2, 3)},
        binding_index=_make_index(),
        batch_size=2,
        seed=11,
    )

    assert list(first) == list(second)


def test_sampler_seed_and_epoch_are_reproducible():
    first = HomogeneousBindingBatchSampler(
        _make_sample_indices(),
        binding_index=_make_index(),
        batch_size=2,
        seed=13,
    )
    second = HomogeneousBindingBatchSampler(
        _make_sample_indices(),
        binding_index=_make_index(),
        batch_size=2,
        seed=13,
    )

    epoch_zero = list(first)
    assert epoch_zero == list(first)

    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != epoch_zero
    assert first.epoch == 1


def test_sampler_copies_input_sequences():
    indices = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    sampler = HomogeneousBindingBatchSampler(
        indices,
        binding_index=_make_index(),
        batch_size=2,
    )
    expected = list(sampler)

    indices[0].clear()
    indices[1][:] = [99, 98, 97, 96]

    assert list(sampler) == expected


def test_sampler_stats_are_immutable():
    sampler = HomogeneousBindingBatchSampler(
        _make_sample_indices(),
        binding_index=_make_index(),
        batch_size=2,
    )

    with pytest.raises(FrozenInstanceError):
        sampler.stats[0].used_samples = 0


def test_sampler_set_epoch_does_not_change_automatically_during_iteration():
    sampler = HomogeneousBindingBatchSampler(
        _make_sample_indices(),
        binding_index=_make_index(),
        batch_size=2,
        seed=17,
    )

    list(sampler)

    assert sampler.epoch == 0


@pytest.mark.parametrize(
    "bad_mapping",
    [
        {},
        {0: ()},
        {0: (0,)},
        {0: (0, 1, 1, 2)},
        {0: (0, 1, 2, 3), 1: (3, 4, 5, 6)},
        {99: (0, 1, 2, 3)},
    ],
)
def test_sampler_rejects_invalid_groups(bad_mapping):
    with pytest.raises(ValueError, match=r"binding|sample|batch|unknown"):
        HomogeneousBindingBatchSampler(
            bad_mapping,
            binding_index=_make_index(),
            batch_size=2,
        )


def test_sampler_rejects_invalid_batch_size_seed_and_epoch():
    with pytest.raises(ValueError, match="batch_size"):
        HomogeneousBindingBatchSampler(
            _make_sample_indices(),
            binding_index=_make_index(),
            batch_size=0,
        )

    with pytest.raises(ValueError, match="batch_size"):
        HomogeneousBindingBatchSampler(
            _make_sample_indices(),
            binding_index=_make_index(),
            batch_size=True,
        )

    with pytest.raises(ValueError, match="seed"):
        HomogeneousBindingBatchSampler(
            _make_sample_indices(),
            binding_index=_make_index(),
            batch_size=2,
            seed=True,
        )

    sampler = HomogeneousBindingBatchSampler(
        _make_sample_indices(),
        binding_index=_make_index(),
        batch_size=2,
    )
    with pytest.raises(ValueError, match="epoch"):
        sampler.set_epoch(-1)


def test_grouped_sampler_mixes_distinct_guides_and_tasks_without_changing_sample_marginal():
    index = GuideBindingIndex.from_bindings(
        [
            _SupportBinding(10, 11, 0, "doc-a"),
            _SupportBinding(20, 21, 1, "doc-b"),
            _SupportBinding(30, 31, 0, "doc-c"),
            _SupportBinding(40, 41, 2, "doc-d"),
        ]
    )
    mapping = {
        0: tuple(range(8)),
        1: tuple(range(8, 16)),
        2: tuple(range(16, 24)),
        3: tuple(range(24, 32)),
    }
    sampler = GroupedBindingBatchSampler(
        mapping,
        binding_index=index,
        guides_per_batch=2,
        queries_per_guide=2,
        seed=19,
    )

    batches = list(sampler)

    assert len(batches) == len(sampler) == 8
    assert all(len(batch) == 4 for batch in batches)
    assert {sample for batch in batches for sample in batch} == set(range(32))
    for batch in batches:
        first_group = batch[:2]
        second_group = batch[2:]
        first_binding = next(binding for binding, samples in mapping.items() if first_group[0] in samples)
        second_binding = next(binding for binding, samples in mapping.items() if second_group[0] in samples)
        assert first_binding != second_binding
        assert set(first_group) <= set(mapping[first_binding])
        assert set(second_group) <= set(mapping[second_binding])
        assert (
            index.by_binding_index(first_binding).support_document_id
            != index.by_binding_index(second_binding).support_document_id
        )

    assert sampler.stats.used_query_groups == 16
    assert sampler.stats.dropped_query_groups == 0
    assert sampler.stats.mixed_task_batches > 0


def test_grouped_sampler_is_reproducible_and_changes_order_by_epoch():
    index = GuideBindingIndex.from_bindings(
        [
            _SupportBinding(10, 11, 0, "doc-a"),
            _SupportBinding(20, 21, 1, "doc-b"),
        ]
    )
    mapping = {0: tuple(range(8)), 1: tuple(range(8, 16))}
    first = GroupedBindingBatchSampler(
        mapping,
        binding_index=index,
        guides_per_batch=2,
        queries_per_guide=2,
        seed=3,
    )
    second = GroupedBindingBatchSampler(
        dict(reversed(tuple(mapping.items()))),
        binding_index=index,
        guides_per_batch=2,
        queries_per_guide=2,
        seed=3,
    )

    epoch_zero = list(first)
    assert epoch_zero == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != epoch_zero
