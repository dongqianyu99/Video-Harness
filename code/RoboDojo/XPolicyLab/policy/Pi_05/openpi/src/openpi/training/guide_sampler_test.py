from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import TaskSampleIndex
from openpi.training.guide_sampler import GuidanceFirstBatchSampler


def _catalog() -> GuideCatalog:
    documents = tuple(
        SimpleNamespace(
            document_id=f"doc-{index}",
            source_episode_index=10 + index,
            task_index=0 if index < 4 else 1,
            task_instruction="task zero" if index < 4 else "task one",
        )
        for index in range(6)
    )
    return GuideCatalog.from_document_catalog(
        SimpleNamespace(catalog_digest="digest", documents=documents)
    )


def _samples() -> TaskSampleIndex:
    return TaskSampleIndex.from_episode_records(
        (
            SimpleNamespace(episode_index=10, task_index=0, dataset_from_index=0, dataset_to_index=8),
            SimpleNamespace(episode_index=20, task_index=1, dataset_from_index=8, dataset_to_index=16),
        ),
        dataset_length=16,
    )


def _groups(batch, q):
    return tuple(batch[index : index + q] for index in range(0, len(batch), q))


def test_guidance_first_sampler_uses_distinct_guides_and_masks_native_tail():
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=_catalog(),
        task_sample_index=_samples(),
        guides_per_batch=4,
        queries_per_guide=3,
        seed=7,
        remainder_strategy="pad_mask",
    )
    batches = list(sampler)
    for batch in batches:
        guide_indices = [group[0].guide_index for group in _groups(batch, 3)]
        assert len(guide_indices) == len(set(guide_indices)) == 4
    assert sampler.stats.total_native_samples == 16
    assert sampler.stats.valid_query_samples == 16
    assert sampler.stats.padded_query_slots == 8


def test_guidance_first_sampler_draws_q_distinct_same_task_samples_and_allows_source():
    catalog = _catalog()
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=catalog,
        task_sample_index=_samples(),
        guides_per_batch=2,
        queries_per_guide=3,
        seed=3,
        remainder_strategy="drop",
    )

    for batch in sampler:
        for group in _groups(batch, 3):
            assert len({sample.sample_index for sample in group}) == 3
            task = catalog.by_guide_index(group[0].guide_index).task_index
            expected = set(range(8) if task == 0 else range(8, 16))
            assert {sample.sample_index for sample in group} <= expected

    source_catalog = GuideCatalog.from_document_catalog(
        SimpleNamespace(
            catalog_digest="source",
            documents=(
                SimpleNamespace(
                    document_id="source-guide",
                    source_episode_index=10,
                    task_index=0,
                    task_instruction="source task",
                ),
            ),
        )
    )
    source_samples = TaskSampleIndex.from_episode_records(
        (
            SimpleNamespace(
                episode_index=10,
                task_index=0,
                dataset_from_index=0,
                dataset_to_index=4,
            ),
        )
    )
    source_sampler = GuidanceFirstBatchSampler(
        guide_catalog=source_catalog,
        task_sample_index=source_samples,
        guides_per_batch=1,
        queries_per_guide=2,
    )
    assert {sample.sample_index for sample in next(iter(source_sampler))} <= set(range(4))


def test_sampler_is_reproducible_and_changes_order_across_epochs():
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=_catalog(),
        task_sample_index=_samples(),
        guides_per_batch=2,
        queries_per_guide=2,
        seed=11,
        remainder_strategy="drop",
    )
    first = list(sampler)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_sampler_keeps_accumulation_blocks_bucket_homogeneous():
    catalog = _catalog()
    buckets = {
        record.guide_index: ("small" if record.guide_index < 4 else "large")
        for record in catalog.records
    }
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=catalog,
        task_sample_index=_samples(),
        guides_per_batch=1,
        queries_per_guide=2,
        seed=5,
        guide_to_bucket=buckets,
        remainder_strategy="pad_mask",
        batch_block_size=2,
    )
    batches = list(sampler)

    for start in range(0, len(batches), 2):
        block = batches[start : start + 2]
        block_buckets = {
            buckets[batch[0].guide_index]
            for batch in block
        }
        assert len(block_buckets) == 1


def test_bucket_count_weighting_gives_each_document_equal_first_batch_marginal():
    documents = tuple(
        SimpleNamespace(
            document_id=f"weighted-{index}",
            source_episode_index=index,
            task_index=0,
            task_instruction="shared task",
        )
        for index in range(12)
    )
    catalog = GuideCatalog.from_document_catalog(
        SimpleNamespace(catalog_digest="weighted", documents=documents)
    )
    task_samples = TaskSampleIndex.from_episode_records(
        (
            SimpleNamespace(
                episode_index=0,
                task_index=0,
                dataset_from_index=0,
                dataset_to_index=20,
            ),
        )
    )
    buckets = {
        record.guide_index: ("small" if record.guide_index < 4 else "large")
        for record in catalog.records
    }
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=catalog,
        task_sample_index=task_samples,
        guides_per_batch=2,
        queries_per_guide=1,
        seed=17,
        guide_to_bucket=buckets,
        remainder_strategy="drop",
    )
    exposure = Counter()
    for epoch in range(1200):
        sampler.set_epoch(epoch)
        exposure.update(sample.guide_index for sample in next(iter(sampler)))

    assert set(exposure) == set(range(12))
    assert max(exposure.values()) / min(exposure.values()) < 1.25


def test_drop_sampler_rejects_unpromoted_small_bucket():
    catalog = _catalog()
    buckets = {
        record.guide_index: ("orphan" if record.guide_index == 0 else "large")
        for record in catalog.records
    }
    with pytest.raises(ValueError, match="promote it to a larger bucket"):
        GuidanceFirstBatchSampler(
            guide_catalog=catalog,
            task_sample_index=_samples(),
            guides_per_batch=2,
            queries_per_guide=1,
            guide_to_bucket=buckets,
            remainder_strategy="drop",
        )


def test_pad_sampler_masks_queries_when_task_pool_is_smaller_than_q():
    documents = tuple(
        SimpleNamespace(
            document_id=f"small-{index}",
            source_episode_index=index,
            task_index=0,
            task_instruction="small task",
        )
        for index in range(2)
    )
    catalog = GuideCatalog.from_document_catalog(
        SimpleNamespace(catalog_digest="small", documents=documents)
    )
    samples = TaskSampleIndex.from_episode_records(
        (
            SimpleNamespace(
                episode_index=0,
                task_index=0,
                dataset_from_index=0,
                dataset_to_index=2,
            ),
        ),
        dataset_length=2,
    )

    with pytest.raises(ValueError, match="fewer than queries_per_guide"):
        GuidanceFirstBatchSampler(
            guide_catalog=catalog,
            task_sample_index=samples,
            guides_per_batch=2,
            queries_per_guide=3,
            remainder_strategy="drop",
        )

    batch = next(
        iter(
            GuidanceFirstBatchSampler(
                guide_catalog=catalog,
                task_sample_index=samples,
                guides_per_batch=2,
                queries_per_guide=3,
                remainder_strategy="pad_mask",
            )
        )
    )
    first_group = batch[:3]
    assert [sample.query_valid for sample in first_group] == [True, True, False]
    assert len({sample.sample_index for sample in first_group[:2]}) == 2
    assert first_group[2].sample_index == first_group[1].sample_index
