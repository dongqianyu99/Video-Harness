from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training.guide_collator import GuidanceBatchCollator
from openpi.training.guide_collator_test import _guide
from openpi.training.guide_data_loader import GuidedDataLoader
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import GuidedDataset
from openpi.training.guide_dataset import TaskSampleIndex
from openpi.training.guide_sampler import GuidanceFirstBatchSampler


def _catalog():
    documents = tuple(
        SimpleNamespace(
            document_id=f"doc-{index}",
            source_episode_index=10 + index,
            task_index=index,
            task_instruction=f"task {index}",
        )
        for index in range(2)
    )
    return GuideCatalog.from_document_catalog(
        SimpleNamespace(catalog_digest="digest", documents=documents)
    )


class _NativeDataset:
    def __init__(self):
        self.samples = [
            {
                "image": {
                    key: np.full((2, 2, 3), index, dtype=np.float32)
                    for key in _model.IMAGE_KEYS
                },
                "image_mask": {
                    key: np.asarray(np.bool_(1), dtype=np.bool_) for key in _model.IMAGE_KEYS
                },
                "state": np.asarray([index, index + 1], dtype=np.float32),
                "actions": np.full((50, 32), index, dtype=np.float32),
                "episode_index": np.asarray(10 if index < 4 else 11, dtype=np.int64),
                "task_index": np.asarray(0 if index < 4 else 1, dtype=np.int64),
            }
            for index in range(8)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _loader(num_batches=3):
    catalog = _catalog()
    task_samples = TaskSampleIndex.from_episode_records(
        (
            SimpleNamespace(episode_index=10, task_index=0, dataset_from_index=0, dataset_to_index=4),
            SimpleNamespace(episode_index=11, task_index=1, dataset_from_index=4, dataset_to_index=8),
        )
    )
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=catalog,
        task_sample_index=task_samples,
        guides_per_batch=2,
        queries_per_guide=2,
        seed=9,
        remainder_strategy="drop",
    )
    def resolver(record):
        return _guide(record.guide_index)
    return GuidedDataLoader(
        GuidedDataset(_NativeDataset(), catalog, task_samples),
        batch_sampler=sampler,
        collator=GuidanceBatchCollator(
            guide_catalog=catalog,
            guide_input_resolver=resolver,
            guides_per_batch=2,
            queries_per_guide=2,
        ),
        num_batches=num_batches,
        guide_catalog=catalog,
        data_config=object(),
    )


def test_loader_runs_guidance_first_pipeline_across_epochs():
    loader = _loader(num_batches=3)
    batches = list(loader)

    assert len(batches) == 3
    assert loader.epoch == 1
    assert loader.guide_catalog.catalog_digest == "digest"
    for batch in batches:
        assert validate_guide_conditioned_batch(batch) == (2, 2)
        assert batch.actions.shape == (2, 2, 50, 32)
        assert all(not isinstance(leaf, str) for leaf in jax.tree_util.tree_leaves(batch))


def test_loader_honors_zero_num_batches_and_exposes_native_data_config():
    loader = _loader(num_batches=0)
    assert list(loader) == []
    assert loader.data_config() is not None


def test_loader_rejects_invalid_worker_settings():
    loader = _loader(num_batches=1)
    with pytest.raises(ValueError, match="num_workers"):
        GuidedDataLoader(
            loader.torch_loader.dataset,
            batch_sampler=loader.batch_sampler,
            collator=loader.torch_loader.collate_fn,
            num_workers=-1,
        )
