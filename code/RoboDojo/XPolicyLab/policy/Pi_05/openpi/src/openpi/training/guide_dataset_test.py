from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import GuidedDataset
from openpi.training.guide_dataset import GuidedSampleIndex
from openpi.training.guide_dataset import TaskSampleIndex


def _document(
    document_id: str,
    *,
    episode: int,
    task: int,
    instruction: str | None = None,
):
    return SimpleNamespace(
        document_id=document_id,
        source_episode_index=episode,
        task_index=task,
        task_instruction=instruction or f"task {task}",
    )


def _catalog(*documents) -> GuideCatalog:
    return GuideCatalog.from_document_catalog(
        SimpleNamespace(catalog_digest="digest", documents=documents)
    )


def test_catalog_stably_numbers_documents_and_round_trips_through_pickle():
    catalog = _catalog(
        _document("z", episode=20, task=1),
        _document("b", episode=11, task=0),
        _document("a", episode=11, task=0),
    )

    assert [record.document_id for record in catalog.records] == ["a", "b", "z"]
    assert [record.guide_index for record in catalog.records] == [0, 1, 2]
    assert [record.document_id for record in catalog.records_for_task(0)] == ["a", "b"]
    assert pickle.loads(pickle.dumps(catalog)).records == catalog.records


def test_catalog_rejects_duplicate_documents_and_instruction_drift():
    with pytest.raises(ValueError, match="duplicate document_id"):
        _catalog(
            _document("same", episode=1, task=0),
            _document("same", episode=2, task=0),
        )
    with pytest.raises(ValueError, match="inconsistent task instructions"):
        _catalog(
            _document("a", episode=1, task=0, instruction="first"),
            _document("b", episode=2, task=0, instruction="second"),
        )


def test_task_sample_index_uses_explicit_half_open_ranges_and_allows_source_episode():
    records = (
        SimpleNamespace(episode_index=10, task_index=4, dataset_from_index=0, dataset_to_index=2),
        SimpleNamespace(episode_index=11, task_index=4, dataset_from_index=2, dataset_to_index=5),
        SimpleNamespace(episode_index=20, task_index=5, dataset_from_index=5, dataset_to_index=7),
    )
    index = TaskSampleIndex.from_episode_records(records, dataset_length=7)

    assert index.samples_for_task(4) == (0, 1, 2, 3, 4)
    assert index.range_for_episode(11).dataset_from_index == 2
    assert index.range_for_sample(0).episode_index == 10
    assert index.range_for_sample(4).episode_index == 11
    assert index.range_for_sample(6).episode_index == 20
    with pytest.raises(ValueError, match="outside episode ranges"):
        index.range_for_sample(7)
    restored = pickle.loads(pickle.dumps(index))
    assert restored.samples_for_task(5) == (5, 6)
    assert restored.range_for_sample(5).episode_index == 20
    assert restored.digest == index.digest
    assert len(index.digest) == 64


def test_task_sample_index_rejects_overlap_and_out_of_bounds():
    with pytest.raises(ValueError, match="overlap"):
        TaskSampleIndex.from_episode_records(
            (
                SimpleNamespace(episode_index=0, task_index=0, dataset_from_index=0, dataset_to_index=3),
                SimpleNamespace(episode_index=1, task_index=0, dataset_from_index=2, dataset_to_index=4),
            )
        )
    with pytest.raises(ValueError, match="dataset_length"):
        TaskSampleIndex.from_episode_records(
            (SimpleNamespace(episode_index=0, task_index=0, dataset_from_index=0, dataset_to_index=3),),
            dataset_length=2,
        )


@pytest.mark.parametrize(
    "records",
    [
        (SimpleNamespace(episode_index=0, task_index=0, dataset_from_index=1, dataset_to_index=4),),
        (
            SimpleNamespace(episode_index=0, task_index=0, dataset_from_index=0, dataset_to_index=2),
            SimpleNamespace(episode_index=1, task_index=0, dataset_from_index=3, dataset_to_index=4),
        ),
        (SimpleNamespace(episode_index=0, task_index=0, dataset_from_index=0, dataset_to_index=3),),
    ],
)
def test_task_sample_index_requires_exact_formal_dataset_coverage(records):
    with pytest.raises(ValueError, match=r"start|gap|exactly cover"):
        TaskSampleIndex.from_episode_records(records, dataset_length=4)


class _Dataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return copy.deepcopy(self.samples[index])


def _sample(*, episode: int, task: int):
    return {
        "episode_index": np.asarray(episode, dtype=np.int64),
        "task_index": np.asarray(task, dtype=np.int64),
        "state": np.asarray([1.0], dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.float32),
    }


def _task_samples(*, episode: int, task: int) -> TaskSampleIndex:
    return TaskSampleIndex.from_episode_records(
        (
            SimpleNamespace(
                episode_index=episode,
                task_index=task,
                dataset_from_index=0,
                dataset_to_index=1,
            ),
        ),
        dataset_length=1,
    )


def test_guided_dataset_attaches_dynamic_guide_and_allows_same_source_episode():
    catalog = _catalog(_document("guide", episode=10, task=4))
    dataset = GuidedDataset(
        _Dataset([_sample(episode=10, task=4)]),
        catalog,
        _task_samples(episode=10, task=4),
    )

    item = dataset[GuidedSampleIndex(0, 0)]

    assert set(item) == {"query", "guide_index", "query_valid"}
    assert item["guide_index"].item() == 0
    assert bool(item["query_valid"])
    assert set(item["query"]) == {"state", "actions"}


def test_guided_dataset_allows_expected_same_task_episode_different_from_guide():
    catalog = _catalog(_document("guide", episode=10, task=4))
    dataset = GuidedDataset(
        _Dataset([_sample(episode=11, task=4)]),
        catalog,
        _task_samples(episode=11, task=4),
    )

    item = dataset[GuidedSampleIndex(0, 0)]

    assert item["guide_index"].item() == 0


def test_guided_dataset_fails_on_cross_task_pairing():
    catalog = _catalog(_document("guide", episode=10, task=4))
    dataset = GuidedDataset(
        _Dataset([_sample(episode=20, task=5)]),
        catalog,
        _task_samples(episode=20, task=5),
    )

    with pytest.raises(ValueError, match="does not match Guide"):
        dataset[GuidedSampleIndex(0, 0)]


def test_guided_dataset_fails_when_sample_reports_wrong_episode():
    catalog = _catalog(_document("guide", episode=99, task=4))
    dataset = GuidedDataset(
        _Dataset([_sample(episode=11, task=4)]),
        catalog,
        _task_samples(episode=10, task=4),
    )

    with pytest.raises(
        ValueError,
        match=r"sample_index=0 expected episode_index=10, got 11",
    ):
        dataset[GuidedSampleIndex(0, 0)]


def test_guided_dataset_fails_when_sample_reports_wrong_task_for_range():
    catalog = _catalog(_document("guide", episode=99, task=4))
    dataset = GuidedDataset(
        _Dataset([_sample(episode=10, task=5)]),
        catalog,
        _task_samples(episode=10, task=4),
    )

    with pytest.raises(
        ValueError,
        match=r"sample_index=0 expected task_index=4, got 5",
    ):
        dataset[GuidedSampleIndex(0, 0)]
