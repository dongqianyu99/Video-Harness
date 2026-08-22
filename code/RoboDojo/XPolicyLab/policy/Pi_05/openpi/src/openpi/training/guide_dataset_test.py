import copy
from dataclasses import FrozenInstanceError
from dataclasses import dataclass

import numpy as np
import pytest

from openpi.training import guide_dataset as _guide_dataset


@dataclass(frozen=True)
class _SupportBinding:
    build_id: str = "build-0"
    pair_id: str = "pair-0"
    query_episode_index: int = 10
    support_episode_index: int = 11
    support_document_id: str = "document-0"
    support_rank: int = 0
    task_index: int = 4
    task_instruction: str = "open the cabinet"
    guide_schema_version: str = "actuator"


def _make_binding(**overrides) -> _SupportBinding:
    values = {
        "build_id": "build-0",
        "pair_id": "pair-0",
        "query_episode_index": 10,
        "support_episode_index": 11,
        "support_document_id": "document-0",
        "support_rank": 0,
        "task_index": 4,
        "task_instruction": "open the cabinet",
        "guide_schema_version": "actuator",
    }
    values.update(overrides)
    return _SupportBinding(**values)


def _make_index(*bindings: _SupportBinding):
    return _guide_dataset.GuideBindingIndex.from_bindings(list(bindings))


def _index_value(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def _make_sample(
    episode_index,
    task_index,
    *,
    value: float = 0.0,
) -> dict[str, object]:
    return {
        "observation": {
            "state": np.asarray([value, value + 1], dtype=np.float32),
        },
        "actions": np.asarray([[value, value + 1]], dtype=np.float32),
        "episode_index": episode_index,
        "task_index": task_index,
    }


class _CountingDataset:
    def __init__(self, samples: list[dict[str, object]]):
        self.samples = samples
        self.calls: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        self.calls.append(index)
        return self.samples[index]


def _contains_string(value) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(_contains_string(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_string(child) for child in value)
    return False


def test_binding_index_sorts_queries_and_assigns_contiguous_indices():
    bindings = [
        _make_binding(
            pair_id="pair-20",
            query_episode_index=20,
            support_episode_index=21,
            support_document_id="document-20",
        ),
        _make_binding(
            pair_id="pair-3",
            query_episode_index=3,
            support_episode_index=4,
            support_document_id="document-3",
        ),
        _make_binding(
            pair_id="pair-11",
            query_episode_index=11,
            support_episode_index=12,
            support_document_id="document-11",
        ),
    ]

    index = _make_index(*bindings)

    assert [record.binding_index for record in index.records] == [0, 1, 2]
    assert [record.query_episode_index for record in index.records] == [3, 11, 20]
    assert index.by_query_episode(11).support_document_id == "document-11"
    assert index.by_binding_index(0).query_episode_index == 3


def test_binding_index_does_not_modify_input_sequence():
    bindings = [
        _make_binding(query_episode_index=20, support_episode_index=21),
        _make_binding(
            pair_id="pair-3",
            query_episode_index=3,
            support_episode_index=4,
        ),
    ]
    original = list(bindings)

    _make_index(*bindings)

    assert bindings == original


def test_binding_records_and_record_sequence_are_immutable():
    index = _make_index(_make_binding())

    assert isinstance(index.records, tuple)

    with pytest.raises(TypeError):
        index.records[0] = index.records[0]

    with pytest.raises(FrozenInstanceError):
        index.records[0].binding_index = 99


def test_binding_index_rejects_duplicate_query_episode():
    with pytest.raises(ValueError, match=r"duplicate|query_episode_index"):
        _make_index(
            _make_binding(query_episode_index=10, support_episode_index=11),
            _make_binding(
                pair_id="pair-10-second",
                query_episode_index=10,
                support_episode_index=12,
            ),
        )


def test_binding_index_rejects_query_equal_to_support_episode():
    with pytest.raises(ValueError, match=r"different|support_episode_index"):
        _make_index(
            _make_binding(query_episode_index=10, support_episode_index=10),
        )


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"query_episode_index": -1}, "query_episode_index"),
        ({"query_episode_index": True}, "query_episode_index"),
        ({"support_episode_index": -1}, "support_episode_index"),
        ({"support_episode_index": False}, "support_episode_index"),
        ({"task_index": -1}, "task_index"),
        ({"task_index": True}, "task_index"),
        ({"support_document_id": ""}, "support_document_id"),
    ],
)
def test_binding_index_rejects_invalid_binding_fields(overrides, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        _make_index(_make_binding(**overrides))


def test_binding_index_rejects_unknown_lookups():
    index = _make_index()

    with pytest.raises(ValueError, match="query_episode_index"):
        index.by_query_episode(999)

    with pytest.raises(ValueError, match="binding_index"):
        index.by_binding_index(999)

    with pytest.raises(ValueError, match="binding_index"):
        index.by_binding_index(-1)


def test_bound_dataset_removes_identity_metadata_and_preserves_content():
    binding_index = _make_index(_make_binding())
    sample = _make_sample(_index_value(10), _index_value(4), value=1.0)
    sample_before = copy.deepcopy(sample)
    dataset = _CountingDataset([sample])
    bound_dataset = _guide_dataset.GuideBoundDataset(dataset, binding_index)

    result = bound_dataset[0]

    assert len(bound_dataset) == len(dataset)
    assert dataset.calls == [0]
    assert set(result) == {"query", "guide_binding_index", "query_valid"}
    assert bool(result["query_valid"])
    assert set(result["query"]) == {"observation", "actions"}
    assert result["guide_binding_index"].shape == ()
    assert result["guide_binding_index"].dtype == np.int32
    assert result["guide_binding_index"].item() == 0
    np.testing.assert_array_equal(
        result["query"]["observation"]["state"],
        sample_before["observation"]["state"],
    )
    np.testing.assert_array_equal(result["query"]["actions"], sample_before["actions"])
    assert set(sample) == set(sample_before)
    np.testing.assert_array_equal(sample["episode_index"], sample_before["episode_index"])
    np.testing.assert_array_equal(sample["task_index"], sample_before["task_index"])


def test_bound_dataset_binds_all_frames_of_an_episode_to_one_record():
    bindings = [
        _make_binding(
            pair_id="pair-10",
            query_episode_index=10,
            support_episode_index=11,
        ),
        _make_binding(
            pair_id="pair-20",
            query_episode_index=20,
            support_episode_index=21,
        ),
    ]
    dataset = _CountingDataset(
        [
            _make_sample(_index_value(20), _index_value(4), value=0.0),
            _make_sample(_index_value(10), _index_value(4), value=1.0),
            _make_sample(_index_value(10), _index_value(4), value=2.0),
        ]
    )
    bound_dataset = _guide_dataset.GuideBoundDataset(
        dataset,
        _make_index(*bindings),
    )

    results = [bound_dataset[index] for index in range(len(bound_dataset))]

    assert [result["guide_binding_index"].item() for result in results] == [1, 0, 0]
    assert dataset.calls == [0, 1, 2]


def test_bound_dataset_rejects_task_mismatch():
    dataset = _CountingDataset([_make_sample(_index_value(10), _index_value(999))])
    bound_dataset = _guide_dataset.GuideBoundDataset(
        dataset,
        _make_index(_make_binding()),
    )

    with pytest.raises(ValueError, match="task_index"):
        bound_dataset[0]


def test_bound_dataset_rejects_unknown_query_episode():
    dataset = _CountingDataset([_make_sample(_index_value(999), _index_value(4))])
    bound_dataset = _guide_dataset.GuideBoundDataset(dataset, _make_index())

    with pytest.raises(ValueError, match="query_episode_index"):
        bound_dataset[0]


@pytest.mark.parametrize("missing_key", ["episode_index", "task_index"])
def test_bound_dataset_rejects_missing_identity_metadata(missing_key):
    sample = _make_sample(_index_value(10), _index_value(4))
    sample.pop(missing_key)
    bound_dataset = _guide_dataset.GuideBoundDataset(
        _CountingDataset([sample]),
        _make_index(),
    )

    with pytest.raises(ValueError, match=missing_key):
        bound_dataset[0]


@pytest.mark.parametrize(
    ("episode_value", "task_value", "error_pattern"),
    [
        (np.asarray([10], dtype=np.int64), _index_value(4), "episode_index"),
        (np.asarray(1, dtype=np.bool_), _index_value(4), "episode_index"),
        (np.asarray(10.0), _index_value(4), "episode_index"),
        (np.asarray(-1, dtype=np.int64), _index_value(4), "episode_index"),
        (_index_value(10), np.asarray([4], dtype=np.int64), "task_index"),
        (_index_value(10), np.asarray(0, dtype=np.bool_), "task_index"),
        (_index_value(10), np.asarray(4.0), "task_index"),
        (_index_value(10), np.asarray(-1, dtype=np.int64), "task_index"),
    ],
)
def test_bound_dataset_rejects_non_scalar_or_invalid_identity_metadata(
    episode_value,
    task_value,
    error_pattern,
):
    dataset = _CountingDataset([_make_sample(episode_value, task_value)])
    bound_dataset = _guide_dataset.GuideBoundDataset(dataset, _make_index())

    with pytest.raises(ValueError, match=error_pattern):
        bound_dataset[0]


def test_bound_dataset_does_not_add_strings_to_model_facing_output():
    dataset = _CountingDataset([_make_sample(_index_value(10), _index_value(4))])
    bound_dataset = _guide_dataset.GuideBoundDataset(
        dataset,
        _make_index(_make_binding()),
    )

    result = bound_dataset[0]

    assert not _contains_string(result)
