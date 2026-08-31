from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _field(value: Any, name: str, *, owner: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"{owner} must provide field {name!r}") from exc


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class GuideRecord:
    guide_index: int
    document_id: str
    source_episode_index: int
    task_index: int
    task_instruction: str


@dataclass(frozen=True, slots=True)
class GuideCatalog:
    """Stable numeric view of accepted VideoHarness Guidance documents."""

    catalog_digest: str
    _records: tuple[GuideRecord, ...]
    _by_guide: Mapping[int, GuideRecord]
    _by_document: Mapping[str, GuideRecord]
    _by_task: Mapping[int, tuple[GuideRecord, ...]]

    @classmethod
    def from_document_catalog(cls, document_catalog: Any) -> GuideCatalog:
        digest = _nonempty_string(
            _field(document_catalog, "catalog_digest", owner="document catalog"),
            name="catalog_digest",
        )
        documents = tuple(
            _field(document_catalog, "documents", owner="document catalog")
        )
        if not documents:
            raise ValueError("document catalog contains no accepted Guidance documents")

        normalized: list[tuple[int, int, str, str]] = []
        for document in documents:
            document_id = _nonempty_string(
                _field(document, "document_id", owner="GuideDocument"),
                name="document_id",
            )
            source_episode_index = _require_nonnegative_integer(
                _field(document, "source_episode_index", owner="GuideDocument"),
                name="source_episode_index",
            )
            task_index = _require_nonnegative_integer(
                _field(document, "task_index", owner="GuideDocument"),
                name="task_index",
            )
            task_instruction = _nonempty_string(
                _field(document, "task_instruction", owner="GuideDocument"),
                name="task_instruction",
            )
            normalized.append(
                (
                    task_index,
                    source_episode_index,
                    document_id,
                    task_instruction,
                )
            )

        normalized.sort(key=lambda item: (item[0], item[1], item[2]))
        records = tuple(
            GuideRecord(
                guide_index=guide_index,
                document_id=document_id,
                source_episode_index=source_episode_index,
                task_index=task_index,
                task_instruction=task_instruction,
            )
            for guide_index, (
                task_index,
                source_episode_index,
                document_id,
                task_instruction,
            ) in enumerate(normalized)
        )
        return cls.from_records(records, digest)

    @classmethod
    def from_records(
        cls,
        records: Sequence[GuideRecord],
        catalog_digest: str,
    ) -> GuideCatalog:
        records = tuple(records)
        if not records:
            raise ValueError("GuideCatalog records must not be empty")
        if tuple(record.guide_index for record in records) != tuple(range(len(records))):
            raise ValueError("GuideRecord guide_index values must be dense and ordered")

        by_document: dict[str, GuideRecord] = {}
        by_task_lists: dict[int, list[GuideRecord]] = defaultdict(list)
        instructions: dict[int, str] = {}
        for record in records:
            if record.document_id in by_document:
                raise ValueError(f"duplicate document_id={record.document_id!r}")
            expected_instruction = instructions.setdefault(
                record.task_index, record.task_instruction
            )
            if record.task_instruction != expected_instruction:
                raise ValueError(
                    f"task_index={record.task_index} has inconsistent task instructions"
                )
            by_document[record.document_id] = record
            by_task_lists[record.task_index].append(record)

        return cls(
            catalog_digest=_nonempty_string(catalog_digest, name="catalog_digest"),
            _records=records,
            _by_guide=MappingProxyType(
                {record.guide_index: record for record in records}
            ),
            _by_document=MappingProxyType(by_document),
            _by_task=MappingProxyType(
                {task: tuple(task_records) for task, task_records in by_task_lists.items()}
            ),
        )

    def __reduce__(self):
        return (type(self).from_records, (self._records, self.catalog_digest))

    @property
    def records(self) -> tuple[GuideRecord, ...]:
        return self._records

    @property
    def task_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_task))

    def by_guide_index(self, guide_index: int) -> GuideRecord:
        guide_index = _require_nonnegative_integer(guide_index, name="guide_index")
        try:
            return self._by_guide[guide_index]
        except KeyError as exc:
            raise ValueError(f"unknown guide_index={guide_index}") from exc

    def by_document_id(self, document_id: str) -> GuideRecord:
        document_id = _nonempty_string(document_id, name="document_id")
        try:
            return self._by_document[document_id]
        except KeyError as exc:
            raise ValueError(f"unknown document_id={document_id!r}") from exc

    def records_for_task(self, task_index: int) -> tuple[GuideRecord, ...]:
        task_index = _require_nonnegative_integer(task_index, name="task_index")
        try:
            return self._by_task[task_index]
        except KeyError as exc:
            raise ValueError(f"unknown task_index={task_index}") from exc


@dataclass(frozen=True, slots=True)
class EpisodeSampleRange:
    episode_index: int
    task_index: int
    dataset_from_index: int
    dataset_to_index: int
    data_path: str = ""


@dataclass(frozen=True, slots=True)
class TaskSampleIndex:
    """Explicit native sample pools keyed by RoboDojo task."""

    _ranges: tuple[EpisodeSampleRange, ...]
    _range_starts: tuple[int, ...]
    _by_task: Mapping[int, tuple[int, ...]]
    _by_episode: Mapping[int, EpisodeSampleRange]
    digest: str

    @classmethod
    def from_episode_records(
        cls,
        episode_records: Sequence[Any],
        *,
        dataset_length: int | None = None,
    ) -> TaskSampleIndex:
        if dataset_length is not None:
            dataset_length = _require_nonnegative_integer(
                dataset_length, name="dataset_length"
            )
        ranges: list[EpisodeSampleRange] = []
        seen_episodes: set[int] = set()
        for record in episode_records:
            episode_index = _require_nonnegative_integer(
                _field(record, "episode_index", owner="episode record"),
                name="episode_index",
            )
            task_index = _require_nonnegative_integer(
                _field(record, "task_index", owner="episode record"),
                name="task_index",
            )
            start = _require_nonnegative_integer(
                _field(record, "dataset_from_index", owner="episode record"),
                name="dataset_from_index",
            )
            stop = _require_nonnegative_integer(
                _field(record, "dataset_to_index", owner="episode record"),
                name="dataset_to_index",
            )
            raw_data_path = getattr(record, "data_path", "")
            if not isinstance(raw_data_path, str):
                raise ValueError("episode record data_path must be a string")
            if stop <= start:
                raise ValueError(
                    f"episode_index={episode_index} has empty sample range [{start}, {stop})"
                )
            if dataset_length is not None and stop > dataset_length:
                raise ValueError(
                    f"episode_index={episode_index} range ends at {stop}, "
                    f"beyond dataset_length={dataset_length}"
                )
            if episode_index in seen_episodes:
                raise ValueError(f"duplicate episode_index={episode_index}")
            seen_episodes.add(episode_index)
            ranges.append(
                EpisodeSampleRange(
                    episode_index=episode_index,
                    task_index=task_index,
                    dataset_from_index=start,
                    dataset_to_index=stop,
                    data_path=raw_data_path,
                )
            )

        if not ranges:
            raise ValueError("episode_records must not be empty")
        ranges.sort(
            key=lambda item: (
                item.dataset_from_index,
                item.dataset_to_index,
                item.episode_index,
            )
        )
        for previous, current in pairwise(ranges):
            if current.dataset_from_index < previous.dataset_to_index:
                raise ValueError(
                    "episode sample ranges overlap: "
                    f"[{previous.dataset_from_index}, {previous.dataset_to_index}) and "
                    f"[{current.dataset_from_index}, {current.dataset_to_index})"
                )
        if dataset_length is not None:
            if ranges[0].dataset_from_index != 0:
                raise ValueError(
                    "episode sample ranges must start at dataset index 0"
                )
            for previous, current in pairwise(ranges):
                if current.dataset_from_index != previous.dataset_to_index:
                    raise ValueError(
                        "episode sample ranges contain a dataset index gap: "
                        f"{previous.dataset_to_index}..{current.dataset_from_index}"
                    )
            if ranges[-1].dataset_to_index != dataset_length:
                raise ValueError(
                    "episode sample ranges must exactly cover the native dataset: "
                    f"last={ranges[-1].dataset_to_index}, dataset_length={dataset_length}"
                )
        return cls.from_ranges(tuple(ranges))

    @classmethod
    def from_ranges(
        cls, ranges: tuple[EpisodeSampleRange, ...]
    ) -> TaskSampleIndex:
        by_task_lists: dict[int, list[int]] = defaultdict(list)
        for sample_range in ranges:
            by_task_lists[sample_range.task_index].extend(
                range(
                    sample_range.dataset_from_index,
                    sample_range.dataset_to_index,
                )
            )
        digest_payload = json.dumps(
            [
                {
                    "episode_index": item.episode_index,
                    "task_index": item.task_index,
                    "dataset_from_index": item.dataset_from_index,
                    "dataset_to_index": item.dataset_to_index,
                    "data_path": item.data_path,
                }
                for item in ranges
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            _ranges=ranges,
            _range_starts=tuple(item.dataset_from_index for item in ranges),
            _by_task=MappingProxyType(
                {task: tuple(indices) for task, indices in by_task_lists.items()}
            ),
            _by_episode=MappingProxyType(
                {item.episode_index: item for item in ranges}
            ),
            digest=hashlib.sha256(digest_payload).hexdigest(),
        )

    def __reduce__(self):
        return (type(self).from_ranges, (self._ranges,))

    @property
    def task_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_task))

    @property
    def total_samples(self) -> int:
        return sum(len(samples) for samples in self._by_task.values())

    @property
    def ranges(self) -> tuple[EpisodeSampleRange, ...]:
        return self._ranges

    def samples_for_task(self, task_index: int) -> tuple[int, ...]:
        task_index = _require_nonnegative_integer(task_index, name="task_index")
        try:
            return self._by_task[task_index]
        except KeyError as exc:
            raise ValueError(f"task_index={task_index} has no native samples") from exc

    def range_for_episode(self, episode_index: int) -> EpisodeSampleRange:
        episode_index = _require_nonnegative_integer(
            episode_index, name="episode_index"
        )
        try:
            return self._by_episode[episode_index]
        except KeyError as exc:
            raise ValueError(f"unknown episode_index={episode_index}") from exc

    def range_for_sample(self, sample_index: int) -> EpisodeSampleRange:
        sample_index = _require_nonnegative_integer(
            sample_index, name="sample_index"
        )
        position = bisect_right(self._range_starts, sample_index) - 1
        if position >= 0:
            sample_range = self._ranges[position]
            if sample_index < sample_range.dataset_to_index:
                return sample_range
        raise ValueError(f"sample_index={sample_index} is outside episode ranges")


@dataclass(frozen=True, slots=True)
class GuidedSampleIndex:
    sample_index: int
    guide_index: int
    query_valid: bool = True

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.sample_index, name="sample_index")
        _require_nonnegative_integer(self.guide_index, name="guide_index")
        if not isinstance(self.query_valid, bool):
            raise ValueError("query_valid must be bool")


def _sample_identity_index(sample: Mapping[str, Any], name: str) -> int:
    if name not in sample:
        raise ValueError(f"sample is missing {name!r}")
    value = np.asarray(sample[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"sample {name!r} must be a scalar integer")
    result = int(value.item())
    if result < 0:
        raise ValueError(f"sample {name!r} must be non-negative")
    return result


class GuidedDataset:
    """Attach a dynamically selected same-task Guide to one native query."""

    def __init__(
        self,
        dataset: Any,
        guide_catalog: GuideCatalog,
        task_sample_index: TaskSampleIndex,
    ):
        self._dataset = dataset
        self._guide_catalog = guide_catalog
        self._task_sample_index = task_sample_index

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: GuidedSampleIndex) -> dict[str, Any]:
        if not isinstance(index, GuidedSampleIndex):
            raise ValueError("GuidedDataset indices must be GuidedSampleIndex")
        expected_range = self._task_sample_index.range_for_sample(
            index.sample_index
        )
        sample = self._dataset[index.sample_index]
        if not isinstance(sample, Mapping):
            raise ValueError(
                f"dataset sample must be a mapping, got {type(sample).__name__}"
            )

        episode_index = _sample_identity_index(sample, "episode_index")
        task_index = _sample_identity_index(sample, "task_index")
        if episode_index != expected_range.episode_index:
            raise ValueError(
                f"sample_index={index.sample_index} expected episode_index="
                f"{expected_range.episode_index}, got {episode_index}"
            )
        if task_index != expected_range.task_index:
            raise ValueError(
                f"sample_index={index.sample_index} expected task_index="
                f"{expected_range.task_index}, got {task_index}"
            )
        guide = self._guide_catalog.by_guide_index(index.guide_index)
        if expected_range.task_index != guide.task_index:
            raise ValueError(
                f"sample task_index={expected_range.task_index} does not match Guide "
                f"task_index={guide.task_index} for guide_index={guide.guide_index}"
            )

        query = dict(sample)
        query.pop("episode_index")
        query.pop("task_index")
        return {
            "query": query,
            "guide_index": np.asarray(guide.guide_index, dtype=np.int32),
            "query_valid": np.asarray(index.query_valid, dtype=np.bool_),
        }
