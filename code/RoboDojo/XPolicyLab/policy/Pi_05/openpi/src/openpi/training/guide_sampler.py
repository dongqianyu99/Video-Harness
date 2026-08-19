from __future__ import annotations

from collections import Counter
from collections import deque
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal
import zlib

import numpy as np

from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.guide_dataset import GuideSampleIndex


@dataclass(frozen=True, slots=True)
class QueryEpisodeRange:
    """Half-open native-dataset index range for one query episode."""

    episode_index: int
    dataset_from_index: int
    dataset_to_index: int


@dataclass(frozen=True, slots=True)
class BindingBatchStats:
    """Sample and full-batch accounting for one static Guide binding."""

    binding_index: int
    total_samples: int
    used_samples: int
    dropped_samples: int
    num_batches: int


@dataclass(frozen=True, slots=True)
class GroupedBatchStats:
    """Accounting for grouped ``G guides x Q queries`` batches."""

    guides_per_batch: int
    queries_per_guide: int
    total_query_groups: int
    used_query_groups: int
    dropped_query_groups: int
    num_batches: int
    mixed_task_batches: int
    valid_query_samples: int
    padded_query_slots: int
    dropped_query_samples: int
    bucket_batch_counts: tuple[tuple[str, int], ...]


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _range_field(entry: Any, name: str) -> Any:
    try:
        return getattr(entry, name)
    except AttributeError as exc:
        raise ValueError(f"episode range must provide field {name!r}") from exc


def build_binding_to_sample_indices(
    episode_ranges: Sequence[Any],
    *,
    binding_index: GuideBindingIndex,
) -> Mapping[int, tuple[int, ...]]:
    """Map each Guide binding to its query episode's native sample indices.

    ``dataset_to_index`` is exclusive.  Episode ranges are validated and
    checked for overlap before non-query ranges are discarded from the
    returned mapping.
    """

    normalized_ranges: list[tuple[int, int, int]] = []
    seen_episodes: set[int] = set()

    for entry in episode_ranges:
        episode_index = _require_nonnegative_integer(
            _range_field(entry, "episode_index"),
            name="episode_index",
        )
        dataset_from_index = _require_nonnegative_integer(
            _range_field(entry, "dataset_from_index"),
            name="dataset_from_index",
        )
        dataset_to_index = _require_nonnegative_integer(
            _range_field(entry, "dataset_to_index"),
            name="dataset_to_index",
        )

        if dataset_to_index <= dataset_from_index:
            raise ValueError(
                "episode range must be non-empty with dataset_to_index > "
                f"dataset_from_index, got [{dataset_from_index}, {dataset_to_index})"
            )

        if episode_index in seen_episodes:
            raise ValueError(f"duplicate episode_index={episode_index} range")
        seen_episodes.add(episode_index)
        normalized_ranges.append(
            (episode_index, dataset_from_index, dataset_to_index)
        )

    normalized_ranges.sort(key=lambda item: (item[1], item[2], item[0]))

    for previous, current in pairwise(normalized_ranges):
        if current[1] < previous[2]:
            raise ValueError(
                "episode ranges overlap: "
                f"[{previous[1]}, {previous[2]}) and "
                f"[{current[1]}, {current[2]})"
            )

    ranges_by_episode = {
        episode_index: (dataset_from_index, dataset_to_index)
        for episode_index, dataset_from_index, dataset_to_index in normalized_ranges
    }

    result: dict[int, tuple[int, ...]] = {}
    for record in binding_index.records:
        try:
            dataset_from_index, dataset_to_index = ranges_by_episode[
                record.query_episode_index
            ]
        except KeyError as exc:
            raise ValueError(
                "missing dataset range for "
                f"query_episode_index={record.query_episode_index}"
            ) from exc

        result[record.binding_index] = tuple(
            range(dataset_from_index, dataset_to_index)
        )

    return MappingProxyType(result)


class HomogeneousBindingBatchSampler:
    """Shuffle and batch samples without mixing Guide bindings."""

    def __init__(
        self,
        binding_to_sample_indices: Mapping[int, Sequence[int]],
        *,
        binding_index: GuideBindingIndex,
        batch_size: int,
        seed: int = 0,
    ):
        if not isinstance(binding_to_sample_indices, Mapping) or not binding_to_sample_indices:
            raise ValueError("binding_to_sample_indices must be a non-empty mapping")

        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

        self._batch_size = int(batch_size)
        self._seed = _require_nonnegative_integer(seed, name="seed")
        self._epoch = 0

        normalized_groups: list[tuple[int, tuple[int, ...]]] = []
        seen_samples: set[int] = set()

        for raw_binding_index, raw_sample_indices in binding_to_sample_indices.items():
            current_binding_index = _require_nonnegative_integer(
                raw_binding_index,
                name="binding_index",
            )
            binding_index.by_binding_index(current_binding_index)

            try:
                sample_indices = tuple(raw_sample_indices)
            except TypeError as exc:
                raise ValueError(
                    f"sample indices for binding_index={current_binding_index} must be a sequence"
                ) from exc

            if len(sample_indices) < self._batch_size:
                raise ValueError(
                    f"binding_index={current_binding_index} has {len(sample_indices)} "
                    f"samples, fewer than batch_size={self._batch_size}"
                )

            normalized_samples: list[int] = []
            local_samples: set[int] = set()
            for raw_sample_index in sample_indices:
                sample_index = _require_nonnegative_integer(
                    raw_sample_index,
                    name="sample_index",
                )
                if sample_index in local_samples:
                    raise ValueError(
                        f"duplicate sample_index={sample_index} within "
                        f"binding_index={current_binding_index}"
                    )
                if sample_index in seen_samples:
                    raise ValueError(
                        f"sample_index={sample_index} appears in multiple bindings"
                    )

                local_samples.add(sample_index)
                seen_samples.add(sample_index)
                normalized_samples.append(sample_index)

            normalized_groups.append(
                (current_binding_index, tuple(normalized_samples))
            )

        normalized_groups.sort(key=lambda item: item[0])
        self._groups = tuple(normalized_groups)

        stats = []
        for current_binding_index, sample_indices in self._groups:
            num_batches = len(sample_indices) // self._batch_size
            used_samples = num_batches * self._batch_size
            stats.append(
                BindingBatchStats(
                    binding_index=current_binding_index,
                    total_samples=len(sample_indices),
                    used_samples=used_samples,
                    dropped_samples=len(sample_indices) - used_samples,
                    num_batches=num_batches,
                )
            )
        self._stats = tuple(stats)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stats(self) -> tuple[BindingBatchStats, ...]:
        return self._stats

    @property
    def total_samples(self) -> int:
        return sum(item.total_samples for item in self._stats)

    @property
    def used_samples(self) -> int:
        return sum(item.used_samples for item in self._stats)

    @property
    def dropped_samples(self) -> int:
        return sum(item.dropped_samples for item in self._stats)

    @property
    def num_batches(self) -> int:
        return sum(item.num_batches for item in self._stats)

    @property
    def total_batches(self) -> int:
        """Alias for callers that use the global batch terminology."""

        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = _require_nonnegative_integer(epoch, name="epoch")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        complete_batches: list[list[int]] = []

        for current_binding_index, sample_indices in self._groups:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self._seed, self._epoch, current_binding_index]
                )
            )
            shuffled_indices = np.asarray(sample_indices, dtype=np.int64).copy()
            rng.shuffle(shuffled_indices)

            num_batches = len(sample_indices) // self._batch_size
            for batch_index in range(num_batches):
                start = batch_index * self._batch_size
                stop = start + self._batch_size
                complete_batches.append(
                    [int(value) for value in shuffled_indices[start:stop]]
                )

        global_rng = np.random.default_rng(
            np.random.SeedSequence(
                [self._seed, self._epoch, 0xA7C0_0000]
            )
        )
        global_rng.shuffle(complete_batches)

        yield from complete_batches


class GroupedBindingBatchSampler:
    """Build fixed ``[G, Q]`` batches with distinct Guide documents.

    Each contiguous block of ``Q`` sample indices belongs to one static query
    episode/Guide binding.  A batch contains ``G`` such blocks in group-major
    order.  Candidate selection prefers task diversity while preserving every
    binding's frame-level samples except incomplete ``Q`` tails and the final
    groups that cannot form a complete ``G`` batch.
    """

    def __init__(
        self,
        binding_to_sample_indices: Mapping[int, Sequence[int]],
        *,
        binding_index: GuideBindingIndex,
        guides_per_batch: int,
        queries_per_guide: int,
        seed: int = 0,
        binding_to_bucket: Mapping[int, Hashable] | None = None,
        remainder_strategy: Literal["drop", "pad_mask"] = "drop",
        batch_block_size: int = 1,
    ):
        if not isinstance(binding_to_sample_indices, Mapping) or not binding_to_sample_indices:
            raise ValueError("binding_to_sample_indices must be a non-empty mapping")

        self._guides_per_batch = _require_nonnegative_integer(
            guides_per_batch, name="guides_per_batch"
        )
        self._queries_per_guide = _require_nonnegative_integer(
            queries_per_guide, name="queries_per_guide"
        )
        if self._guides_per_batch <= 0:
            raise ValueError("guides_per_batch must be positive")
        if self._queries_per_guide <= 0:
            raise ValueError("queries_per_guide must be positive")
        if remainder_strategy not in {"drop", "pad_mask"}:
            raise ValueError(
                "remainder_strategy must be 'drop' or 'pad_mask', got "
                f"{remainder_strategy!r}"
            )
        self._batch_block_size = _require_nonnegative_integer(
            batch_block_size, name="batch_block_size"
        )
        if self._batch_block_size <= 0:
            raise ValueError("batch_block_size must be positive")

        self._seed = _require_nonnegative_integer(seed, name="seed")
        self._epoch = 0
        self._binding_index = binding_index
        self._remainder_strategy = remainder_strategy

        normalized_groups: list[tuple[int, tuple[int, ...], Hashable]] = []
        seen_samples: set[int] = set()
        documents_by_bucket: dict[Hashable, set[str]] = {}
        binding_stats: list[BindingBatchStats] = []

        if binding_to_bucket is None:
            binding_to_bucket = dict.fromkeys(binding_to_sample_indices, "default")
        if set(binding_to_bucket) != set(binding_to_sample_indices):
            raise ValueError(
                "binding_to_bucket keys must exactly match binding_to_sample_indices"
            )

        for raw_binding_index, raw_sample_indices in binding_to_sample_indices.items():
            current_binding_index = _require_nonnegative_integer(
                raw_binding_index, name="binding_index"
            )
            record = binding_index.by_binding_index(current_binding_index)
            bucket = binding_to_bucket[current_binding_index]
            try:
                hash(bucket)
            except TypeError as exc:
                raise ValueError("Guide bucket identifiers must be hashable") from exc
            documents_by_bucket.setdefault(bucket, set()).add(
                record.support_document_id
            )
            try:
                sample_indices = tuple(raw_sample_indices)
            except TypeError as exc:
                raise ValueError(
                    f"sample indices for binding_index={current_binding_index} must be a sequence"
                ) from exc

            if (
                len(sample_indices) < self._queries_per_guide
                and self._remainder_strategy == "drop"
            ):
                raise ValueError(
                    f"binding_index={current_binding_index} has {len(sample_indices)} samples, "
                    f"fewer than queries_per_guide={self._queries_per_guide}"
                )

            normalized_samples: list[int] = []
            local_samples: set[int] = set()
            for raw_sample_index in sample_indices:
                sample_index = _require_nonnegative_integer(
                    raw_sample_index, name="sample_index"
                )
                if sample_index in local_samples:
                    raise ValueError(
                        f"duplicate sample_index={sample_index} within "
                        f"binding_index={current_binding_index}"
                    )
                if sample_index in seen_samples:
                    raise ValueError(
                        f"sample_index={sample_index} appears in multiple bindings"
                    )
                local_samples.add(sample_index)
                seen_samples.add(sample_index)
                normalized_samples.append(sample_index)

            if self._remainder_strategy == "drop":
                num_groups = len(normalized_samples) // self._queries_per_guide
                used_samples = num_groups * self._queries_per_guide
            else:
                num_groups = (
                    len(normalized_samples) + self._queries_per_guide - 1
                ) // self._queries_per_guide
                used_samples = len(normalized_samples)
            binding_stats.append(
                BindingBatchStats(
                    binding_index=current_binding_index,
                    total_samples=len(normalized_samples),
                    used_samples=used_samples,
                    dropped_samples=(
                        len(normalized_samples) - used_samples
                        if self._remainder_strategy == "drop"
                        else 0
                    ),
                    num_batches=num_groups,
                )
            )
            normalized_groups.append((current_binding_index, tuple(normalized_samples), bucket))

        if self._remainder_strategy == "drop":
            for bucket, documents in documents_by_bucket.items():
                if len(documents) < self._guides_per_batch:
                    raise ValueError(
                        f"bucket {bucket!r} needs at least {self._guides_per_batch} "
                        f"distinct support documents, found {len(documents)}"
                    )

        normalized_groups.sort(key=lambda item: item[0])
        self._groups = tuple(normalized_groups)
        self._sample_to_binding = {
            sample_index: binding
            for binding, sample_indices, _ in self._groups
            for sample_index in sample_indices
        }
        self._binding_stats = tuple(sorted(binding_stats, key=lambda item: item.binding_index))
        preview_batches = self._build_batches(epoch=0)
        mixed_task_batches = sum(
            len(
                {
                    binding_index.by_binding_index(
                        self._sample_binding(batch[group * self._queries_per_guide])
                    ).task_index
                    for group in range(self._guides_per_batch)
                    if any(
                        self._sample_valid(sample)
                        for sample in batch[
                            group * self._queries_per_guide :
                            (group + 1) * self._queries_per_guide
                        ]
                    )
                }
            )
            > 1
            for batch in preview_batches
        )
        total_query_groups = sum(item.num_batches for item in self._binding_stats)
        used_query_groups = sum(
            any(self._sample_valid(sample) for sample in group)
            for batch in preview_batches
            for group in self._batch_groups(batch)
        )
        valid_query_samples = sum(
            self._sample_valid(sample)
            for batch in preview_batches
            for sample in batch
        )
        total_slots = len(preview_batches) * self.batch_size
        bucket_batch_counts = Counter(
            str(self._batch_bucket(batch)) for batch in preview_batches
        )
        self._stats = GroupedBatchStats(
            guides_per_batch=self._guides_per_batch,
            queries_per_guide=self._queries_per_guide,
            total_query_groups=total_query_groups,
            used_query_groups=used_query_groups,
            dropped_query_groups=total_query_groups - used_query_groups,
            num_batches=len(preview_batches),
            mixed_task_batches=mixed_task_batches,
            valid_query_samples=valid_query_samples,
            padded_query_slots=total_slots - valid_query_samples,
            dropped_query_samples=sum(
                len(samples) for _, samples, _ in self._groups
            )
            - valid_query_samples,
            bucket_batch_counts=tuple(sorted(bucket_batch_counts.items())),
        )
        if not preview_batches:
            raise ValueError("grouped sampler cannot form one complete batch")

    @staticmethod
    def _raw_sample_index(sample: int | GuideSampleIndex) -> int:
        return sample.sample_index if isinstance(sample, GuideSampleIndex) else sample

    @staticmethod
    def _sample_valid(sample: int | GuideSampleIndex) -> bool:
        return sample.query_valid if isinstance(sample, GuideSampleIndex) else True

    def _sample_binding(self, sample: int | GuideSampleIndex) -> int:
        return self._sample_to_binding[self._raw_sample_index(sample)]

    def _batch_groups(
        self, batch: Sequence[int | GuideSampleIndex]
    ) -> tuple[Sequence[int | GuideSampleIndex], ...]:
        return tuple(
            batch[index : index + self._queries_per_guide]
            for index in range(0, len(batch), self._queries_per_guide)
        )

    def _batch_bucket(self, batch: Sequence[int | GuideSampleIndex]) -> Hashable:
        binding = self._sample_binding(batch[0])
        return next(bucket for current, _, bucket in self._groups if current == binding)

    @property
    def guides_per_batch(self) -> int:
        return self._guides_per_batch

    @property
    def queries_per_guide(self) -> int:
        return self._queries_per_guide

    @property
    def batch_size(self) -> int:
        return self._guides_per_batch * self._queries_per_guide

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stats(self) -> GroupedBatchStats:
        return self._stats

    @property
    def binding_stats(self) -> tuple[BindingBatchStats, ...]:
        return self._binding_stats

    @property
    def remainder_strategy(self) -> str:
        return self._remainder_strategy

    @property
    def batch_block_size(self) -> int:
        return self._batch_block_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = _require_nonnegative_integer(epoch, name="epoch")

    def __len__(self) -> int:
        return self._stats.num_batches

    def _make_query_groups(
        self, epoch: int, bucket: Hashable
    ) -> dict[int, deque[tuple[int | GuideSampleIndex, ...]]]:
        groups: dict[int, deque[tuple[int | GuideSampleIndex, ...]]] = {}
        for binding_index, sample_indices, current_bucket in self._groups:
            if current_bucket != bucket:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([self._seed, epoch, binding_index])
            )
            shuffled = np.asarray(sample_indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            count = len(shuffled) // self._queries_per_guide
            chunks = [
                tuple(int(value) for value in shuffled[
                    index * self._queries_per_guide : (index + 1) * self._queries_per_guide
                ])
                for index in range(count)
            ]
            remainder = len(shuffled) % self._queries_per_guide
            if remainder and self._remainder_strategy == "pad_mask":
                tail = [int(value) for value in shuffled[count * self._queries_per_guide :]]
                pad_index = tail[-1]
                tail.extend(
                    GuideSampleIndex(pad_index, query_valid=False)
                    for _ in range(self._queries_per_guide - remainder)
                )
                chunks.append(tuple(tail))
            groups[binding_index] = deque(chunks)
        return groups

    def _build_bucket_batches(
        self,
        epoch: int,
        bucket: Hashable,
    ) -> list[list[int | GuideSampleIndex]]:
        queues = self._make_query_groups(epoch, bucket)
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self._seed,
                    epoch,
                    0xA7C0_2000,
                    zlib.crc32(str(bucket).encode()),
                ]
            )
        )
        tie_break = {
            binding_index: float(rng.random())
            for binding_index in queues
        }
        batches: list[list[int | GuideSampleIndex]] = []

        while True:
            active = [binding for binding, queue in queues.items() if queue]
            selected: list[int] = []
            selected_documents: set[str] = set()
            selected_tasks: set[int] = set()

            while len(selected) < self._guides_per_batch:
                candidates = [
                    binding
                    for binding in active
                    if binding not in selected
                    and self._binding_index.by_binding_index(binding).support_document_id
                    not in selected_documents
                ]
                if not candidates:
                    break

                unseen_task = [
                    binding
                    for binding in candidates
                    if self._binding_index.by_binding_index(binding).task_index
                    not in selected_tasks
                ]
                pool = unseen_task or candidates
                chosen = max(
                    pool,
                    key=lambda binding: (
                        len(queues[binding]),
                        tie_break[binding],
                        -binding,
                    ),
                )
                record = self._binding_index.by_binding_index(chosen)
                selected.append(chosen)
                selected_documents.add(record.support_document_id)
                selected_tasks.add(record.task_index)

            if not selected:
                break
            real_group_count = len(selected)
            if (
                real_group_count < self._guides_per_batch
                and self._remainder_strategy == "drop"
            ):
                break

            batch: list[int | GuideSampleIndex] = []
            for binding in selected:
                batch.extend(queues[binding].popleft())
            if real_group_count < self._guides_per_batch:
                source_group = batch[: self._queries_per_guide]
                for _ in range(self._guides_per_batch - real_group_count):
                    batch.extend(
                        GuideSampleIndex(
                            self._raw_sample_index(sample), query_valid=False
                        )
                        for sample in source_group
                    )
            batches.append(batch)

            for binding in selected:
                tie_break[binding] = float(rng.random())

        rng.shuffle(batches)

        block_remainder = len(batches) % self._batch_block_size
        if block_remainder:
            if self._remainder_strategy == "drop":
                batches = batches[:-block_remainder]
            else:
                template = batches[-1]
                for _ in range(self._batch_block_size - block_remainder):
                    batches.append(
                        [
                            GuideSampleIndex(
                                self._raw_sample_index(sample), query_valid=False
                            )
                            for sample in template
                        ]
                    )
        return batches

    def _build_batches(self, epoch: int) -> list[list[int | GuideSampleIndex]]:
        buckets = sorted({bucket for _, _, bucket in self._groups}, key=str)
        blocks: list[list[list[int | GuideSampleIndex]]] = []
        for bucket in buckets:
            batches = self._build_bucket_batches(epoch, bucket)
            blocks.extend(
                batches[index : index + self._batch_block_size]
                for index in range(0, len(batches), self._batch_block_size)
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, epoch, 0xA7C0_3000])
        )
        rng.shuffle(blocks)
        return [batch for block in blocks for batch in block]

    def __iter__(self) -> Iterator[list[int | GuideSampleIndex]]:
        batches = self._build_batches(self._epoch)
        if len(batches) != len(self):
            raise RuntimeError(
                "grouped sampler batch count changed across epochs: "
                f"expected {len(self)}, got {len(batches)}"
            )
        yield from batches
