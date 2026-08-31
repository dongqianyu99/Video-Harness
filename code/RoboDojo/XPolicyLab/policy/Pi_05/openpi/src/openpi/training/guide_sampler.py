from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections import deque
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np

from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import GuidedSampleIndex
from openpi.training.guide_dataset import TaskSampleIndex


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class GuidanceFirstBatchStats:
    guides_per_batch: int
    queries_per_guide: int
    total_native_samples: int
    guide_draws: int
    padded_guide_slots: int
    num_batches: int
    mixed_task_batches: int
    valid_query_samples: int
    padded_query_slots: int
    dropped_query_samples: int
    bucket_batch_counts: tuple[tuple[str, int], ...]
    bucket_probabilities: tuple[tuple[str, float], ...]


class _TaskSampleCycle:
    def __init__(
        self,
        samples: Sequence[int],
        *,
        seed: int,
        epoch: int,
        task_index: int,
    ) -> None:
        self._samples = tuple(int(sample) for sample in samples)
        self._seed = seed
        self._epoch = epoch
        self._task_index = task_index
        self._cycle = 0
        self._queue: deque[int] = deque()

    def _refill(self) -> None:
        values = np.asarray(self._samples, dtype=np.int64).copy()
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self._seed, self._epoch, self._task_index, self._cycle]
            )
        )
        rng.shuffle(values)
        self._queue.extend(int(value) for value in values)
        self._cycle += 1

    def take_distinct(self, count: int) -> tuple[int, ...]:
        if len(self._samples) < count:
            raise ValueError(
                f"task_index={self._task_index} has {len(self._samples)} samples, "
                f"fewer than queries_per_guide={count}"
            )
        selected: list[int] = []
        seen: set[int] = set()
        while len(selected) < count:
            if not self._queue:
                self._refill()
            sample = self._queue.popleft()
            if sample in seen:
                continue
            seen.add(sample)
            selected.append(sample)
        return tuple(selected)


class GuidanceFirstBatchSampler:
    """Sample Guidance documents globally, then Q samples from each Guide task."""

    def __init__(
        self,
        *,
        guide_catalog: GuideCatalog,
        task_sample_index: TaskSampleIndex,
        guides_per_batch: int,
        queries_per_guide: int,
        seed: int = 0,
        guide_to_bucket: Mapping[int, Hashable] | None = None,
        remainder_strategy: Literal["drop", "pad_mask"] = "drop",
        batch_block_size: int = 1,
    ) -> None:
        self._guides_per_batch = _require_nonnegative_integer(
            guides_per_batch, name="guides_per_batch"
        )
        self._queries_per_guide = _require_nonnegative_integer(
            queries_per_guide, name="queries_per_guide"
        )
        self._batch_block_size = _require_nonnegative_integer(
            batch_block_size, name="batch_block_size"
        )
        if self._guides_per_batch <= 0 or self._queries_per_guide <= 0:
            raise ValueError("guides_per_batch and queries_per_guide must be positive")
        if self._batch_block_size <= 0:
            raise ValueError("batch_block_size must be positive")
        if remainder_strategy not in {"drop", "pad_mask"}:
            raise ValueError("remainder_strategy must be 'drop' or 'pad_mask'")

        self._guide_catalog = guide_catalog
        self._task_sample_index = task_sample_index
        self._seed = _require_nonnegative_integer(seed, name="seed")
        self._epoch = 0
        self._remainder_strategy = remainder_strategy

        if guide_to_bucket is None:
            guide_to_bucket = {
                record.guide_index: "default" for record in guide_catalog.records
            }
        expected_guides = {record.guide_index for record in guide_catalog.records}
        if set(guide_to_bucket) != expected_guides:
            raise ValueError("guide_to_bucket keys must exactly match GuideCatalog")

        normalized_buckets: dict[Hashable, list[int]] = defaultdict(list)
        for record in guide_catalog.records:
            task_samples = task_sample_index.samples_for_task(record.task_index)
            if (
                len(task_samples) < self._queries_per_guide
                and self._remainder_strategy == "drop"
            ):
                raise ValueError(
                    f"task_index={record.task_index} has {len(task_samples)} samples, "
                    f"fewer than queries_per_guide={self._queries_per_guide}"
                )
            bucket = guide_to_bucket[record.guide_index]
            try:
                hash(bucket)
            except TypeError as exc:
                raise ValueError("Guide bucket identifiers must be hashable") from exc
            normalized_buckets[bucket].append(record.guide_index)

        self._buckets = tuple(
            (bucket, tuple(indices))
            for bucket, indices in sorted(
                normalized_buckets.items(), key=lambda item: str(item[0])
            )
        )
        for bucket, guide_indices in self._buckets:
            if len(guide_indices) < self._guides_per_batch:
                raise ValueError(
                    f"bucket {bucket!r} has {len(guide_indices)} Guides; "
                    "promote it to a larger bucket before sampling with "
                    f"G={self._guides_per_batch}"
                )
        native_samples = task_sample_index.total_samples
        if self._remainder_strategy == "drop":
            num_batches = native_samples // self.batch_size
            num_batches -= num_batches % self._batch_block_size
        else:
            num_batches = (native_samples + self.batch_size - 1) // self.batch_size
            remainder = num_batches % self._batch_block_size
            if remainder:
                num_batches += self._batch_block_size - remainder
        if num_batches <= 0:
            raise ValueError(
                "native sample count cannot form one Guidance-first batch block"
            )
        self._num_batches = num_batches
        self._valid_queries_per_epoch = min(
            native_samples,
            num_batches * self.batch_size,
        )
        preview = self._build_batches(epoch=0)
        if not preview:
            raise ValueError("Guidance-first sampler cannot form one batch")
        if len(preview) != self._num_batches:
            raise AssertionError("Guidance-first sampler batch count drifted")
        self._stats = self._summarize(preview)

    @property
    def batch_size(self) -> int:
        return self._guides_per_batch * self._queries_per_guide

    @property
    def guides_per_batch(self) -> int:
        return self._guides_per_batch

    @property
    def queries_per_guide(self) -> int:
        return self._queries_per_guide

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stats(self) -> GuidanceFirstBatchStats:
        return self._stats

    def set_epoch(self, epoch: int) -> None:
        self._epoch = _require_nonnegative_integer(epoch, name="epoch")

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[GuidedSampleIndex]]:
        yield from self._build_batches(self._epoch)

    def _sample_cycles(self, epoch: int) -> dict[int, _TaskSampleCycle]:
        return {
            task_index: _TaskSampleCycle(
                self._task_sample_index.samples_for_task(task_index),
                seed=self._seed,
                epoch=epoch,
                task_index=task_index,
            )
            for task_index in self._guide_catalog.task_indices
        }

    def _guide_group(
        self,
        guide_indices: Sequence[int],
        validity: Sequence[bool],
        cycles: Mapping[int, _TaskSampleCycle],
    ) -> list[GuidedSampleIndex]:
        result: list[GuidedSampleIndex] = []
        for guide_index, guide_valid in zip(guide_indices, validity, strict=True):
            record = self._guide_catalog.by_guide_index(guide_index)
            if guide_valid:
                task_samples = self._task_sample_index.samples_for_task(
                    record.task_index
                )
                valid_count = min(len(task_samples), self._queries_per_guide)
                valid_samples = cycles[record.task_index].take_distinct(valid_count)
                samples = (
                    *valid_samples,
                    *(
                        [valid_samples[-1]]
                        * (self._queries_per_guide - valid_count)
                    ),
                )
                query_validity = (
                    *([True] * valid_count),
                    *([False] * (self._queries_per_guide - valid_count)),
                )
            else:
                sample = self._task_sample_index.samples_for_task(record.task_index)[0]
                samples = (sample,) * self._queries_per_guide
                query_validity = (False,) * self._queries_per_guide
            result.extend(
                GuidedSampleIndex(
                    sample_index=sample,
                    guide_index=guide_index,
                    query_valid=query_valid,
                )
                for sample, query_valid in zip(samples, query_validity, strict=True)
            )
        return result

    def _build_batches(self, epoch: int) -> list[list[GuidedSampleIndex]]:
        cycles = self._sample_cycles(epoch)
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, epoch, 0x424C4F43])
        )
        bucket_weights = np.asarray(
            [len(guide_indices) for _, guide_indices in self._buckets],
            dtype=np.float64,
        )
        bucket_weights /= np.sum(bucket_weights)
        batches: list[list[GuidedSampleIndex]] = []
        for block_start in range(0, self._num_batches, self._batch_block_size):
            bucket_index = int(rng.choice(len(self._buckets), p=bucket_weights))
            _, guide_indices = self._buckets[bucket_index]
            for batch_index in range(
                block_start,
                block_start + self._batch_block_size,
            ):
                selected = rng.choice(
                    np.asarray(guide_indices, dtype=np.int64),
                    size=self._guides_per_batch,
                    replace=False,
                )
                batch = self._guide_group(
                    [int(value) for value in selected],
                    [True] * self._guides_per_batch,
                    cycles,
                )
                remaining_valid = max(
                    self._valid_queries_per_epoch
                    - batch_index * self.batch_size,
                    0,
                )
                valid_slots = min(remaining_valid, self.batch_size)
                if valid_slots < self.batch_size:
                    batch = [
                        GuidedSampleIndex(
                            sample_index=sample.sample_index,
                            guide_index=sample.guide_index,
                            query_valid=sample.query_valid and slot < valid_slots,
                        )
                        for slot, sample in enumerate(batch)
                    ]
                batches.append(batch)
        return batches

    def _summarize(
        self, batches: Sequence[Sequence[GuidedSampleIndex]]
    ) -> GuidanceFirstBatchStats:
        used_guides = 0
        padded_guides = 0
        mixed_task_batches = 0
        bucket_counts: Counter[str] = Counter()
        guide_to_bucket = {
            guide_index: bucket
            for bucket, guide_indices in self._buckets
            for guide_index in guide_indices
        }
        for batch in batches:
            tasks: set[int] = set()
            first_valid_guide = None
            for start in range(0, len(batch), self._queries_per_guide):
                group = batch[start : start + self._queries_per_guide]
                if any(sample.query_valid for sample in group):
                    used_guides += 1
                    first_valid_guide = group[0].guide_index
                    tasks.add(
                        self._guide_catalog.by_guide_index(
                            group[0].guide_index
                        ).task_index
                    )
                else:
                    padded_guides += 1
            if len(tasks) > 1:
                mixed_task_batches += 1
            if first_valid_guide is None:
                first_valid_guide = batch[0].guide_index
            bucket_counts[str(guide_to_bucket[first_valid_guide])] += 1

        valid_queries = sum(
            sample.query_valid for batch in batches for sample in batch
        )
        total_slots = len(batches) * self.batch_size
        total_native_samples = self._task_sample_index.total_samples
        total_guides = len(self._guide_catalog.records)
        return GuidanceFirstBatchStats(
            guides_per_batch=self._guides_per_batch,
            queries_per_guide=self._queries_per_guide,
            total_native_samples=total_native_samples,
            guide_draws=used_guides,
            padded_guide_slots=padded_guides,
            num_batches=len(batches),
            mixed_task_batches=mixed_task_batches,
            valid_query_samples=valid_queries,
            padded_query_slots=total_slots - valid_queries,
            dropped_query_samples=total_native_samples - valid_queries,
            bucket_batch_counts=tuple(sorted(bucket_counts.items())),
            bucket_probabilities=tuple(
                (
                    str(bucket),
                    len(guide_indices) / total_guides,
                )
                for bucket, guide_indices in self._buckets
            ),
        )
