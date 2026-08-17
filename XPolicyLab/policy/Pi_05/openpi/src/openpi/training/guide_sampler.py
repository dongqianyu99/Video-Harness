from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from openpi.training.guide_dataset import GuideBindingIndex


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
