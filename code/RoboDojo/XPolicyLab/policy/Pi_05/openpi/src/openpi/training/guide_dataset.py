from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GuideBindingRecord:
    binding_index: int
    query_episode_index: int
    task_index: int
    support_episode_index: int
    support_document_id: str


def _require_non_negative_integral(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        )

    return int(value)


def _binding_field(binding: Any, name: str) -> Any:
    try:
        return getattr(binding, name)
    except AttributeError as exc:
        raise ValueError(
            f"binding must provide field {name!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class GuideBindingIndex:
    _records: tuple[GuideBindingRecord, ...]
    _by_query: Mapping[int, GuideBindingRecord]
    _by_binding: Mapping[int, GuideBindingRecord]

    @classmethod
    def from_bindings(
        cls,
        bindings: Sequence[Any],
    ) -> GuideBindingIndex:
        validated_bindings: list[tuple[int, int, int, str]] = []

        for binding in bindings:
            query_episode_index = _require_non_negative_integral(
                _binding_field(binding, "query_episode_index"),
                name="query_episode_index",
            )
            support_episode_index = _require_non_negative_integral(
                _binding_field(binding, "support_episode_index"),
                name="support_episode_index",
            )
            task_index = _require_non_negative_integral(
                _binding_field(binding, "task_index"),
                name="task_index",
            )
            support_document_id = _binding_field(
                binding,
                "support_document_id",
            )

            if not isinstance(support_document_id, str) or not support_document_id.strip():
                raise ValueError(
                    "support_document_id must be a non-empty string"
                )

            if query_episode_index == support_episode_index:
                raise ValueError(
                    "query and support episodes must be different"
                )

            validated_bindings.append(
                (
                    query_episode_index,
                    support_episode_index,
                    task_index,
                    support_document_id,
                )
            )

        validated_bindings.sort(key=lambda item: item[0])

        records: list[GuideBindingRecord] = []
        seen_queries: set[int] = set()

        for binding_index, (
            query_episode_index,
            support_episode_index,
            task_index,
            support_document_id,
        ) in enumerate(validated_bindings):
            if query_episode_index in seen_queries:
                raise ValueError(
                    f"duplicate query_episode_index={query_episode_index}"
                )

            seen_queries.add(query_episode_index)
            records.append(
                GuideBindingRecord(
                    binding_index=binding_index,
                    query_episode_index=query_episode_index,
                    task_index=task_index,
                    support_episode_index=support_episode_index,
                    support_document_id=support_document_id,
                )
            )

        records_tuple = tuple(records)

        return cls(
            _records=records_tuple,
            _by_query=MappingProxyType(
                {
                    record.query_episode_index: record
                    for record in records_tuple
                }
            ),
            _by_binding=MappingProxyType(
                {
                    record.binding_index: record
                    for record in records_tuple
                }
            ),
        )

    @property
    def records(self) -> tuple[GuideBindingRecord, ...]:
        return self._records

    def __reduce__(self):
        """Rebuild immutable lookup tables inside PyTorch spawn workers."""

        return (type(self).from_bindings, (self._records,))

    def by_query_episode(
        self,
        query_episode_index: int,
    ) -> GuideBindingRecord:
        query_episode_index = _require_non_negative_integral(
            query_episode_index,
            name="query_episode_index",
        )

        try:
            return self._by_query[query_episode_index]
        except KeyError as exc:
            raise ValueError(
                f"unknown query_episode_index={query_episode_index}"
            ) from exc

    def by_binding_index(
        self,
        binding_index: int,
    ) -> GuideBindingRecord:
        binding_index = _require_non_negative_integral(
            binding_index,
            name="binding_index",
        )

        try:
            return self._by_binding[binding_index]
        except KeyError as exc:
            raise ValueError(
                f"unknown binding_index={binding_index}"
            ) from exc


def _sample_identity_index(
    sample: Mapping[str, Any],
    name: str,
) -> int:
    if name not in sample:
        raise ValueError(f"sample is missing {name!r}")

    value = np.asarray(sample[name])

    if value.ndim != 0:
        raise ValueError(
            f"sample {name!r} must be scalar, got shape {value.shape}"
        )

    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(
            f"sample {name!r} must have an integer dtype, got {value.dtype}"
        )

    integer_value = int(value.item())

    if integer_value < 0:
        raise ValueError(
            f"sample {name!r} must be non-negative, got {integer_value}"
        )

    return integer_value


class GuideBoundDataset:
    def __init__(
        self,
        dataset: Any,
        binding_index: GuideBindingIndex,
    ):
        self._dataset = dataset
        self._binding_index = binding_index

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._dataset[index]

        if not isinstance(sample, Mapping):
            raise ValueError(
                f"dataset sample must be a mapping, got {type(sample).__name__}"
            )

        episode_index = _sample_identity_index(
            sample,
            "episode_index",
        )
        task_index = _sample_identity_index(
            sample,
            "task_index",
        )

        binding = self._binding_index.by_query_episode(episode_index)

        if task_index != binding.task_index:
            raise ValueError(
                f"sample task_index={task_index} does not match "
                f"binding task_index={binding.task_index} "
                f"for query_episode_index={episode_index}"
            )

        query = dict(sample)
        query.pop("episode_index")
        query.pop("task_index")

        return {
            "query": query,
            "guide_binding_index": np.asarray(
                binding.binding_index,
                dtype=np.int32,
            ),
        }
