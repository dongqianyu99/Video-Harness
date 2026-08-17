from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral
from typing import Any

import torch

from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training.guide_collator import SingleGuideBatchCollator
from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.guide_dataset import GuideBoundDataset
from openpi.training.guide_sampler import HomogeneousBindingBatchSampler


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


class GuidedDataLoader:
    """Single-process DataLoader for homogeneous Guide-conditioned batches."""

    def __init__(
        self,
        dataset: GuideBoundDataset,
        *,
        batch_sampler: HomogeneousBindingBatchSampler,
        collator: SingleGuideBatchCollator,
        num_batches: int | None = None,
        num_workers: int = 0,
        data_config: Any | None = None,
        binding_index: GuideBindingIndex | None = None,
        host_metadata: dict[str, Any] | None = None,
    ):
        if isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers != 0:
            raise ValueError("GuidedDataLoader only supports num_workers=0")

        if num_batches is not None:
            _require_nonnegative_integer(num_batches, name="num_batches")

        if not callable(getattr(batch_sampler, "set_epoch", None)):
            raise ValueError("batch_sampler must provide set_epoch(epoch)")

        try:
            batch_count = len(batch_sampler)
        except TypeError as exc:
            raise ValueError("batch_sampler must provide __len__()") from exc

        if batch_count <= 0:
            raise ValueError("batch_sampler must not be empty")

        batch_size = getattr(batch_sampler, "batch_size", None)
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("batch_sampler.batch_size must be a positive integer")

        self._dataset = dataset
        self._batch_sampler = batch_sampler
        self._collator = collator
        self._batch_size = int(batch_size)
        self._num_batches = None if num_batches is None else int(num_batches)
        self._epoch = 0
        self._data_config = data_config
        self._binding_index = binding_index
        self._host_metadata = {} if host_metadata is None else dict(host_metadata)
        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=0,
        )

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_sampler(self) -> Any:
        return self._batch_sampler

    @property
    def binding_index(self) -> GuideBindingIndex | None:
        return self._binding_index

    @property
    def host_metadata(self) -> dict[str, Any]:
        return dict(self._host_metadata)

    def data_config(self) -> Any:
        """Return native data config required by stock checkpoint asset saving."""

        if self._data_config is None:
            raise ValueError(
                "GuidedDataLoader has no native data_config; checkpoint assets cannot be saved"
            )
        return self._data_config

    def __len__(self) -> int:
        if self._num_batches is not None:
            return self._num_batches
        return len(self._batch_sampler)

    def __iter__(self) -> Iterator[GuideConditionedBatch]:
        if self._num_batches == 0:
            return

        yielded_total = 0

        while self._num_batches is None or yielded_total < self._num_batches:
            self._batch_sampler.set_epoch(self._epoch)
            epoch_iterator = iter(self._data_loader)
            yielded_in_epoch = 0

            for batch in epoch_iterator:
                if not isinstance(batch, GuideConditionedBatch):
                    raise ValueError(
                        "guided collator must return GuideConditionedBatch, "
                        f"got {type(batch).__name__}"
                    )

                groups, queries = validate_guide_conditioned_batch(batch)
                if groups != 1:
                    raise ValueError(
                        f"GuidedDataLoader requires G=1, got G={groups}"
                    )
                if queries != self._batch_size:
                    raise ValueError(
                        "guided collator returned an unexpected query batch size: "
                        f"expected {self._batch_size}, got {queries}"
                    )

                yielded_in_epoch += 1
                yielded_total += 1
                yield batch

                if (
                    self._num_batches is not None
                    and yielded_total >= self._num_batches
                ):
                    if yielded_in_epoch >= len(self._batch_sampler):
                        self._epoch += 1
                    return

            if yielded_in_epoch == 0:
                raise ValueError("batch_sampler produced no batches")

            self._epoch += 1
