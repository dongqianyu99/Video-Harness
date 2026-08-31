from __future__ import annotations

from collections import deque
from collections.abc import Iterator
import functools
import multiprocessing
from numbers import Integral
import os
from typing import Any

import jax
import torch

from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import GuidedDataset


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


class GuidedDataLoader:
    """Worker-backed loader for fixed ``G guides x Q queries`` batches."""

    def __init__(
        self,
        dataset: GuidedDataset,
        *,
        batch_sampler: Any,
        collator: Any,
        num_batches: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        worker_timeout_s: float = 0.0,
        worker_torch_threads: int = 1,
        data_config: Any | None = None,
        guide_catalog: GuideCatalog | None = None,
        host_metadata: dict[str, Any] | None = None,
    ):
        if isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, Integral)
            or prefetch_factor <= 0
        ):
            raise ValueError("prefetch_factor must be a positive integer")
        if not isinstance(persistent_workers, bool):
            raise ValueError("persistent_workers must be bool")
        if isinstance(worker_timeout_s, bool) or not isinstance(worker_timeout_s, (int, float)) or worker_timeout_s < 0:
            raise ValueError("worker_timeout_s must be non-negative")
        if (
            isinstance(worker_torch_threads, bool)
            or not isinstance(worker_torch_threads, Integral)
            or worker_torch_threads <= 0
        ):
            raise ValueError("worker_torch_threads must be a positive integer")

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
        self._groups_per_batch = int(getattr(batch_sampler, "guides_per_batch", 1))
        self._queries_per_guide = int(
            getattr(batch_sampler, "queries_per_guide", self._batch_size)
        )
        if self._groups_per_batch * self._queries_per_guide != self._batch_size:
            raise ValueError(
                "batch_sampler grouped dimensions do not match batch_size: "
                f"G={self._groups_per_batch}, Q={self._queries_per_guide}, "
                f"batch_size={self._batch_size}"
            )
        self._num_batches = None if num_batches is None else int(num_batches)
        self._epoch = 0
        self._data_config = data_config
        self._guide_catalog = guide_catalog
        self._host_metadata = {} if host_metadata is None else dict(host_metadata)
        loader_kwargs: dict[str, Any] = {}
        if num_workers > 0:
            loader_kwargs.update(
                multiprocessing_context=multiprocessing.get_context("spawn"),
                persistent_workers=persistent_workers,
                prefetch_factor=int(prefetch_factor),
            )
        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=int(num_workers),
            worker_init_fn=functools.partial(
                _guided_worker_init_fn,
                torch_threads=int(worker_torch_threads),
            ),
            timeout=float(worker_timeout_s),
            **loader_kwargs,
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
    def groups_per_batch(self) -> int:
        return self._groups_per_batch

    @property
    def queries_per_guide(self) -> int:
        return self._queries_per_guide

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def guide_catalog(self) -> GuideCatalog | None:
        return self._guide_catalog

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
                if groups != self._groups_per_batch:
                    raise ValueError(
                        "guided collator returned an unexpected Guide group count: "
                        f"expected {self._groups_per_batch}, got {groups}"
                    )
                if queries != self._queries_per_guide:
                    raise ValueError(
                        "guided collator returned an unexpected query batch size: "
                        f"expected {self._queries_per_guide}, got {queries}"
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


def _guided_worker_init_fn(_worker_id: int, *, torch_threads: int) -> None:
    """Avoid accidental JAX GPU preallocation inside data workers."""

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    torch.set_num_threads(torch_threads)


def prefetch_guided_batches(
    iterator: Iterator[GuideConditionedBatch],
    *,
    sharding: GuideConditionedBatch,
    size: int,
) -> Iterator[GuideConditionedBatch]:
    """Asynchronously enqueue a bounded number of grouped batches on device."""

    if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
        raise ValueError("device prefetch size must be a positive integer")

    queue: deque[GuideConditionedBatch] = deque()
    exhausted = False
    for _ in range(int(size)):
        try:
            queue.append(jax.device_put(next(iterator), sharding))
        except StopIteration:
            exhausted = True
            break

    while queue:
        yield queue.popleft()
        if not exhausted:
            try:
                queue.append(jax.device_put(next(iterator), sharding))
            except StopIteration:
                exhausted = True
