from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

import jax
import numpy as np

from openpi.models.guide_inputs import GuideInput
from openpi.training.guide_dataset import GuideBindingRecord


@dataclass(frozen=True, slots=True)
class GuideCacheInfo:
    hits: int
    misses: int
    entries: int
    bytes: int
    evictions: int


def _guide_nbytes(guide: GuideInput) -> int:
    return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(guide))


def _freeze_guide(guide: GuideInput) -> GuideInput:
    def freeze(value: Any) -> np.ndarray:
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array

    return jax.tree_util.tree_map(freeze, guide)


class ProcessLocalGuideResolver:
    """Lazy per-process resolver with a bounded materialized-Guide LRU.

    PyTorch ``spawn`` workers receive only the resolver factory.  The actual
    VideoHarness bundle, tokenizer, FFmpeg loader, and cache are created after
    the worker starts and are never shared across process boundaries.
    """

    def __init__(
        self,
        *,
        resolver_factory: Callable[[], Callable[[GuideBindingRecord], GuideInput]],
        max_entries: int = 0,
        max_bytes: int = 0,
    ) -> None:
        if not callable(resolver_factory):
            raise ValueError("resolver_factory must be callable")
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

        self._resolver_factory = resolver_factory
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self._owner_pid: int | None = None
        self._resolver: Callable[[GuideBindingRecord], GuideInput] | None = None
        self._cache: OrderedDict[str, tuple[GuideInput, int]] = OrderedDict()
        self._cache_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_resolver_factory": self._resolver_factory,
            "_max_entries": self._max_entries,
            "_max_bytes": self._max_bytes,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._resolver_factory = state["_resolver_factory"]
        self._max_entries = state["_max_entries"]
        self._max_bytes = state["_max_bytes"]
        self._reset_runtime()

    def _ensure_process(self) -> Callable[[GuideBindingRecord], GuideInput]:
        pid = os.getpid()
        if self._owner_pid != pid or self._resolver is None:
            self._reset_runtime()
            self._owner_pid = pid
            self._resolver = self._resolver_factory()
            if not callable(self._resolver):
                raise ValueError("resolver_factory must return a callable resolver")
        return self._resolver

    @property
    def cache_info(self) -> GuideCacheInfo:
        return GuideCacheInfo(
            hits=self._hits,
            misses=self._misses,
            entries=len(self._cache),
            bytes=self._cache_bytes,
            evictions=self._evictions,
        )

    def _cache_enabled(self) -> bool:
        return self._max_entries > 0

    def _insert(self, key: str, guide: GuideInput) -> GuideInput:
        frozen = _freeze_guide(guide)
        size = _guide_nbytes(frozen)
        if self._max_bytes > 0 and size > self._max_bytes:
            return frozen

        self._cache[key] = (frozen, size)
        self._cache.move_to_end(key)
        self._cache_bytes += size

        while len(self._cache) > self._max_entries or (
            self._max_bytes > 0 and self._cache_bytes > self._max_bytes
        ):
            _, (_, evicted_size) = self._cache.popitem(last=False)
            self._cache_bytes -= evicted_size
            self._evictions += 1
        return frozen

    def __call__(self, record: GuideBindingRecord) -> GuideInput:
        resolver = self._ensure_process()
        key = record.support_document_id
        if self._cache_enabled() and key in self._cache:
            guide, size = self._cache.pop(key)
            self._cache[key] = (guide, size)
            self._hits += 1
            return guide

        self._misses += 1
        guide = resolver(record)
        if not isinstance(guide, GuideInput):
            raise ValueError(
                f"Guide resolver returned {type(guide).__name__}, expected GuideInput"
            )
        if not self._cache_enabled():
            return guide
        return self._insert(key, guide)


class ConstantResolverFactory:
    """Pickle-friendly adapter for single-process tests and custom resolvers."""

    def __init__(self, resolver: Callable[[GuideBindingRecord], GuideInput]) -> None:
        if not callable(resolver):
            raise ValueError("resolver must be callable")
        self._resolver = resolver

    def __call__(self) -> Callable[[GuideBindingRecord], GuideInput]:
        return self._resolver
