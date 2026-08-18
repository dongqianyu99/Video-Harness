from __future__ import annotations

import pickle

import jax.numpy as jnp

from openpi.models.guide_inputs import GuideInput
from openpi.training.guide_cache import ConstantResolverFactory
from openpi.training.guide_cache import ProcessLocalGuideResolver
from openpi.training.guide_dataset import GuideBindingRecord


def _guide(marker: int) -> GuideInput:
    return GuideInput(
        images=jnp.full((1, 1, 2, 2, 3), marker, dtype=jnp.float32),
        image_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        text_tokens=jnp.full((1, 1, 2), marker, dtype=jnp.int32),
        text_mask=jnp.ones((1, 1, 2), dtype=jnp.bool_),
        unit_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        before_slot=jnp.zeros((1, 1), dtype=jnp.int32),
        after_slot=jnp.zeros((1, 1), dtype=jnp.int32),
    )


class _Resolver:
    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, record: GuideBindingRecord) -> GuideInput:
        self.calls.append(record.binding_index)
        return _guide(record.binding_index)


def _record(binding_index: int, document_id: str) -> GuideBindingRecord:
    return GuideBindingRecord(
        binding_index=binding_index,
        query_episode_index=10 + binding_index,
        task_index=4,
        support_episode_index=20 + binding_index,
        support_document_id=document_id,
    )


def test_process_local_cache_reuses_same_support_document_across_bindings():
    resolver = _Resolver()
    cached = ProcessLocalGuideResolver(
        resolver_factory=ConstantResolverFactory(resolver),
        max_entries=2,
    )

    first = cached(_record(0, "shared-document"))
    second = cached(_record(1, "shared-document"))

    assert first is second
    assert resolver.calls == [0]
    assert cached.cache_info.hits == 1
    assert cached.cache_info.misses == 1
    assert cached.cache_info.entries == 1
    assert not first.images.flags.writeable


def test_process_local_cache_evicts_lru_by_entry_count():
    resolver = _Resolver()
    cached = ProcessLocalGuideResolver(
        resolver_factory=ConstantResolverFactory(resolver),
        max_entries=1,
    )

    cached(_record(0, "document-a"))
    cached(_record(1, "document-b"))
    cached(_record(2, "document-a"))

    assert resolver.calls == [0, 1, 2]
    assert cached.cache_info.evictions == 2
    assert cached.cache_info.entries == 1


def test_process_local_cache_pickle_drops_runtime_cache():
    resolver = _Resolver()
    cached = ProcessLocalGuideResolver(
        resolver_factory=ConstantResolverFactory(resolver),
        max_entries=2,
    )
    cached(_record(0, "document-a"))

    restored = pickle.loads(pickle.dumps(cached))

    assert restored.cache_info.entries == 0
    restored(_record(0, "document-a"))
    assert restored.cache_info.misses == 1
