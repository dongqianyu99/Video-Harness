import pickle

from openpi.training.guide_cache import ConstantResolverFactory
from openpi.training.guide_cache import ProcessLocalGuideResolver
from openpi.training.guide_collator_test import _guide
from openpi.training.guide_dataset import GuideRecord


class _Resolver:
    def __init__(self):
        self.calls = []

    def __call__(self, record):
        self.calls.append(record.guide_index)
        return _guide(record.guide_index)


def _record(index: int, document_id: str):
    return GuideRecord(index, document_id, 10 + index, 4, "task four")


def test_process_local_cache_keys_by_document_id_and_evicts_lru():
    resolver = _Resolver()
    cached = ProcessLocalGuideResolver(
        resolver_factory=ConstantResolverFactory(resolver), max_entries=1
    )

    first = cached(_record(0, "document-a"))
    assert cached(_record(0, "document-a")) is first
    cached(_record(1, "document-b"))
    cached(_record(0, "document-a"))

    assert resolver.calls == [0, 1, 0]
    assert cached.cache_info.hits == 1
    assert cached.cache_info.evictions == 2
    assert not first.boundary_images.flags.writeable


def test_process_local_cache_pickle_drops_runtime_state():
    resolver = _Resolver()
    cached = ProcessLocalGuideResolver(
        resolver_factory=ConstantResolverFactory(resolver), max_entries=2
    )
    cached(_record(0, "document-a"))

    restored = pickle.loads(pickle.dumps(cached))

    assert restored.cache_info.entries == 0
    restored(_record(0, "document-a"))
    assert restored.cache_info.misses == 1
