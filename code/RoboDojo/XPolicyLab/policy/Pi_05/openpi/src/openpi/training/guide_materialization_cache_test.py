from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import numpy as np

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training.guide_dataset import GuideRecord
from openpi.training.guide_materialization_cache import CachedGuideResolverFactory
from openpi.training.guide_materialization_cache import ensure_guide_materialization_cache
from openpi.training.guide_materialization_cache import open_guide_materialization_cache


class _Tokenizer:
    cache_digest = "test-tokenizer"


class _Catalog:
    def __init__(self):
        self.source = SimpleNamespace(
            document_id="doc",
            document_sha256="document-sha",
        )

    def by_document_id(self, document_id):
        assert document_id == "doc"
        return self.source


def _config(*, max_boundaries=2, max_units=1):
    return GuideMaterializerConfig(
        max_boundaries=max_boundaries,
        max_units=max_units,
        max_boundary_text_tokens=4,
        max_transition_text_tokens=5,
        boundary_num_queries=2,
        transition_num_queries=1,
    )


def _plan():
    return SimpleNamespace(
        document_id="doc",
        source_episode_index=10,
        task_index=3,
        task_instruction="stack blocks",
        boundaries=tuple(
            SimpleNamespace(
                boundary_id=f"b{index}",
                order=index,
                slot=index,
                episode_frame_index=index * 10,
                timestamp_s=index * 0.4,
                view_texts=(f"h{index}", f"l{index}", f"r{index}"),
            )
            for index in range(2)
        ),
        units=(
            SimpleNamespace(
                unit_id="u0",
                order=0,
                before_slot=0,
                after_slot=1,
                transition_text="move then place",
            ),
        ),
    )


def _guide(config):
    images = np.arange(
        2 * 3 * 224 * 224 * 3,
        dtype=np.float32,
    ).reshape(1, 2, 3, 224, 224, 3)
    images = images / images.max() * 2.0 - 1.0
    boundary_tokens = np.zeros((1, 2, 3, 4), dtype=np.int32)
    boundary_tokens[..., :2] = (1, 2)
    boundary_text_mask = boundary_tokens != 0
    transition_tokens = np.asarray([[[3, 4, 0, 0, 0]]], dtype=np.int32)
    transition_text_mask = transition_tokens != 0
    sequence = config.max_boundaries * 2 + config.max_units
    source_kind = np.zeros((1, sequence), dtype=np.int32)
    source_index = np.zeros((1, sequence), dtype=np.int32)
    source_offset = np.zeros((1, sequence), dtype=np.int32)
    memory_mask = np.zeros((1, sequence), dtype=np.bool_)
    source_kind[0, :5] = (0, 0, 1, 0, 0)
    source_index[0, :5] = (0, 0, 0, 1, 1)
    source_offset[0, :5] = (0, 1, 0, 0, 1)
    memory_mask[0, :5] = True
    return GuideInput(
        boundary_images=np.pad(
            images,
            ((0, 0), (0, config.max_boundaries - 2), (0, 0), (0, 0), (0, 0), (0, 0)),
            constant_values=-1.0,
        ),
        boundary_image_mask=np.pad(
            np.ones((1, 2, 3), dtype=np.bool_),
            ((0, 0), (0, config.max_boundaries - 2), (0, 0)),
        ),
        boundary_text_tokens=np.pad(
            boundary_tokens,
            ((0, 0), (0, config.max_boundaries - 2), (0, 0), (0, 0)),
        ),
        boundary_text_mask=np.pad(
            boundary_text_mask,
            ((0, 0), (0, config.max_boundaries - 2), (0, 0), (0, 0)),
        ),
        transition_text_tokens=np.pad(
            transition_tokens,
            ((0, 0), (0, config.max_units - 1), (0, 0)),
        ),
        transition_text_mask=np.pad(
            transition_text_mask,
            ((0, 0), (0, config.max_units - 1), (0, 0)),
        ),
        boundary_mask=np.asarray(
            [[True, True, *([False] * (config.max_boundaries - 2))]],
            dtype=np.bool_,
        ),
        unit_mask=np.asarray(
            [[True, *([False] * (config.max_units - 1))]],
            dtype=np.bool_,
        ),
        memory_source_kind=source_kind,
        memory_source_index=source_index,
        memory_source_offset=source_offset,
        memory_mask=memory_mask,
    )


def _ensure(cache_root: Path, *, config=None, resolver=None):
    config = _config() if config is None else config
    record = GuideRecord(0, "doc", 10, 3, "stack blocks")
    plan = _plan()
    tokenizer = _Tokenizer()
    return ensure_guide_materialization_cache(
        cache_root=cache_root,
        catalog_digest="catalog-digest",
        guide_records=(record,),
        document_catalog=_Catalog(),
        plans_by_document={"doc": plan},
        materializer_config=config,
        boundary_tokenizer=tokenizer,
        transition_tokenizer=tokenizer,
        source_resolver=(lambda _record: _guide(config)) if resolver is None else resolver,
    )


def test_cache_round_trip_is_exact_and_second_ensure_never_materializes(tmp_path):
    config = _config()
    first = _ensure(tmp_path)
    calls = []
    second = _ensure(
        tmp_path,
        resolver=lambda record: calls.append(record) or _guide(config),
    )
    resolver = pickle.loads(
        pickle.dumps(
            CachedGuideResolverFactory(
                guide_records=(GuideRecord(0, "doc", 10, 3, "stack blocks"),),
                artifact_records=second.records,
                materializer_config=config,
            )
        )
    )()
    restored = resolver(GuideRecord(0, "doc", 10, 3, "stack blocks"))

    assert first.stats["built"] == 1
    assert second.stats["reused"] == 1
    assert calls == []
    for expected, actual in zip(
        jax.tree_util.tree_leaves(_guide(config)),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_corrupt_artifact_is_rebuilt_before_cache_returns(tmp_path):
    first = _ensure(tmp_path)
    first.records[0].artifact_path.write_bytes(b"corrupt")
    calls = []

    repaired = _ensure(
        tmp_path,
        resolver=lambda record: calls.append(record.document_id) or _guide(_config()),
    )

    assert calls == ["doc"]
    assert repaired.stats["rebuilt_corrupt"] == 1


def test_compact_artifact_reuses_under_larger_shared_shape_and_read_only_open(tmp_path):
    first = _ensure(tmp_path)
    larger = _config(max_boundaries=3, max_units=2)
    reused = _ensure(
        tmp_path,
        config=larger,
        resolver=lambda _record: (_ for _ in ()).throw(AssertionError("must reuse")),
    )
    tokenizer = _Tokenizer()
    opened = open_guide_materialization_cache(
        cache_root=tmp_path,
        catalog_digest="catalog-digest",
        guide_records=(GuideRecord(0, "doc", 10, 3, "stack blocks"),),
        document_catalog=_Catalog(),
        plans_by_document={"doc": _plan()},
        materializer_config=larger,
        boundary_tokenizer=tokenizer,
        transition_tokenizer=tokenizer,
    )
    resolver = CachedGuideResolverFactory(
        guide_records=(GuideRecord(0, "doc", 10, 3, "stack blocks"),),
        artifact_records=opened.records,
        materializer_config=larger,
    )()

    assert first.records[0].artifact_key == reused.records[0].artifact_key
    assert opened.cache_digest == reused.cache_digest
    assert resolver(GuideRecord(0, "doc", 10, 3, "stack blocks")).boundary_mask.shape == (1, 3)
