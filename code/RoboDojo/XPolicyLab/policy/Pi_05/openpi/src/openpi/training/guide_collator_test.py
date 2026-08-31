from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training.guide_collator import GuidanceBatchCollator
from openpi.training.guide_dataset import GuideCatalog


def _catalog():
    documents = tuple(
        SimpleNamespace(
            document_id=f"doc-{index}",
            source_episode_index=10 + index,
            task_index=index,
            task_instruction=f"task {index}",
        )
        for index in range(2)
    )
    return GuideCatalog.from_document_catalog(
        SimpleNamespace(catalog_digest="digest", documents=documents)
    )


def _guide(marker: int) -> GuideInput:
    return GuideInput(
        boundary_images=jnp.full((1, 2, 3, 2, 2, 3), marker, dtype=jnp.float32),
        boundary_image_mask=jnp.ones((1, 2, 3), dtype=jnp.bool_),
        boundary_text_tokens=jnp.full((1, 2, 3, 2), marker, dtype=jnp.int32),
        boundary_text_mask=jnp.ones((1, 2, 3, 2), dtype=jnp.bool_),
        transition_text_tokens=jnp.full((1, 1, 2), marker, dtype=jnp.int32),
        transition_text_mask=jnp.ones((1, 1, 2), dtype=jnp.bool_),
        boundary_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        unit_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        memory_source_kind=jnp.zeros((1, 20), dtype=jnp.int32),
        memory_source_index=jnp.zeros((1, 20), dtype=jnp.int32),
        memory_source_offset=jnp.zeros((1, 20), dtype=jnp.int32),
        memory_mask=jnp.ones((1, 20), dtype=jnp.bool_),
    )


def _query(value: float):
    return {
        "image": {
            key: np.full((2, 2, 3), value, dtype=np.float32)
            for key in _model.IMAGE_KEYS
        },
        "image_mask": {
            key: np.asarray(np.bool_(1), dtype=np.bool_) for key in _model.IMAGE_KEYS
        },
        "state": np.asarray([value, value + 1], dtype=np.float32),
        "actions": np.full((50, 32), value, dtype=np.float32),
    }


def _item(guide_index: int, value: float, *, valid: bool = True):
    return {
        "query": _query(value),
        "guide_index": np.asarray(guide_index, dtype=np.int32),
        "query_valid": np.asarray(valid, dtype=np.bool_),
    }


class _Resolver:
    def __init__(self):
        self.calls = []

    def __call__(self, record):
        self.calls.append(record)
        return _guide(record.guide_index)


def test_collator_builds_group_major_batch_and_resolves_each_guide_once():
    resolver = _Resolver()
    collator = GuidanceBatchCollator(
        guide_catalog=_catalog(),
        guide_input_resolver=resolver,
        guides_per_batch=2,
        queries_per_guide=2,
    )
    batch = collator(
        [_item(0, 1), _item(0, 2), _item(1, 3), _item(1, 4)]
    )

    assert validate_guide_conditioned_batch(batch) == (2, 2)
    np.testing.assert_array_equal(batch.observation.state[:, :, 0], [[1, 2], [3, 4]])
    assert [record.guide_index for record in resolver.calls] == [0, 1]
    assert batch.guide.boundary_images.shape[0] == 2
    assert all(not isinstance(leaf, str) for leaf in jax.tree_util.tree_leaves(batch))


def test_collator_allows_invalid_padding_but_rejects_duplicate_valid_document():
    catalog = _catalog()
    collator = GuidanceBatchCollator(
        guide_catalog=catalog,
        guide_input_resolver=_Resolver(),
        guides_per_batch=2,
        queries_per_guide=1,
    )
    padded = collator([_item(0, 1), _item(0, 1, valid=False)])
    np.testing.assert_array_equal(padded.query_mask, [[True], [False]])

    with pytest.raises(ValueError, match="distinct document"):
        collator([_item(0, 1), _item(0, 2)])


def test_collator_rejects_mixed_guide_group_and_preserves_resolver_context():
    collator = GuidanceBatchCollator(
        guide_catalog=_catalog(),
        guide_input_resolver=_Resolver(),
        guides_per_batch=1,
        queries_per_guide=2,
    )
    with pytest.raises(ValueError, match="mixes guide_index"):
        collator([_item(0, 1), _item(1, 2)])

    class _Failing:
        def __call__(self, _record):
            raise RuntimeError("boom")

    failing = GuidanceBatchCollator(
        guide_catalog=_catalog(),
        guide_input_resolver=_Failing(),
        guides_per_batch=1,
        queries_per_guide=1,
    )
    with pytest.raises(RuntimeError, match=r"guide_index=0.*doc-0"):
        failing([_item(0, 1)])
