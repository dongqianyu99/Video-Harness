import copy
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training import guide_collator as _guide_collator
from openpi.training.guide_dataset import GuideBindingIndex


@dataclass(frozen=True)
class _SupportBinding:
    query_episode_index: int
    support_episode_index: int
    task_index: int
    support_document_id: str


def _make_binding(
    *,
    query_episode_index: int = 10,
    support_episode_index: int = 11,
    task_index: int = 4,
    support_document_id: str = "document-0",
) -> _SupportBinding:
    return _SupportBinding(
        query_episode_index=query_episode_index,
        support_episode_index=support_episode_index,
        task_index=task_index,
        support_document_id=support_document_id,
    )


def _make_index(*bindings: _SupportBinding) -> GuideBindingIndex:
    return GuideBindingIndex.from_bindings(list(bindings))


def _make_guide(*, groups: int = 1) -> GuideInput:
    return GuideInput(
        images=jnp.zeros((groups, 2, 2, 2, 3), dtype=jnp.float32),
        image_mask=jnp.ones((groups, 2), dtype=jnp.bool_),
        text_tokens=jnp.zeros((groups, 2, 5), dtype=jnp.int32),
        text_mask=jnp.ones((groups, 2, 5), dtype=jnp.bool_),
        unit_mask=jnp.ones((groups, 2), dtype=jnp.bool_),
        before_slot=jnp.zeros((groups, 2), dtype=jnp.int32),
        after_slot=jnp.ones((groups, 2), dtype=jnp.int32),
    )


def _make_query(
    value: float,
    *,
    include_prompt: bool = True,
) -> dict[str, Any]:
    query = {
        "image": {
            key: np.full((2, 2, 3), value + image_index, dtype=np.float32)
            for image_index, key in enumerate(_model.IMAGE_KEYS)
        },
        "image_mask": {
            key: np.asarray(image_index % 2 == 0, dtype=np.bool_)
            for image_index, key in enumerate(_model.IMAGE_KEYS)
        },
        "state": np.asarray([value, value + 1, value + 2, value + 3], dtype=np.float32),
        "actions": np.full((50, 32), value, dtype=np.float32),
    }

    if include_prompt:
        query.update(
            {
                "tokenized_prompt": np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
                "tokenized_prompt_mask": np.asarray(
                    [True, True, True, False, False],
                    dtype=np.bool_,
                ),
                "token_ar_mask": np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
                "token_loss_mask": np.asarray(
                    [False, False, True, True, True],
                    dtype=np.bool_,
                ),
            }
        )

    return query


def _make_item(
    *,
    episode_index: int = 10,
    task_index: int = 4,
    binding_index: int = 0,
    value: float = 0.0,
    include_prompt: bool = True,
) -> dict[str, Any]:
    return {
        "query": _make_query(value, include_prompt=include_prompt),
        "guide_binding_index": np.asarray(binding_index, dtype=np.int32),
        "_episode_for_test": episode_index,
        "_task_for_test": task_index,
    }


def _strip_test_identity(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    item.pop("_episode_for_test")
    item.pop("_task_for_test")
    return item


def _make_bound_item(
    *,
    episode_index: int = 10,
    task_index: int = 4,
    binding_index: int = 0,
    value: float = 0.0,
    include_prompt: bool = True,
) -> dict[str, Any]:
    item = _make_item(
        episode_index=episode_index,
        task_index=task_index,
        binding_index=binding_index,
        value=value,
        include_prompt=include_prompt,
    )
    query = item.pop("query")
    query["episode_index"] = np.asarray(episode_index, dtype=np.int64)
    query["task_index"] = np.asarray(task_index, dtype=np.int64)
    item["query"] = query
    item.pop("_episode_for_test")
    item.pop("_task_for_test")
    return item


class _RecordingResolver:
    def __init__(self, guide: GuideInput | None = None):
        self.calls = []
        self.guide = _make_guide() if guide is None else guide

    def __call__(self, record):
        self.calls.append(record)
        return self.guide


class _FailingResolver:
    def __call__(self, record):
        raise RuntimeError(f"cannot resolve binding_index={record.binding_index}")


def _make_collator(
    *,
    resolver=None,
    bindings: tuple[_SupportBinding, ...] = (_SupportBinding(10, 11, 4, "document-0"),),
):
    if resolver is None:
        resolver = _RecordingResolver()
    return _guide_collator.SingleGuideBatchCollator(
        binding_index=_make_index(*bindings),
        guide_input_resolver=resolver,
    )


def _assert_tree_equal(expected, actual) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected) == set(actual)
        for key in expected:
            _assert_tree_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_child, actual_child in zip(expected, actual, strict=True):
            _assert_tree_equal(expected_child, actual_child)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    else:
        assert expected == actual


def test_single_guide_collator_groups_all_native_observation_fields():
    resolver = _RecordingResolver()
    items = [
        _make_bound_item(value=1.0),
        _make_bound_item(value=2.0),
        _make_bound_item(value=3.0),
    ]

    batch = _make_collator(resolver=resolver)(items)

    assert isinstance(batch, GuideConditionedBatch)
    assert isinstance(batch.observation, _model.Observation)
    assert batch.actions.shape == (1, 3, 50, 32)
    assert batch.observation.state.shape == (1, 3, 4)

    for key in _model.IMAGE_KEYS:
        assert batch.observation.images[key].shape == (1, 3, 2, 2, 3)
        assert batch.observation.image_masks[key].shape == (1, 3)

    assert batch.observation.tokenized_prompt.shape == (1, 3, 5)
    assert batch.observation.tokenized_prompt_mask.shape == (1, 3, 5)
    assert batch.observation.token_ar_mask.shape == (1, 3, 5)
    assert batch.observation.token_loss_mask.shape == (1, 3, 5)
    assert batch.guide.images.shape[0] == 1
    assert len(resolver.calls) == 1
    assert resolver.calls[0].binding_index == 0

    groups, queries = validate_guide_conditioned_batch(batch)
    assert (groups, queries) == (1, 3)

    np.testing.assert_array_equal(
        batch.observation.state[0, :, 0],
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(batch.actions[0, 2], np.full((50, 32), 3.0, dtype=np.float32))


def test_single_guide_collator_preserves_optional_none_observation_fields():
    items = [
        _make_bound_item(value=1.0, include_prompt=False),
        _make_bound_item(value=2.0, include_prompt=False),
    ]

    batch = _make_collator()(items)

    assert batch.observation.tokenized_prompt is None
    assert batch.observation.tokenized_prompt_mask is None
    assert batch.observation.token_ar_mask is None
    assert batch.observation.token_loss_mask is None
    assert batch.actions.shape == (1, 2, 50, 32)


def test_single_guide_collator_keeps_query_order_and_does_not_mutate_items():
    items = [
        _make_bound_item(value=10.0),
        _make_bound_item(value=20.0),
    ]
    before = copy.deepcopy(items)

    batch = _make_collator()(items)

    np.testing.assert_array_equal(batch.observation.state[0, :, 0], [10.0, 20.0])
    _assert_tree_equal(before, items)


def test_single_guide_collator_resolves_guide_exactly_once():
    resolver = _RecordingResolver()
    items = [_make_bound_item(value=float(index)) for index in range(4)]

    _make_collator(resolver=resolver)(items)

    assert len(resolver.calls) == 1
    assert resolver.calls[0].support_document_id == "document-0"


def test_multi_guide_collator_builds_group_major_batch_and_resolves_each_guide_once():
    bindings = (
        _make_binding(
            query_episode_index=10,
            support_episode_index=11,
            task_index=4,
            support_document_id="document-0",
        ),
        _make_binding(
            query_episode_index=20,
            support_episode_index=21,
            task_index=5,
            support_document_id="document-1",
        ),
    )
    resolver = _RecordingResolver()
    collator = _guide_collator.MultiGuideBatchCollator(
        binding_index=_make_index(*bindings),
        guide_input_resolver=resolver,
        guides_per_batch=2,
        queries_per_guide=2,
    )
    items = [
        _make_bound_item(episode_index=10, task_index=4, binding_index=0, value=1.0),
        _make_bound_item(episode_index=10, task_index=4, binding_index=0, value=2.0),
        _make_bound_item(episode_index=20, task_index=5, binding_index=1, value=3.0),
        _make_bound_item(episode_index=20, task_index=5, binding_index=1, value=4.0),
    ]

    batch = collator(items)

    assert validate_guide_conditioned_batch(batch) == (2, 2)
    assert batch.actions.shape == (2, 2, 50, 32)
    np.testing.assert_array_equal(batch.observation.state[:, :, 0], [[1.0, 2.0], [3.0, 4.0]])
    assert batch.guide.images.shape[0] == 2
    assert [record.binding_index for record in resolver.calls] == [0, 1]


def test_multi_guide_collator_rejects_duplicate_support_document():
    bindings = (
        _make_binding(query_episode_index=10, support_episode_index=11),
        _make_binding(
            query_episode_index=20,
            support_episode_index=21,
            support_document_id="document-0",
        ),
    )
    collator = _guide_collator.MultiGuideBatchCollator(
        binding_index=_make_index(*bindings),
        guide_input_resolver=_RecordingResolver(),
        guides_per_batch=2,
        queries_per_guide=1,
    )
    items = [
        _make_bound_item(episode_index=10, binding_index=0),
        _make_bound_item(episode_index=20, binding_index=1),
    ]

    with pytest.raises(ValueError, match="distinct support document"):
        collator(items)


def test_single_guide_collator_rejects_empty_batch_before_resolver():
    resolver = _RecordingResolver()

    with pytest.raises(ValueError, match="empty"):
        _make_collator(resolver=resolver)([])

    assert resolver.calls == []


def test_single_guide_collator_rejects_mixed_bindings_before_resolver():
    resolver = _RecordingResolver()
    bindings = (
        _make_binding(query_episode_index=10, support_episode_index=11),
        _make_binding(
            query_episode_index=20,
            support_episode_index=21,
            support_document_id="document-1",
        ),
    )
    items = [
        _make_bound_item(episode_index=10, binding_index=0, value=1.0),
        _make_bound_item(episode_index=20, binding_index=1, value=2.0),
    ]

    with pytest.raises(ValueError, match="binding"):
        _make_collator(resolver=resolver, bindings=bindings)(items)

    assert resolver.calls == []


@pytest.mark.parametrize(
    "item",
    [
        {"query": {}},
        {"guide_binding_index": np.asarray(0, dtype=np.int32)},
        {
            "query": {},
            "guide_binding_index": np.asarray(0, dtype=np.int32),
            "extra": np.asarray(1, dtype=np.int32),
        },
    ],
)
def test_single_guide_collator_requires_exact_item_keys(item):
    with pytest.raises(ValueError, match=r"query|guide_binding_index|keys"):
        _make_collator()([item])


@pytest.mark.parametrize(
    "binding_value",
    [
        np.asarray([0], dtype=np.int32),
        np.asarray(1, dtype=np.bool_),
        np.asarray(0.0, dtype=np.float32),
        np.asarray(-1, dtype=np.int32),
    ],
)
def test_single_guide_collator_rejects_invalid_binding_scalar(binding_value):
    item = _make_bound_item()
    item["guide_binding_index"] = binding_value

    with pytest.raises(ValueError, match="binding"):
        _make_collator()([item])


@pytest.mark.parametrize("query", [None, [], "not-a-mapping"])
def test_single_guide_collator_rejects_non_mapping_query(query):
    item = _make_bound_item()
    item["query"] = query

    with pytest.raises(ValueError, match=r"query|mapping"):
        _make_collator()([item])


def test_single_guide_collator_rejects_invalid_guide_shape():
    resolver = _RecordingResolver(guide=_make_guide(groups=2))

    with pytest.raises(ValueError, match=r"1|group"):
        _make_collator(resolver=resolver)([_make_bound_item()])

    assert len(resolver.calls) == 1


def test_single_guide_collator_requires_resolver_to_return_guide_input():
    class _WrongResolver:
        def __call__(self, _record):
            return {"images": np.zeros((1, 1, 1, 1, 3), dtype=np.float32)}

    with pytest.raises(ValueError, match="GuideInput"):
        _make_collator(resolver=_WrongResolver())([_make_bound_item()])


def test_single_guide_collator_preserves_binding_context_on_resolver_failure():
    with pytest.raises(RuntimeError, match="binding_index=0"):
        _make_collator(resolver=_FailingResolver())([_make_bound_item()])


def test_single_guide_collator_rejects_unknown_binding_before_resolver():
    resolver = _RecordingResolver()
    item = _make_bound_item(binding_index=99)

    with pytest.raises(ValueError, match="binding_index"):
        _make_collator(resolver=resolver)([item])

    assert resolver.calls == []


def test_single_guide_collator_requires_actions():
    item = _make_bound_item()
    item["query"].pop("actions")

    with pytest.raises(ValueError, match="actions"):
        _make_collator()([item])


def test_single_guide_collator_rejects_invalid_action_rank():
    item = _make_bound_item()
    item["query"]["actions"] = np.zeros((50,), dtype=np.float32)

    with pytest.raises(ValueError, match=r"actions|shape"):
        _make_collator()([item])


def test_single_guide_collator_rejects_incompatible_query_leaf_shapes():
    first = _make_bound_item(value=1.0)
    second = _make_bound_item(value=2.0)
    second["query"]["state"] = np.zeros((5,), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape|stack"):
        _make_collator()([first, second])


def test_single_guide_collator_output_contains_only_numeric_model_leaves():
    batch = _make_collator()([_make_bound_item()])
    leaves = jax.tree_util.tree_leaves(batch)

    assert leaves
    assert all(not isinstance(leaf, str) for leaf in leaves)
