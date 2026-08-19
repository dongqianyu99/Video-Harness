from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.guide_dataset import GuideBindingRecord

_REQUIRED_ITEM_KEYS = frozenset(("query", "guide_binding_index"))
_OPTIONAL_ITEM_KEYS = frozenset(("query_valid",))


def _read_binding_index(
    value: Any,
    *,
    item_index: int,
) -> int:
    binding_value = np.asarray(value)

    if binding_value.ndim != 0:
        raise ValueError(
            f"item {item_index} guide_binding_index must be scalar, "
            f"got shape {binding_value.shape}"
        )

    if not np.issubdtype(binding_value.dtype, np.integer):
        raise ValueError(
            f"item {item_index} guide_binding_index must have integer dtype, "
            f"got {binding_value.dtype}"
        )

    binding_index = int(binding_value.item())

    if binding_index < 0:
        raise ValueError(
            f"item {item_index} guide_binding_index must be non-negative, "
            f"got {binding_index}"
        )

    return binding_index


def _validate_items(
    items: Sequence[Any],
) -> tuple[tuple[Mapping[str, Any], ...], int, np.ndarray]:
    if not items:
        raise ValueError("cannot collate an empty batch")

    queries: list[Mapping[str, Any]] = []
    query_validity: list[bool] = []
    shared_binding_index: int | None = None

    for item_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"item {item_index} must be a mapping, "
                f"got {type(item).__name__}"
            )

        item_keys = set(item)
        if not item_keys >= _REQUIRED_ITEM_KEYS or not item_keys <= (
            _REQUIRED_ITEM_KEYS | _OPTIONAL_ITEM_KEYS
        ):
            raise ValueError(
                f"item {item_index} must contain {sorted(_REQUIRED_ITEM_KEYS)} "
                f"and only optional {sorted(_OPTIONAL_ITEM_KEYS)}, got {sorted(item)}"
            )

        query = item["query"]
        if not isinstance(query, Mapping):
            raise ValueError(
                f"item {item_index} query must be a mapping, "
                f"got {type(query).__name__}"
            )

        binding_index = _read_binding_index(
            item["guide_binding_index"],
            item_index=item_index,
        )

        if shared_binding_index is None:
            shared_binding_index = binding_index
        elif binding_index != shared_binding_index:
            raise ValueError(
                "all items in a single-guide batch must share one "
                f"guide_binding_index, got {shared_binding_index} and "
                f"{binding_index} at item {item_index}"
            )

        queries.append(query)
        validity = np.asarray(item.get("query_valid", True))
        if validity.ndim != 0 or validity.dtype != np.bool_:
            raise ValueError(
                f"item {item_index} query_valid must be a scalar bool"
            )
        query_validity.append(bool(validity.item()))

    assert shared_binding_index is not None
    return (
        tuple(queries),
        shared_binding_index,
        np.asarray(query_validity, dtype=np.bool_),
    )


def _validate_grouped_items(
    items: Sequence[Any],
    *,
    guides_per_batch: int,
    queries_per_guide: int,
    binding_index: GuideBindingIndex,
) -> tuple[
    tuple[tuple[Mapping[str, Any], ...], ...],
    tuple[int, ...],
    np.ndarray,
]:
    expected_items = guides_per_batch * queries_per_guide
    if len(items) != expected_items:
        raise ValueError(
            f"grouped batch must contain G*Q={expected_items} items, got {len(items)}"
        )

    query_groups: list[tuple[Mapping[str, Any], ...]] = []
    binding_indices: list[int] = []
    query_masks: list[np.ndarray] = []
    document_ids: set[str] = set()

    for group_index in range(guides_per_batch):
        start = group_index * queries_per_guide
        stop = start + queries_per_guide
        group_queries, current_binding_index, group_query_mask = _validate_items(
            items[start:stop]
        )
        record = binding_index.by_binding_index(current_binding_index)
        if record.support_document_id in document_ids and bool(np.any(group_query_mask)):
            raise ValueError(
                "each Guide group must use a distinct support document, got duplicate "
                f"{record.support_document_id!r}"
            )
        if bool(np.any(group_query_mask)):
            document_ids.add(record.support_document_id)
        query_groups.append(group_queries)
        binding_indices.append(current_binding_index)
        query_masks.append(group_query_mask)

    return (
        tuple(query_groups),
        tuple(binding_indices),
        np.stack(query_masks, axis=0),
    )


def _stack_tree(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty sequence")

    first = values[0]

    if isinstance(first, Mapping):
        expected_keys = set(first)

        for value in values:
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise ValueError("all nested query mappings must have the same keys")

        return {
            key: _stack_tree([value[key] for value in values])
            for key in first
        }

    if first is None:
        if any(value is not None for value in values):
            raise ValueError("query leaves must consistently be None or arrays")
        return None

    if any(value is None or isinstance(value, Mapping) for value in values):
        raise ValueError("query leaves must have consistent array structure")

    try:
        return np.stack(
            [np.asarray(value) for value in values],
            axis=0,
        )
    except ValueError as exc:
        raise ValueError(
            "query leaves have incompatible shapes and cannot be stacked"
        ) from exc


def _stack_query_items(
    queries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not queries:
        raise ValueError("cannot stack an empty query sequence")

    stacked = _stack_tree(queries)

    if not isinstance(stacked, Mapping):
        raise ValueError("stacked query must remain a mapping")

    return stacked


def _add_group_dimension(value: Any) -> Any:
    if value is None:
        return None
    return value[None, ...]


class SingleGuideBatchCollator:
    def __init__(
        self,
        *,
        binding_index: GuideBindingIndex,
        guide_input_resolver: Callable[[GuideBindingRecord], GuideInput],
    ):
        if not callable(guide_input_resolver):
            raise ValueError("guide_input_resolver must be callable")

        self._binding_index = binding_index
        self._guide_input_resolver = guide_input_resolver

    def __call__(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> GuideConditionedBatch:
        queries, binding_index, query_mask = _validate_items(items)
        collated_query = dict(_stack_query_items(queries))

        if "actions" not in collated_query:
            raise ValueError("collated query must contain actions")

        actions = collated_query.pop("actions")
        observation = _model.Observation.from_dict(collated_query)

        grouped_observation = jax.tree_util.tree_map(
            _add_group_dimension,
            observation,
        )
        grouped_actions = actions[None, ...]

        record = self._binding_index.by_binding_index(binding_index)

        try:
            guide = self._guide_input_resolver(record)
        except Exception as exc:
            raise RuntimeError(
                f"failed to resolve Guide for binding_index={binding_index}"
            ) from exc

        if not isinstance(guide, GuideInput):
            raise ValueError(
                "guide_input_resolver must return a GuideInput"
            )

        batch = GuideConditionedBatch(
            observation=grouped_observation,
            actions=grouped_actions,
            guide=guide,
            query_mask=query_mask[None, ...],
        )

        groups, queries_count = validate_guide_conditioned_batch(batch)

        if groups != 1:
            raise ValueError(
                f"SingleGuideBatchCollator requires G=1, got G={groups}"
            )

        if queries_count != len(items):
            raise ValueError(
                f"collated query count {queries_count} does not match "
                f"item count {len(items)}"
            )

        return batch


def _concatenate_guides(guides: Sequence[GuideInput]) -> GuideInput:
    if not guides:
        raise ValueError("cannot concatenate an empty Guide sequence")

    for index, guide in enumerate(guides):
        if not isinstance(guide, GuideInput):
            raise ValueError(
                f"resolved guide {index} must be GuideInput, got {type(guide).__name__}"
            )
        leaves = jax.tree_util.tree_leaves(guide)
        if any(not hasattr(leaf, "shape") or leaf.shape[0] != 1 for leaf in leaves):
            raise ValueError(
                f"resolved guide {index} must have exactly one leading Guide group"
            )

    return jax.tree_util.tree_map(
        lambda *values: np.concatenate(
            [np.asarray(value) for value in values], axis=0
        ),
        *guides,
    )


class MultiGuideBatchCollator:
    """Collate one fixed ``G guides x Q queries`` batch.

    The sampler is responsible for group-major item ordering.  This collator
    validates that ordering, resolves each distinct Guide exactly once, and
    returns the grouped tensor contract consumed by ``GuidePi0``.
    """

    def __init__(
        self,
        *,
        binding_index: GuideBindingIndex,
        guide_input_resolver: Callable[[GuideBindingRecord], GuideInput],
        guides_per_batch: int,
        queries_per_guide: int,
    ):
        if not callable(guide_input_resolver):
            raise ValueError("guide_input_resolver must be callable")
        for name, value in (
            ("guides_per_batch", guides_per_batch),
            ("queries_per_guide", queries_per_guide),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        self._binding_index = binding_index
        self._guide_input_resolver = guide_input_resolver
        self._guides_per_batch = guides_per_batch
        self._queries_per_guide = queries_per_guide

    def __call__(self, items: Sequence[Mapping[str, Any]]) -> GuideConditionedBatch:
        query_groups, binding_indices, query_mask = _validate_grouped_items(
            items,
            guides_per_batch=self._guides_per_batch,
            queries_per_guide=self._queries_per_guide,
            binding_index=self._binding_index,
        )

        stacked_groups = [
            _stack_query_items(group_queries)
            for group_queries in query_groups
        ]
        collated_query = dict(_stack_tree(stacked_groups))
        if "actions" not in collated_query:
            raise ValueError("collated query must contain actions")

        actions = collated_query.pop("actions")
        observation = _model.Observation.from_dict(collated_query)

        resolved_guides: list[GuideInput] = []
        for current_binding_index in binding_indices:
            record = self._binding_index.by_binding_index(current_binding_index)
            try:
                guide = self._guide_input_resolver(record)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to resolve Guide for binding_index={current_binding_index}"
                ) from exc
            resolved_guides.append(guide)

        batch = GuideConditionedBatch(
            observation=observation,
            actions=actions,
            guide=_concatenate_guides(resolved_guides),
            query_mask=query_mask,
        )
        groups, queries = validate_guide_conditioned_batch(batch)
        if (groups, queries) != (
            self._guides_per_batch,
            self._queries_per_guide,
        ):
            raise ValueError(
                "collated grouped shape mismatch: expected "
                f"{(self._guides_per_batch, self._queries_per_guide)}, "
                f"got {(groups, queries)}"
            )
        return batch
