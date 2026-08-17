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

_ITEM_KEYS = frozenset(("query", "guide_binding_index"))


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
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    if not items:
        raise ValueError("cannot collate an empty batch")

    queries: list[Mapping[str, Any]] = []
    shared_binding_index: int | None = None

    for item_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"item {item_index} must be a mapping, "
                f"got {type(item).__name__}"
            )

        if set(item) != _ITEM_KEYS:
            raise ValueError(
                f"item {item_index} must contain exactly "
                f"{sorted(_ITEM_KEYS)}, got {sorted(item)}"
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

    assert shared_binding_index is not None
    return tuple(queries), shared_binding_index


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
        queries, binding_index = _validate_items(items)
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
