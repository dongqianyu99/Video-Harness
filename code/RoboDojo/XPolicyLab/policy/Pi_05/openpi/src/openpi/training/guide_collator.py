from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_inputs import validate_guide_conditioned_batch
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import GuideRecord

_REQUIRED_ITEM_KEYS = frozenset(("query", "guide_index"))
_OPTIONAL_ITEM_KEYS = frozenset(("query_valid",))


def _read_guide_index(value: Any, *, item_index: int) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"item {item_index} guide_index must be a scalar integer")
    guide_index = int(array.item())
    if guide_index < 0:
        raise ValueError(f"item {item_index} guide_index must be non-negative")
    return guide_index


def _read_validity(value: Any, *, item_index: int) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype != np.bool_:
        raise ValueError(f"item {item_index} query_valid must be a scalar bool")
    return bool(array.item())


def _validate_query_group(
    items: Sequence[Any],
    *,
    offset: int,
) -> tuple[tuple[Mapping[str, Any], ...], int, np.ndarray]:
    queries: list[Mapping[str, Any]] = []
    validity: list[bool] = []
    shared_guide: int | None = None
    for local_index, item in enumerate(items):
        item_index = offset + local_index
        if not isinstance(item, Mapping):
            raise ValueError(f"item {item_index} must be a mapping")
        keys = set(item)
        if not keys >= _REQUIRED_ITEM_KEYS or not keys <= (
            _REQUIRED_ITEM_KEYS | _OPTIONAL_ITEM_KEYS
        ):
            raise ValueError(
                f"item {item_index} must contain {sorted(_REQUIRED_ITEM_KEYS)} "
                f"and only optional {sorted(_OPTIONAL_ITEM_KEYS)}, got {sorted(keys)}"
            )
        query = item["query"]
        if not isinstance(query, Mapping):
            raise ValueError(f"item {item_index} query must be a mapping")
        guide_index = _read_guide_index(
            item["guide_index"], item_index=item_index
        )
        if shared_guide is None:
            shared_guide = guide_index
        elif guide_index != shared_guide:
            raise ValueError(
                f"Guide group mixes guide_index={shared_guide} and {guide_index}"
            )
        queries.append(query)
        validity.append(
            _read_validity(item.get("query_valid", True), item_index=item_index)
        )
    assert shared_guide is not None
    return tuple(queries), shared_guide, np.asarray(validity, dtype=np.bool_)


def _validate_grouped_items(
    items: Sequence[Any],
    *,
    guides_per_batch: int,
    queries_per_guide: int,
    guide_catalog: GuideCatalog,
) -> tuple[
    tuple[tuple[Mapping[str, Any], ...], ...],
    tuple[int, ...],
    np.ndarray,
]:
    expected = guides_per_batch * queries_per_guide
    if len(items) != expected:
        raise ValueError(f"grouped batch must contain G*Q={expected} items")
    query_groups = []
    guide_indices = []
    masks = []
    valid_documents: set[str] = set()
    for group_index in range(guides_per_batch):
        start = group_index * queries_per_guide
        group, guide_index, mask = _validate_query_group(
            items[start : start + queries_per_guide], offset=start
        )
        record = guide_catalog.by_guide_index(guide_index)
        if bool(np.any(mask)):
            if record.document_id in valid_documents:
                raise ValueError(
                    "each valid Guide group must use a distinct document, got "
                    f"{record.document_id!r}"
                )
            valid_documents.add(record.document_id)
        query_groups.append(group)
        guide_indices.append(guide_index)
        masks.append(mask)
    return tuple(query_groups), tuple(guide_indices), np.stack(masks, axis=0)


def _stack_tree(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty sequence")
    first = values[0]
    if isinstance(first, Mapping):
        keys = set(first)
        if any(not isinstance(value, Mapping) or set(value) != keys for value in values):
            raise ValueError("all nested query mappings must have the same keys")
        return {key: _stack_tree([value[key] for value in values]) for key in first}
    if first is None:
        if any(value is not None for value in values):
            raise ValueError("query leaves must consistently be None or arrays")
        return None
    if any(value is None or isinstance(value, Mapping) for value in values):
        raise ValueError("query leaves must have consistent array structure")
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except ValueError as exc:
        raise ValueError("query leaves have incompatible shapes") from exc


def _concatenate_guides(guides: Sequence[GuideInput]) -> GuideInput:
    if not guides:
        raise ValueError("cannot concatenate an empty Guide sequence")
    for index, guide in enumerate(guides):
        if not isinstance(guide, GuideInput):
            raise ValueError(
                f"resolved guide {index} must be GuideInput, got {type(guide).__name__}"
            )
        if any(
            not hasattr(leaf, "shape") or leaf.shape[0] != 1
            for leaf in jax.tree_util.tree_leaves(guide)
        ):
            raise ValueError(
                f"resolved guide {index} must have one leading Guide group"
            )
    return jax.tree_util.tree_map(
        lambda *values: np.concatenate(
            [np.asarray(value) for value in values], axis=0
        ),
        *guides,
    )


class GuidanceBatchCollator:
    """Collate one fixed group-major ``G Guidance x Q queries`` batch."""

    def __init__(
        self,
        *,
        guide_catalog: GuideCatalog,
        guide_input_resolver: Callable[[GuideRecord], GuideInput],
        guides_per_batch: int,
        queries_per_guide: int,
    ) -> None:
        if not callable(guide_input_resolver):
            raise ValueError("guide_input_resolver must be callable")
        for name, value in (
            ("guides_per_batch", guides_per_batch),
            ("queries_per_guide", queries_per_guide),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._guide_catalog = guide_catalog
        self._resolver = guide_input_resolver
        self._guides_per_batch = guides_per_batch
        self._queries_per_guide = queries_per_guide

    def _resolve(self, guide_index: int) -> GuideInput:
        record = self._guide_catalog.by_guide_index(guide_index)
        try:
            return self._resolver(record)
        except Exception as exc:
            raise RuntimeError(
                f"failed to resolve Guide guide_index={guide_index} "
                f"document_id={record.document_id!r}"
            ) from exc

    def __call__(self, items: Sequence[Mapping[str, Any]]) -> GuideConditionedBatch:
        query_groups, guide_indices, query_mask = _validate_grouped_items(
            items,
            guides_per_batch=self._guides_per_batch,
            queries_per_guide=self._queries_per_guide,
            guide_catalog=self._guide_catalog,
        )
        collated_query = dict(
            _stack_tree([_stack_tree(group) for group in query_groups])
        )
        if "actions" not in collated_query:
            raise ValueError("collated query must contain actions")
        actions = collated_query.pop("actions")
        batch = GuideConditionedBatch(
            observation=_model.Observation.from_dict(collated_query),
            actions=actions,
            guide=_concatenate_guides(
                [self._resolve(guide_index) for guide_index in guide_indices]
            ),
            query_mask=query_mask,
        )
        shape = validate_guide_conditioned_batch(batch)
        expected = (self._guides_per_batch, self._queries_per_guide)
        if shape != expected:
            raise ValueError(
                f"collated grouped shape mismatch: expected {expected}, got {shape}"
            )
        return batch
