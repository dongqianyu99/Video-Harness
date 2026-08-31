from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_tokens import BOUNDARY_MEMORY_KIND
from openpi.models.guide_tokens import TRANSITION_MEMORY_KIND
from openpi.shared import image_tools


@dataclass(frozen=True)
class GuideMaterializerConfig:
    max_boundaries: int
    max_units: int
    max_boundary_text_tokens: int
    max_transition_text_tokens: int
    boundary_num_queries: int = 8
    transition_num_queries: int = 4
    image_size: tuple[int, int] = (224, 224)


BoundaryDecoder = Callable[[Any], np.ndarray]
BoundariesDecoder = Callable[[Sequence[Any]], Sequence[np.ndarray]]


def materialize_guide(
    plan: Any,
    *,
    boundary_decoder: BoundaryDecoder,
    boundaries_decoder: BoundariesDecoder | None = None,
    boundary_tokenizer: Any,
    transition_tokenizer: Any,
    config: GuideMaterializerConfig,
) -> GuideInput:
    _validate_materializer_config(config)
    _validate_plan_sizes(plan, config)
    boundary_images, boundary_image_mask, boundary_mask = _materialize_boundaries(
        plan,
        boundary_decoder=boundary_decoder,
        boundaries_decoder=boundaries_decoder,
        config=config,
    )
    boundary_text_tokens, boundary_text_mask = _materialize_boundary_text(
        plan,
        tokenizer=boundary_tokenizer,
        config=config,
    )
    transition_text_tokens, transition_text_mask = _materialize_transition_text(
        plan,
        tokenizer=transition_tokenizer,
        config=config,
    )
    unit_mask = _validate_unit_slots(plan, config=config)
    source_kind, source_index, source_offset, memory_mask = _materialize_memory_map(
        plan,
        config=config,
    )
    return GuideInput(
        boundary_images=boundary_images[None, ...],
        boundary_image_mask=boundary_image_mask[None, ...],
        boundary_text_tokens=boundary_text_tokens[None, ...],
        boundary_text_mask=boundary_text_mask[None, ...],
        transition_text_tokens=transition_text_tokens[None, ...],
        transition_text_mask=transition_text_mask[None, ...],
        boundary_mask=boundary_mask[None, ...],
        unit_mask=unit_mask[None, ...],
        memory_source_kind=source_kind[None, ...],
        memory_source_index=source_index[None, ...],
        memory_source_offset=source_offset[None, ...],
        memory_mask=memory_mask[None, ...],
    )


def _validate_materializer_config(config: GuideMaterializerConfig) -> None:
    for name in (
        "max_boundaries",
        "max_units",
        "max_boundary_text_tokens",
        "max_transition_text_tokens",
        "boundary_num_queries",
        "transition_num_queries",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if tuple(config.image_size) != (224, 224):
        raise ValueError(
            f"Guide images must use the Pi0.5 size (224, 224), got {config.image_size!r}"
        )


def _validate_plan_sizes(plan: Any, config: GuideMaterializerConfig) -> None:
    if not hasattr(plan, "boundaries") or not hasattr(plan, "units"):
        raise ValueError("plan must provide boundaries and units")
    if not plan.boundaries or not plan.units:
        raise ValueError("plan must provide at least one Boundary and accepted Unit")
    if len(plan.boundaries) > config.max_boundaries:
        raise ValueError(
            f"GuidePlan contains {len(plan.boundaries)} Boundaries, "
            f"exceeding max_boundaries={config.max_boundaries}"
        )
    if len(plan.units) > config.max_units:
        raise ValueError(
            f"GuidePlan contains {len(plan.units)} units, exceeding max_units={config.max_units}"
        )
    previous_order = -1
    previous_frame = -1
    previous_timestamp = -1.0
    for slot, boundary in enumerate(plan.boundaries):
        if getattr(boundary, "slot", slot) != slot:
            raise ValueError("GuidePlan Boundary slots must be dense and ordered")
        order = getattr(boundary, "order", None)
        frame = getattr(boundary, "episode_frame_index", None)
        timestamp = getattr(boundary, "timestamp_s", None)
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or order <= previous_order
        ):
            raise ValueError("GuidePlan Boundary order must advance strictly")
        if (
            isinstance(frame, bool)
            or not isinstance(frame, (int, np.integer))
            or int(frame) <= previous_frame
        ):
            raise ValueError("GuidePlan Boundary frames must advance strictly")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or float(timestamp) <= previous_timestamp
        ):
            raise ValueError("GuidePlan Boundary timestamps must advance strictly")
        previous_order = order
        previous_frame = int(frame)
        previous_timestamp = float(timestamp)


def _validate_decoded_boundary(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"decoded Boundary must be a numpy.ndarray, got {type(value).__name__}"
        )
    if value.ndim != 4 or value.shape[0] != 3 or value.shape[-1] != 3:
        raise ValueError(
            "decoded Boundary must have three-view RGB shape [3, H, W, 3], "
            f"got {value.shape}"
        )
    if value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"decoded Boundary has an empty spatial dimension: {value.shape}")
    if value.dtype != np.uint8:
        raise ValueError(f"decoded Boundary must have dtype uint8, got {value.dtype}")
    return value


def _preprocess_boundary_batch(
    boundaries: Sequence[np.ndarray],
    *,
    config: GuideMaterializerConfig,
) -> np.ndarray:
    validated = tuple(_validate_decoded_boundary(value) for value in boundaries)
    shapes = {value.shape for value in validated}
    if len(shapes) != 1:
        raise ValueError(
            f"all decoded Boundaries must share one three-view RGB shape, got {sorted(shapes)}"
        )
    flat = np.stack(validated, axis=0).reshape(
        len(validated) * 3,
        validated[0].shape[1],
        validated[0].shape[2],
        3,
    )
    normalized = flat.astype(np.float32) / 127.5 - 1.0
    resized = image_tools.resize_with_pad_torch(
        torch.from_numpy(normalized),
        config.image_size[0],
        config.image_size[1],
    )
    result = np.asarray(resized.cpu().numpy(), dtype=np.float32)
    return result.reshape(len(validated), 3, *config.image_size, 3)


def _materialize_boundaries(
    plan: Any,
    *,
    boundary_decoder: BoundaryDecoder,
    boundaries_decoder: BoundariesDecoder | None,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = config.image_size
    images = np.full(
        (config.max_boundaries, 3, height, width, 3),
        -1.0,
        dtype=np.float32,
    )
    image_mask = np.zeros((config.max_boundaries, 3), dtype=np.bool_)
    boundary_mask = np.zeros(config.max_boundaries, dtype=np.bool_)
    if boundaries_decoder is None:
        decoded = tuple(boundary_decoder(boundary) for boundary in plan.boundaries)
    else:
        decoded = tuple(boundaries_decoder(plan.boundaries))
        if len(decoded) != len(plan.boundaries):
            raise ValueError(
                "boundaries_decoder must return exactly one item per GuidePlan Boundary"
            )
    preprocessed = _preprocess_boundary_batch(decoded, config=config)
    count = len(plan.boundaries)
    images[:count] = preprocessed
    image_mask[:count] = True
    boundary_mask[:count] = True
    return images, image_mask, boundary_mask


def _tokenize_text(
    tokenizer: Any,
    text: str,
    *,
    expected_tokens: int,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    tokenize_text = getattr(tokenizer, "tokenize_text", None)
    if not callable(tokenize_text):
        raise ValueError("tokenizer must provide a callable tokenize_text method")
    try:
        tokens, mask = tokenize_text(text)
    except ValueError as exc:
        raise ValueError(f"failed to tokenize {context}: {exc}") from exc
    expected_shape = (expected_tokens,)
    if not isinstance(tokens, np.ndarray) or tokens.shape != expected_shape:
        raise ValueError(
            f"tokenizer tokens for {context} must have shape {expected_shape}"
        )
    if not isinstance(mask, np.ndarray) or mask.shape != expected_shape:
        raise ValueError(f"tokenizer mask for {context} must have shape {expected_shape}")
    if tokens.dtype != np.int32 or mask.dtype != np.bool_:
        raise ValueError(f"tokenizer output for {context} must be int32 tokens and bool mask")
    if not np.any(mask):
        raise ValueError(f"tokenizer produced empty text for {context}")
    return tokens, mask


def _materialize_boundary_text(
    plan: Any,
    *,
    tokenizer: Any,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros(
        (config.max_boundaries, 3, config.max_boundary_text_tokens),
        dtype=np.int32,
    )
    mask = np.zeros(tokens.shape, dtype=np.bool_)
    for boundary_index, boundary in enumerate(plan.boundaries):
        view_texts = getattr(boundary, "view_texts", None)
        if not isinstance(view_texts, tuple) or len(view_texts) != 3:
            raise ValueError(
                f"Boundary {boundary_index} view_texts must be a three-item tuple"
            )
        for view, text in enumerate(view_texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Boundary {boundary_index} view {view} text must be non-empty"
                )
            tokens[boundary_index, view], mask[boundary_index, view] = _tokenize_text(
                tokenizer,
                text,
                expected_tokens=config.max_boundary_text_tokens,
                context=f"Boundary {boundary_index} view {view}",
            )
    return tokens, mask


def _materialize_transition_text(
    plan: Any,
    *,
    tokenizer: Any,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros(
        (config.max_units, config.max_transition_text_tokens),
        dtype=np.int32,
    )
    mask = np.zeros(tokens.shape, dtype=np.bool_)
    for unit_index, unit in enumerate(plan.units):
        text = getattr(unit, "transition_text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"unit {unit_index} transition_text must be non-empty")
        tokens[unit_index], mask[unit_index] = _tokenize_text(
            tokenizer,
            text,
            expected_tokens=config.max_transition_text_tokens,
            context=f"transition {unit_index}",
        )
    return tokens, mask


def _validate_slot(
    value: Any,
    *,
    unit_index: int,
    name: str,
    boundary_count: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"unit {unit_index} {name} must be an integer")
    result = int(value)
    if result < 0 or result >= boundary_count:
        raise ValueError(
            f"unit {unit_index} {name}={result} is outside [0, {boundary_count})"
        )
    return result


def _validate_unit_slots(
    plan: Any,
    *,
    config: GuideMaterializerConfig,
) -> np.ndarray:
    unit_mask = np.zeros(config.max_units, dtype=np.bool_)
    previous_order = -1
    for unit_index, unit in enumerate(plan.units):
        order = getattr(unit, "order", None)
        if isinstance(order, bool) or not isinstance(order, int) or order <= previous_order:
            raise ValueError("GuidePlan units must preserve strictly increasing canonical order")
        previous_order = order
        before = _validate_slot(
            unit.before_slot,
            unit_index=unit_index,
            name="before_slot",
            boundary_count=len(plan.boundaries),
        )
        after = _validate_slot(
            unit.after_slot,
            unit_index=unit_index,
            name="after_slot",
            boundary_count=len(plan.boundaries),
        )
        if before >= after:
            raise ValueError("Each GuidePlan Unit must advance to a later Boundary")
        if plan.boundaries[after].order != plan.boundaries[before].order + 1:
            raise ValueError("Each GuidePlan Unit must reference adjacent canonical Boundaries")
        unit_mask[unit_index] = True
    return unit_mask


def _materialize_memory_map(
    plan: Any,
    *,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence = (
        config.max_boundaries * config.boundary_num_queries
        + config.max_units * config.transition_num_queries
    )
    source_kind = np.zeros(sequence, dtype=np.int32)
    source_index = np.zeros(sequence, dtype=np.int32)
    source_offset = np.zeros(sequence, dtype=np.int32)
    memory_mask = np.zeros(sequence, dtype=np.bool_)
    emitted_boundaries: set[int] = set()
    cursor = 0
    previous_after: int | None = None

    def emit(kind: int, index: int, count: int) -> None:
        nonlocal cursor
        end = cursor + count
        source_kind[cursor:end] = kind
        source_index[cursor:end] = index
        source_offset[cursor:end] = np.arange(count, dtype=np.int32)
        memory_mask[cursor:end] = True
        cursor = end

    for unit_index, unit in enumerate(plan.units):
        before = int(unit.before_slot)
        after = int(unit.after_slot)
        if before != previous_after:
            if previous_after is not None and before <= previous_after:
                raise ValueError(
                    "GuidePlan discontinuities must advance to a later Boundary"
                )
            if before in emitted_boundaries:
                raise ValueError("GuidePlan chain revisits a non-adjacent Boundary")
            emit(BOUNDARY_MEMORY_KIND, before, config.boundary_num_queries)
            emitted_boundaries.add(before)
        elif before not in emitted_boundaries:
            raise ValueError("GuidePlan shared Boundary was not emitted")

        emit(TRANSITION_MEMORY_KIND, unit_index, config.transition_num_queries)
        if after in emitted_boundaries:
            raise ValueError("GuidePlan chain revisits an emitted after Boundary")
        emit(BOUNDARY_MEMORY_KIND, after, config.boundary_num_queries)
        emitted_boundaries.add(after)
        previous_after = after

    if emitted_boundaries != set(range(len(plan.boundaries))):
        raise ValueError("GuidePlan contains a Boundary not referenced by accepted Units")
    return source_kind, source_index, source_offset, memory_mask
