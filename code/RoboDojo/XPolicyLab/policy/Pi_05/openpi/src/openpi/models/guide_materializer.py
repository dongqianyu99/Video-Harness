from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from openpi.models.guide_inputs import GuideInput
from openpi.shared import image_tools


@dataclass(frozen=True)
class GuideMaterializerConfig:
    max_frames: int
    max_units: int
    max_text_tokens: int
    image_size: tuple[int, int] = (224, 224)


FrameDecoder = Callable[[Any], np.ndarray]


def materialize_guide(
    plan: Any,
    *,
    frame_decoder: FrameDecoder,
    tokenizer: Any,
    config: GuideMaterializerConfig,
) -> GuideInput:
    _validate_materializer_config(config)
    _validate_plan_sizes(plan, config)

    images, image_mask = _materialize_frames(
        plan,
        frame_decoder=frame_decoder,
        config=config,
    )

    text_tokens, text_mask = _materialize_text(
        plan,
        tokenizer=tokenizer,
        config=config,
    )

    unit_mask, before_slot, after_slot = _materialize_unit_slots(
        plan,
        config=config,
    )

    return GuideInput(
        images=images[None, ...],
        image_mask=image_mask[None, ...],
        text_tokens=text_tokens[None, ...],
        text_mask=text_mask[None, ...],
        unit_mask=unit_mask[None, ...],
        before_slot=before_slot[None, ...],
        after_slot=after_slot[None, ...],
    )


def _validate_materializer_config(config: GuideMaterializerConfig) -> None:
    for name in ("max_frames", "max_units", "max_text_tokens"):
        value = getattr(config, name)

        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if tuple(config.image_size) != (224, 224):
        raise ValueError(
            f"Guide images must use the Pi0.5 size (224, 224), got {config.image_size!r}"
        )


def _validate_plan_sizes(
    plan: Any,
    config: GuideMaterializerConfig,
) -> None:
    if not hasattr(plan, "frames") or not hasattr(plan, "units"):
        raise ValueError("plan must provide frames and units")

    frame_count = len(plan.frames)
    unit_count = len(plan.units)

    if frame_count > config.max_frames:
        raise ValueError(
            f"GuidePlan contains {frame_count} frames, "
            f"exceeding max_frames={config.max_frames}"
        )

    if unit_count > config.max_units:
        raise ValueError(
            f"GuidePlan contains {unit_count} units, "
            f"exceeding max_units={config.max_units}"
        )


def _decode_and_preprocess_frame(
    frame_ref: Any,
    *,
    frame_decoder: FrameDecoder,
    config: GuideMaterializerConfig,
) -> np.ndarray:
    frame = frame_decoder(frame_ref)

    if not isinstance(frame, np.ndarray):
        raise ValueError(
            f"decoded frame must be a numpy.ndarray, got {type(frame).__name__}"
        )

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"decoded frame must have RGB shape [H, W, 3], got {frame.shape}"
        )

    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError(f"decoded frame must have positive height and width, got {frame.shape}")

    if frame.dtype != np.uint8:
        raise ValueError(f"decoded frame must have dtype uint8, got {frame.dtype}")

    normalized_frame = frame.astype(np.float32) / 127.5 - 1.0

    resized_frame = image_tools.resize_with_pad(
        jnp.asarray(normalized_frame),
        config.image_size[0],
        config.image_size[1],
    )

    resized_frame = np.asarray(resized_frame, dtype=np.float32)
    expected_shape = (*config.image_size, 3)

    if resized_frame.shape != expected_shape:
        raise ValueError(
            f"resized frame must have shape {expected_shape}, got {resized_frame.shape}"
        )

    return resized_frame


def _materialize_frames(
    plan: Any,
    *,
    frame_decoder: FrameDecoder,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = config.image_size
    images = np.full(
        (config.max_frames, height, width, 3),
        -1.0,
        dtype=np.float32,
    )
    image_mask = np.zeros(config.max_frames, dtype=np.bool_)

    for frame_index, frame_ref in enumerate(plan.frames):
        images[frame_index] = _decode_and_preprocess_frame(
            frame_ref,
            frame_decoder=frame_decoder,
            config=config,
        )

    image_mask[: len(plan.frames)] = True
    return images, image_mask


def _materialize_text(
    plan: Any,
    *,
    tokenizer: Any,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tokenize_text = getattr(tokenizer, "tokenize_text", None)

    if not callable(tokenize_text):
        raise ValueError("tokenizer must provide a callable tokenize_text method")

    text_tokens = np.zeros(
        (config.max_units, config.max_text_tokens),
        dtype=np.int32,
    )
    text_mask = np.zeros(
        (config.max_units, config.max_text_tokens),
        dtype=np.bool_,
    )

    expected_shape = (config.max_text_tokens,)

    for unit_index, unit in enumerate(plan.units):
        if not isinstance(unit.transition_text, str):
            raise ValueError(
                f"unit {unit_index} transition_text must be str, "
                f"got {type(unit.transition_text).__name__}"
            )

        try:
            tokens, mask = tokenize_text(unit.transition_text)
        except ValueError as exc:
            raise ValueError(
                f"failed to tokenize Guide unit {unit_index}: {exc}"
            ) from exc

        if not isinstance(tokens, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError(
                f"tokenizer output for unit {unit_index} must be numpy arrays"
            )

        if tokens.shape != expected_shape:
            raise ValueError(
                f"tokenizer tokens for unit {unit_index} must have shape "
                f"{expected_shape}, got {tokens.shape}"
            )

        if mask.shape != expected_shape:
            raise ValueError(
                f"tokenizer mask for unit {unit_index} must have shape "
                f"{expected_shape}, got {mask.shape}"
            )

        if tokens.dtype != np.int32:
            raise ValueError(
                f"tokenizer tokens for unit {unit_index} must have dtype int32, "
                f"got {tokens.dtype}"
            )

        if mask.dtype != np.bool_:
            raise ValueError(
                f"tokenizer mask for unit {unit_index} must have dtype bool, "
                f"got {mask.dtype}"
            )

        text_tokens[unit_index] = tokens
        text_mask[unit_index] = mask

    return text_tokens, text_mask


def _validate_frame_slot(
    slot: Any,
    *,
    unit_index: int,
    slot_name: str,
    frame_count: int,
) -> int:
    if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)):
        raise ValueError(
            f"unit {unit_index} {slot_name} must be an integer, got {slot!r}"
        )

    slot_value = int(slot)

    if slot_value < 0 or slot_value >= frame_count:
        raise ValueError(
            f"unit {unit_index} {slot_name}={slot_value} is outside "
            f"the valid frame range [0, {frame_count})"
        )

    return slot_value


def _materialize_unit_slots(
    plan: Any,
    *,
    config: GuideMaterializerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(plan.frames)

    unit_mask = np.zeros(config.max_units, dtype=np.bool_)
    before_slot = np.zeros(config.max_units, dtype=np.int32)
    after_slot = np.zeros(config.max_units, dtype=np.int32)

    for unit_index, unit in enumerate(plan.units):
        before_value = _validate_frame_slot(
            unit.before_slot,
            unit_index=unit_index,
            slot_name="before_slot",
            frame_count=frame_count,
        )
        after_value = _validate_frame_slot(
            unit.after_slot,
            unit_index=unit_index,
            slot_name="after_slot",
            frame_count=frame_count,
        )

        before_slot[unit_index] = before_value
        after_slot[unit_index] = after_value
        unit_mask[unit_index] = True

    return unit_mask, before_slot, after_slot
