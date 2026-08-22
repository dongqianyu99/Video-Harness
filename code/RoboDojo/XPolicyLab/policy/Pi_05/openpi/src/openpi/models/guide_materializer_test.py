from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import materialize_guide


@dataclass(frozen=True)
class _FrameRef:
    document_id: str
    episode_index: int
    episode_frame_index: int
    timestamp_s: float


@dataclass(frozen=True)
class _PlanUnit:
    unit_id: str
    order: int
    before_slot: int
    after_slot: int
    transition_text: str
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _GuidePlan:
    query_episode_index: int
    support_document_id: str
    support_episode_index: int
    task_index: int
    task_instruction: str
    profile: str
    frames: tuple[_FrameRef, ...]
    units: tuple[_PlanUnit, ...]


def _make_plan() -> _GuidePlan:
    frames = tuple(
        _FrameRef("doc-7", 3, frame_index, float(frame_index))
        for frame_index in range(4)
    )
    units = (
        _PlanUnit(
            "unit-0",
            0,
            0,
            1,
            "Observed before: approach the handle. Observed after: contact.",
            {"source": "evidence-0"},
        ),
        _PlanUnit(
            "unit-2",
            2,
            2,
            3,
            "Observed before: contact. Observed after: move through the opening.",
            {"source": "evidence-2"},
        ),
    )
    return _GuidePlan(
        query_episode_index=11,
        support_document_id="doc-7",
        support_episode_index=3,
        task_index=4,
        task_instruction="open the cabinet",
        profile="actuator",
        frames=frames,
        units=units,
    )


def _make_config(
    *,
    max_frames: int = 4,
    max_units: int = 2,
    max_text_tokens: int = 6,
):
    return SimpleNamespace(
        max_frames=max_frames,
        max_units=max_units,
        max_text_tokens=max_text_tokens,
        image_size=(224, 224),
    )


class _RecordingDecoder:
    def __init__(self):
        self.calls: list[_FrameRef] = []

    def __call__(self, frame_ref: _FrameRef) -> np.ndarray:
        self.calls.append(frame_ref)
        return np.full((2, 4, 3), frame_ref.episode_frame_index * 40, dtype=np.uint8)


class _SingleFrameDecoder:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def __call__(self, _frame_ref: _FrameRef) -> np.ndarray:
        return self.frame


class _RecordingTokenizer:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.calls: list[str] = []

    def tokenize_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(text)
        tokens = np.zeros(self.max_len, dtype=np.int32)
        tokens[:3] = np.asarray([11, 12, 13], dtype=np.int32)
        mask = np.zeros(self.max_len, dtype=np.bool_)
        mask[:3] = True
        return tokens, mask


class _OverflowTokenizer:
    def tokenize_text(self, _text: str) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError("Guide text exceeds max_text_tokens")


def test_materialize_guide_shape_order_slots_and_resize():
    plan = _make_plan()
    decoder = _RecordingDecoder()
    tokenizer = _RecordingTokenizer(max_len=6)

    guide = materialize_guide(
        plan,
        frame_decoder=decoder,
        tokenizer=tokenizer,
        config=_make_config(),
    )

    assert isinstance(guide, GuideInput)
    assert guide.images.shape == (1, 4, 224, 224, 3)
    assert guide.image_mask.shape == (1, 4)
    assert guide.text_tokens.shape == (1, 2, 6)
    assert guide.text_mask.shape == (1, 2, 6)
    assert guide.unit_mask.shape == (1, 2)
    assert guide.before_slot.shape == (1, 2)
    assert guide.after_slot.shape == (1, 2)
    assert guide.images.dtype == np.float32
    assert guide.text_tokens.dtype == np.int32
    assert guide.image_mask.dtype == np.bool_
    assert guide.text_mask.dtype == np.bool_
    assert guide.unit_mask.dtype == np.bool_
    assert guide.before_slot.dtype == np.int32
    assert guide.after_slot.dtype == np.int32
    assert decoder.calls == list(plan.frames)
    assert tokenizer.calls == [unit.transition_text for unit in plan.units]
    np.testing.assert_array_equal(guide.before_slot, [[0, 2]])
    np.testing.assert_array_equal(guide.after_slot, [[1, 3]])
    assert np.asarray(guide.images).min() >= -1.0
    assert np.asarray(guide.images).max() <= 1.0


def test_materialize_guide_padding_is_masked_and_zero_or_minus_one():
    plan = _make_plan()
    guide = materialize_guide(
        plan,
        frame_decoder=_RecordingDecoder(),
        tokenizer=_RecordingTokenizer(max_len=6),
        config=_make_config(max_frames=5, max_units=3),
    )

    np.testing.assert_array_equal(guide.image_mask, [[True, True, True, True, False]])
    np.testing.assert_array_equal(guide.unit_mask, [[True, True, False]])
    assert np.all(np.asarray(guide.images)[0, 4] == -1.0)
    assert np.all(np.asarray(guide.text_tokens)[0, 2] == 0)
    assert not np.any(np.asarray(guide.text_mask)[0, 2])
    np.testing.assert_array_equal(guide.before_slot, [[0, 2, 0]])
    np.testing.assert_array_equal(guide.after_slot, [[1, 3, 0]])


def test_materialize_guide_does_not_mutate_plan():
    plan = _make_plan()
    before = asdict(plan)

    materialize_guide(
        plan,
        frame_decoder=_RecordingDecoder(),
        tokenizer=_RecordingTokenizer(max_len=6),
        config=_make_config(max_frames=5, max_units=3),
    )

    assert asdict(plan) == before


def test_materialize_guide_is_a_guide_input_pytree():
    guide = materialize_guide(
        _make_plan(),
        frame_decoder=_RecordingDecoder(),
        tokenizer=_RecordingTokenizer(max_len=6),
        config=_make_config(),
    )

    leaves, _ = jax.tree_util.tree_flatten(guide)

    assert len(leaves) == 7


@pytest.mark.parametrize(
    ("config_kwargs", "error_pattern"),
    [
        ({"max_frames": 3}, "frame|max_frames|overflow"),
        ({"max_units": 1}, "unit|max_units|overflow"),
    ],
)
def test_materialize_guide_rejects_frame_and_unit_overflow(config_kwargs, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        materialize_guide(
            _make_plan(),
            frame_decoder=_RecordingDecoder(),
            tokenizer=_RecordingTokenizer(max_len=6),
            config=_make_config(**config_kwargs),
        )


def test_materialize_guide_rejects_text_overflow():
    with pytest.raises(ValueError, match=r"text|max_text_tokens|overflow"):
        materialize_guide(
            _make_plan(),
            frame_decoder=_RecordingDecoder(),
            tokenizer=_OverflowTokenizer(),
            config=_make_config(),
        )


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 4), dtype=np.uint8),
        np.zeros((2, 4, 4), dtype=np.uint8),
        np.zeros((2, 4, 3), dtype=np.float32),
    ],
)
def test_materialize_guide_rejects_non_rgb_uint8_frames(frame):
    with pytest.raises(ValueError, match=r"frame|RGB|uint8|channel"):
        materialize_guide(
            _make_plan(),
            frame_decoder=_SingleFrameDecoder(frame),
            tokenizer=_RecordingTokenizer(max_len=6),
            config=_make_config(),
        )


@pytest.mark.parametrize(
    ("before_slot", "after_slot"),
    [(-1, 1), (0, 4)],
)
def test_materialize_guide_rejects_invalid_slots(before_slot, after_slot):
    plan = _make_plan()
    invalid_unit = replace(
        plan.units[0],
        before_slot=before_slot,
        after_slot=after_slot,
    )
    invalid_plan = replace(plan, units=(invalid_unit, *plan.units[1:]))

    with pytest.raises(ValueError, match=r"slot|frame"):
        materialize_guide(
            invalid_plan,
            frame_decoder=_RecordingDecoder(),
            tokenizer=_RecordingTokenizer(max_len=6),
            config=_make_config(),
        )


def test_materializer_uses_batch_frame_decoder_once():
    plan = _make_plan()
    calls = []

    def frames_decoder(frame_refs):
        calls.append(tuple(frame_refs))
        return tuple(
            np.full((2, 4, 3), 10 + index, dtype=np.uint8)
            for index, _ in enumerate(frame_refs)
        )

    guide = materialize_guide(
        plan,
        frame_decoder=lambda _frame: pytest.fail("single-frame decoder should not run"),
        frames_decoder=frames_decoder,
        tokenizer=_RecordingTokenizer(max_len=6),
        config=_make_config(),
    )

    assert calls == [plan.frames]
    assert guide.images.shape[0] == 1


def test_materializer_rejects_batch_decoder_length_mismatch():
    plan = _make_plan()
    with pytest.raises(ValueError, match="exactly one frame"):
        materialize_guide(
            plan,
            frame_decoder=lambda _frame: pytest.fail("single decoder should not run"),
            frames_decoder=lambda _frames: (),
            tokenizer=_RecordingTokenizer(max_len=6),
            config=_make_config(),
        )
