from dataclasses import dataclass

import numpy as np
import pytest

from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.guide_materializer import materialize_guide


@dataclass(frozen=True)
class _Boundary:
    boundary_id: str
    order: int
    slot: int
    episode_frame_index: int
    timestamp_s: float
    view_texts: tuple[str, str, str]


@dataclass(frozen=True)
class _Unit:
    unit_id: str
    order: int
    before_slot: int
    after_slot: int
    transition_text: str


@dataclass(frozen=True)
class _Plan:
    boundaries: tuple[_Boundary, ...]
    units: tuple[_Unit, ...]


class _Tokenizer:
    def __init__(self, length: int):
        self.length = length
        self.calls: list[str] = []

    def tokenize_text(self, text: str):
        self.calls.append(text)
        tokens = np.zeros(self.length, dtype=np.int32)
        mask = np.zeros(self.length, dtype=np.bool_)
        tokens[:2] = [1, 2]
        mask[:2] = True
        return tokens, mask


def _boundary(order: int) -> _Boundary:
    return _Boundary(
        boundary_id=f"b{order:04d}",
        order=order,
        slot=order,
        episode_frame_index=order * 10,
        timestamp_s=order * 0.4,
        view_texts=tuple(f"boundary {order} view {view}" for view in range(3)),
    )


def _plan(*, gap: bool = False) -> _Plan:
    if gap:
        return _Plan(
            boundaries=tuple(_boundary(index) for index in range(4)),
            units=(
                _Unit("u0000", 0, 0, 1, "transition zero"),
                _Unit("u0002", 2, 2, 3, "transition two"),
            ),
        )
    return _Plan(
        boundaries=tuple(_boundary(index) for index in range(3)),
        units=(
            _Unit("u0000", 0, 0, 1, "transition zero"),
            _Unit("u0001", 1, 1, 2, "transition one"),
        ),
    )


def _config(**overrides) -> GuideMaterializerConfig:
    values = {
        "max_boundaries": 4,
        "max_units": 3,
        "max_boundary_text_tokens": 4,
        "max_transition_text_tokens": 5,
        "boundary_num_queries": 2,
        "transition_num_queries": 1,
    }
    values.update(overrides)
    return GuideMaterializerConfig(**values)


def _decoded(boundary: _Boundary) -> np.ndarray:
    return np.full((3, 4, 6, 3), 40 + boundary.order, dtype=np.uint8)


def test_materialize_three_view_boundary_and_separate_text_banks() -> None:
    plan = _plan()
    boundary_tokenizer = _Tokenizer(4)
    transition_tokenizer = _Tokenizer(5)

    guide = materialize_guide(
        plan,
        boundary_decoder=_decoded,
        boundary_tokenizer=boundary_tokenizer,
        transition_tokenizer=transition_tokenizer,
        config=_config(),
    )

    assert guide.boundary_images.shape == (1, 4, 3, 224, 224, 3)
    assert guide.boundary_text_tokens.shape == (1, 4, 3, 4)
    assert guide.transition_text_tokens.shape == (1, 3, 5)
    assert guide.boundary_mask.tolist() == [[True, True, True, False]]
    assert guide.unit_mask.tolist() == [[True, True, False]]
    assert not hasattr(guide, "before_slot")
    assert not hasattr(guide, "after_slot")
    assert boundary_tokenizer.calls == [
        text for boundary in plan.boundaries for text in boundary.view_texts
    ]
    assert transition_tokenizer.calls == [unit.transition_text for unit in plan.units]


def test_materializer_prefers_one_batch_decode() -> None:
    plan = _plan()
    calls = []

    def decode_many(boundaries):
        calls.append(tuple(boundaries))
        return tuple(_decoded(boundary) for boundary in boundaries)

    materialize_guide(
        plan,
        boundary_decoder=lambda _boundary: pytest.fail("single decoder used"),
        boundaries_decoder=decode_many,
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        config=_config(),
    )

    assert calls == [plan.boundaries]


def test_memory_map_interleaves_contiguous_chain_with_tail_padding() -> None:
    guide = materialize_guide(
        _plan(),
        boundary_decoder=_decoded,
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        config=_config(),
    )
    valid = np.asarray(guide.memory_mask[0])

    np.testing.assert_array_equal(
        guide.memory_source_kind[0, valid],
        [0, 0, 1, 0, 0, 1, 0, 0],
    )
    np.testing.assert_array_equal(
        guide.memory_source_index[0, valid],
        [0, 0, 0, 1, 1, 1, 2, 2],
    )
    np.testing.assert_array_equal(
        guide.memory_source_offset[0, valid],
        [0, 1, 0, 0, 1, 0, 0, 1],
    )
    assert not np.any(valid[8:])


def test_memory_map_exposes_rejected_transition_as_boundary_boundary() -> None:
    guide = materialize_guide(
        _plan(gap=True),
        boundary_decoder=_decoded,
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        config=_config(),
    )
    valid = np.asarray(guide.memory_mask[0])
    kinds = np.asarray(guide.memory_source_kind[0, valid])
    indices = np.asarray(guide.memory_source_index[0, valid])

    np.testing.assert_array_equal(kinds, [0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 2, 2, 1, 3, 3])


def test_memory_map_rejects_backward_discontinuity() -> None:
    plan = _Plan(
        boundaries=tuple(_boundary(index) for index in range(4)),
        units=(
            _Unit("u0000", 0, 2, 3, "late transition"),
            _Unit("u0001", 1, 0, 1, "backward transition"),
        ),
    )
    with pytest.raises(ValueError, match="advance to a later Boundary"):
        materialize_guide(
            plan,
            boundary_decoder=_decoded,
            boundary_tokenizer=_Tokenizer(4),
            transition_tokenizer=_Tokenizer(5),
            config=_config(),
        )


@pytest.mark.parametrize(
    "config",
    [
        _config(max_boundaries=2),
        _config(max_units=1),
        _config(max_boundary_text_tokens=0),
        _config(max_transition_text_tokens=0),
        _config(boundary_num_queries=0),
        _config(transition_num_queries=0),
    ],
)
def test_materializer_rejects_invalid_or_insufficient_budgets(config) -> None:
    with pytest.raises(ValueError, match=r"positive|exceeding"):
        materialize_guide(
            _plan(),
            boundary_decoder=_decoded,
            boundary_tokenizer=_Tokenizer(max(config.max_boundary_text_tokens, 1)),
            transition_tokenizer=_Tokenizer(max(config.max_transition_text_tokens, 1)),
            config=config,
        )


def test_materializer_rejects_non_three_view_decode() -> None:
    with pytest.raises(ValueError, match="three-view"):
        materialize_guide(
            _plan(),
            boundary_decoder=lambda _boundary: np.zeros((2, 4, 6, 3), dtype=np.uint8),
            boundary_tokenizer=_Tokenizer(4),
            transition_tokenizer=_Tokenizer(5),
            config=_config(),
        )
