from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.robodojo_guide_resolver import RoboDojoGuideMaterializationConfig
from openpi.training.robodojo_guide_resolver import VideoHarnessGuideResolver


@dataclass(frozen=True)
class _Binding:
    query_episode_index: int
    support_episode_index: int
    task_index: int
    support_document_id: str


@dataclass(frozen=True)
class _Source:
    document_id: str
    episode_index: int
    task_index: int
    document: MappingProxyType


@dataclass(frozen=True)
class _Bundle:
    documents: tuple[_Source, ...]


@dataclass(frozen=True)
class _FrameRef:
    document_id: str
    episode_index: int
    episode_frame_index: int
    timestamp_s: float


@dataclass(frozen=True)
class _Unit:
    before_slot: int
    after_slot: int
    transition_text: str


@dataclass(frozen=True)
class _Plan:
    query_episode_index: int
    support_document_id: str
    support_episode_index: int
    task_index: int
    frames: tuple[_FrameRef, ...]
    units: tuple[_Unit, ...]


def _rgb(value: int) -> np.ndarray:
    return np.full((2, 4, 3), value, dtype=np.uint8)


def _make_setup() -> tuple[GuideBindingIndex, _Bundle, _Plan]:
    binding_index = GuideBindingIndex.from_bindings(
        [_Binding(1, 10, 3, "doc-support")]
    )
    document = MappingProxyType(
        {
            "document_id": "doc-support",
            "source": MappingProxyType(
                {
                    "episode_index": 10,
                    "task_index": 3,
                    "episode_length": 3,
                    "fps": 25,
                    "video_path": "videos/support.mp4",
                }
            ),
        }
    )
    source = _Source("doc-support", 10, 3, document)
    plan = _Plan(
        query_episode_index=1,
        support_document_id="doc-support",
        support_episode_index=10,
        task_index=3,
        frames=(
            _FrameRef("doc-support", 10, 0, 0.0),
            _FrameRef("doc-support", 10, 2, 0.08),
        ),
        units=(_Unit(0, 1, "Observed before: move. Observed after: contact."),),
    )
    return binding_index, _Bundle((source,)), plan


class _FrameLoader:
    def __init__(self):
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def load_rgb(self, document: Any, frame_ref: dict[str, Any]) -> np.ndarray:
        self.calls.append((document, frame_ref))
        return _rgb(80 + frame_ref["episode_frame_index"])


class _Tokenizer:
    def __init__(self):
        self.calls: list[str] = []

    def tokenize_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(text)
        return (
            np.asarray([1, 2, 0, 0], dtype=np.int32),
            np.asarray([True, True, False, False], dtype=np.bool_),
        )


def test_resolver_builds_guide_input_from_support_frames_only():
    binding_index, bundle, plan = _make_setup()
    frame_loader = _FrameLoader()
    tokenizer = _Tokenizer()
    plan_calls: list[tuple[int, str]] = []

    def plan_builder(bundle_arg, *, query_episode_index: int, profile: str):
        assert bundle_arg is bundle
        plan_calls.append((query_episode_index, profile))
        return plan

    resolver = VideoHarnessGuideResolver(
        artifact_bundle=bundle,
        binding_index=binding_index,
        dataset_root=Path("/explicit/dataset"),
        tokenizer=tokenizer,
        materializer_config=GuideMaterializerConfig(2, 1, 4),
        frame_loader=frame_loader,
        plan_builder=plan_builder,
    )

    guide = resolver(binding_index.by_binding_index(0))

    assert isinstance(guide, GuideInput)
    assert guide.images.shape == (1, 2, 224, 224, 3)
    assert guide.text_tokens.shape == (1, 1, 4)
    assert guide.unit_mask.tolist() == [[True]]
    assert plan_calls == [(1, "actuator-v0")]
    assert tokenizer.calls == [plan.units[0].transition_text]
    assert [call[1] for call in frame_loader.calls] == [
        {"episode_frame_index": 0, "timestamp_s": 0.0},
        {"episode_frame_index": 2, "timestamp_s": 0.08},
    ]
    assert all(call[0] is bundle.documents[0].document for call in frame_loader.calls)
    assert np.asarray(guide.images).shape[-1] == 3
    assert all(not isinstance(leaf, str) for leaf in jax.tree_util.tree_leaves(guide))


def test_resolver_rejects_non_rgb_frame_from_video_harness_boundary():
    binding_index, bundle, plan = _make_setup()

    class _GrayscaleLoader:
        def load_rgb(self, _document, _frame_ref):
            return np.full((2, 4), 100, dtype=np.uint8)

    resolver = VideoHarnessGuideResolver(
        artifact_bundle=bundle,
        binding_index=binding_index,
        dataset_root=Path("/explicit/dataset"),
        tokenizer=_Tokenizer(),
        materializer_config=GuideMaterializerConfig(2, 1, 4),
        frame_loader=_GrayscaleLoader(),
        plan_builder=lambda *_args, **_kwargs: plan,
    )

    with pytest.raises(ValueError, match=r"RGB"):
        resolver(binding_index.by_binding_index(0))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: _Plan(2, plan.support_document_id, 10, 3, plan.frames, plan.units),
        lambda plan: _Plan(1, "other-document", 10, 3, plan.frames, plan.units),
        lambda plan: _Plan(1, plan.support_document_id, 99, 3, plan.frames, plan.units),
        lambda plan: _Plan(1, plan.support_document_id, 10, 99, plan.frames, plan.units),
        lambda plan: _Plan(1, plan.support_document_id, 10, 3, plan.frames, ()),
    ],
)
def test_resolver_rejects_plan_identity_or_empty_units(mutate):
    binding_index, bundle, plan = _make_setup()
    invalid_plan = mutate(plan)
    resolver = VideoHarnessGuideResolver(
        artifact_bundle=bundle,
        binding_index=binding_index,
        dataset_root=Path("/explicit/dataset"),
        tokenizer=_Tokenizer(),
        materializer_config=GuideMaterializerConfig(2, 1, 4),
        frame_loader=_FrameLoader(),
        plan_builder=lambda *_args, **_kwargs: invalid_plan,
    )

    with pytest.raises(ValueError, match=r"binding_index|document|mismatch|trainable"):
        resolver(binding_index.by_binding_index(0))


def test_resolver_rejects_encoded_frame_payload_inside_xpolicylab():
    binding_index, bundle, plan = _make_setup()

    class _BadLoader:
        def load(self, _document, _frame_ref):
            return b"encoded-image-payload"

    resolver = VideoHarnessGuideResolver(
        artifact_bundle=bundle,
        binding_index=binding_index,
        dataset_root=Path("/explicit/dataset"),
        tokenizer=_Tokenizer(),
        materializer_config=GuideMaterializerConfig(2, 1, 4),
        frame_loader=_BadLoader(),
        plan_builder=lambda *_args, **_kwargs: plan,
    )

    with pytest.raises(ValueError, match=r"RGB|frame"):
        resolver(binding_index.by_binding_index(0))


def test_materialization_config_has_explicit_dataset_root_and_budgets():
    config = RoboDojoGuideMaterializationConfig(
        dataset_root=Path("/dataset"),
        profile="actuator-v0",
        max_frames=4,
        max_units=2,
        max_text_tokens=8,
    )

    materializer_config = config.to_materializer_config()

    assert config.dataset_root == Path("/dataset")
    assert materializer_config.max_frames == 4
    assert materializer_config.max_units == 2
    assert materializer_config.max_text_tokens == 8
