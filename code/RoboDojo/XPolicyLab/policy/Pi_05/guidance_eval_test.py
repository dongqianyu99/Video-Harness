from __future__ import annotations

from types import MappingProxyType

import numpy as np
from openpi.models.guide_inputs import GuideInput
import pytest
from video_harness.eval_guidance import EvalGuidance, EvalGuidePlan
from video_harness.reader import GuideFrameRef, GuidePlanUnit

from XPolicyLab.policy.Pi_05.guidance_eval import TaskGuideSession


class _Catalog:
    def __init__(self):
        document = MappingProxyType({"source": MappingProxyType({"episode_index": 3})})
        self.guidance = EvalGuidance("build", "doc-3", 4, "toast bread", 3, document)
        self.other = EvalGuidance("build", "doc-8", 5, "put cup away", 8, document)

    def resolve(self, *, task_instruction=None, task_index=None):
        del task_index
        return self.guidance if task_instruction == "toast bread" else self.other

    def build_plan(self, *, task_index=None, profile=None):
        del task_index
        return EvalGuidePlan(
            support_document_id="doc-3",
            support_episode_index=3,
            task_index=4,
            task_instruction="toast bread",
            profile=profile,
            frames=(
                GuideFrameRef("doc-3", 3, 0, 0.0),
                GuideFrameRef("doc-3", 3, 25, 1.0),
            ),
            units=(
                GuidePlanUnit("u0000", 0, 0, 1, "Visible change: bread is in toaster.", {}),
            ),
        )


class _FrameLoader:
    def __init__(self):
        self.calls = []

    def load_rgb(self, document, frame_ref):
        self.calls.append((document, frame_ref))
        return np.zeros((2, 3, 3), dtype=np.uint8)


class _Tokenizer:
    def tokenize_text(self, _text):
        return (
            np.asarray([1, 2, 0, 0], dtype=np.int32),
            np.asarray([True, True, False, False], dtype=np.bool_),
        )


def _session(tmp_path):
    documents = tmp_path / "documents.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    dataset = tmp_path / "dataset"
    documents.touch()
    episodes.touch()
    dataset.mkdir()
    loader = _FrameLoader()
    session = TaskGuideSession(
        documents_path=documents,
        episodes_path=episodes,
        dataset_root=dataset,
        max_frames=2,
        max_units=1,
        max_text_tokens=4,
        catalog=_Catalog(),
        frame_loader=loader,
        tokenizer=_Tokenizer(),
    )
    return session, loader


def test_session_materializes_once_and_reuses_task_guide(tmp_path):
    session, loader = _session(tmp_path)

    first = session.bind_instruction("toast bread")
    second = session.bind_instruction("toast bread")

    assert isinstance(first, GuideInput)
    assert first is second
    assert session.materialization_count == 1
    assert [call[1]["episode_frame_index"] for call in loader.calls] == [0, 25]


def test_session_rejects_task_switch_until_cleared(tmp_path):
    session, _ = _session(tmp_path)
    session.bind_instruction("toast bread")

    with pytest.raises(ValueError, match="cannot switch"):
        session.bind_instruction("put cup away")

    session.clear()
    assert session.guide is None
    assert session.guidance is None
