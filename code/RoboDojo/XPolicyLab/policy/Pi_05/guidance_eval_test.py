from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from openpi.models.guide_inputs import GuideInput
import pytest

from XPolicyLab.policy.Pi_05.guidance_eval import TaskGuideSession


class _Catalog:
    def __init__(self):
        self.catalog_digest = "catalog-digest"
        document = {"source": {"episode_index": 3}}
        self.guidance = SimpleNamespace(
            document_id="doc-3",
            task_index=4,
            task_instruction="toast bread",
            source_episode_index=3,
            document=document,
        )
        self.other = SimpleNamespace(
            document_id="doc-8",
            task_index=5,
            task_instruction="put cup away",
            source_episode_index=8,
            document=document,
        )
        self.documents = (self.guidance, self.other)


class _FrameLoader:
    def __init__(self):
        self.calls = []

    def load_views_rgb_many(self, document, frame_refs):
        self.calls.append((document, tuple(frame_refs)))
        view = np.zeros((2, 3, 3), dtype=np.uint8)
        return tuple((view.copy(), view.copy(), view.copy()) for _ in frame_refs)


class _Tokenizer:
    def tokenize_text(self, _text):
        return (
            np.asarray([1, 2, 0, 0], dtype=np.int32),
            np.asarray([True, True, False, False], dtype=np.bool_),
        )


def _plan_builder(_catalog, *, document_id):
    assert document_id == "doc-3"
    boundaries = tuple(
        SimpleNamespace(
            boundary_id=f"b{index:04d}",
            order=index,
            slot=index,
            episode_frame_index=frame,
            timestamp_s=float(index),
            view_texts=("high", "left", "right"),
        )
        for index, frame in enumerate((0, 25))
    )
    return SimpleNamespace(
        document_id="doc-3",
        source_episode_index=3,
        task_index=4,
        task_instruction="toast bread",
        boundaries=boundaries,
        units=(
            SimpleNamespace(
                unit_id="u0000",
                order=0,
                before_slot=0,
                after_slot=1,
                transition_text="Motion: insert bread\nDetail: contact\nAction: insert\nTask role: complete",
            ),
        ),
    )


def _session(tmp_path, *, catalog=None):
    documents = tmp_path / "documents"
    dataset = tmp_path / "dataset"
    documents.mkdir()
    dataset.mkdir()
    loader = _FrameLoader()
    session = TaskGuideSession(
        documents_root=documents,
        dataset_root=dataset,
        max_boundaries=2,
        max_units=1,
        max_boundary_text_tokens=4,
        max_transition_text_tokens=4,
        boundary_num_queries=2,
        transition_num_queries=1,
        catalog=_Catalog() if catalog is None else catalog,
        frame_loader=loader,
        boundary_tokenizer=_Tokenizer(),
        transition_tokenizer=_Tokenizer(),
        plan_builder=_plan_builder,
    )
    return session, loader


def test_session_materializes_once_and_reuses_first_task_guide(tmp_path):
    session, loader = _session(tmp_path)

    first = session.bind_instruction("toast bread")
    second = session.bind_instruction("toast bread")

    assert isinstance(first, GuideInput)
    assert first is second
    assert session.materialization_count == 1
    assert session.identity == {
        "catalog_digest": "catalog-digest",
        "document_id": "doc-3",
        "source_episode_index": 3,
        "task_index": 4,
    }
    assert [
        frame["episode_frame_index"] for frame in loader.calls[0][1]
    ] == [0, 25]


def test_session_rejects_task_switch_until_cleared(tmp_path):
    session, _ = _session(tmp_path)
    session.bind_instruction("toast bread")

    with pytest.raises(ValueError, match="cannot switch"):
        session.bind_instruction("put cup away")

    session.clear()
    assert session.guide is None
    assert session.guidance is None


def test_session_rejects_missing_task_guidance(tmp_path):
    session, _ = _session(tmp_path)
    with pytest.raises(ValueError, match="No accepted Guidance"):
        session.bind_instruction("unknown task")
    with pytest.raises(ValueError, match="non-empty string"):
        session.bind_instruction("")


def test_session_rejects_instruction_shared_by_multiple_tasks(tmp_path):
    catalog = _Catalog()
    ambiguous = SimpleNamespace(
        document_id="doc-9",
        task_index=9,
        task_instruction="toast bread",
        source_episode_index=9,
        document=catalog.guidance.document,
    )
    catalog.documents = (*catalog.documents, ambiguous)
    session, _ = _session(tmp_path, catalog=catalog)

    with pytest.raises(ValueError, match=r"ambiguous.*\[4, 9\]"):
        session.bind_instruction("toast bread")
