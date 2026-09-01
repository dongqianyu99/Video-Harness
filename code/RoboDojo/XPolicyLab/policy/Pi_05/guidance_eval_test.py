from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig, materialize_guide
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_materialization_cache import ensure_guide_materialization_cache
import pytest

from XPolicyLab.policy.Pi_05.guidance_eval import TaskGuideSession


class _Catalog:
    def __init__(self):
        self.catalog_digest = "catalog-digest"
        document = {"source": {"episode_index": 3}}
        self.guidance = SimpleNamespace(
            document_id="doc-3",
            document_sha256="sha-3",
            task_index=4,
            task_instruction="toast bread",
            source_episode_index=3,
            document=document,
        )
        self.other = SimpleNamespace(
            document_id="doc-8",
            document_sha256="sha-8",
            task_index=5,
            task_instruction="put cup away",
            source_episode_index=8,
            document=document,
        )
        self.documents = (self.guidance, self.other)

    def by_document_id(self, document_id):
        return next(value for value in self.documents if value.document_id == document_id)


class _FrameLoader:
    def __init__(self):
        self.calls = []

    def load_views_rgb_many(self, document, frame_refs):
        self.calls.append((document, tuple(frame_refs)))
        view = np.zeros((2, 3, 3), dtype=np.uint8)
        return tuple((view.copy(), view.copy(), view.copy()) for _ in frame_refs)


class _Tokenizer:
    cache_digest = "eval-test-tokenizer"

    def tokenize_text(self, _text):
        return (
            np.asarray([1, 2, 0, 0], dtype=np.int32),
            np.asarray([True, True, False, False], dtype=np.bool_),
        )


def _plan_builder(_catalog, *, document_id):
    source = next(value for value in _catalog.documents if value.document_id == document_id)
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
        document_id=document_id,
        source_episode_index=source.source_episode_index,
        task_index=source.task_index,
        task_instruction=source.task_instruction,
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
    cache_root = tmp_path / "guide-cache"
    documents.mkdir()
    loader = _FrameLoader()
    catalog = _Catalog() if catalog is None else catalog
    guide_catalog = GuideCatalog.from_document_catalog(catalog)
    plans = {
        record.document_id: _plan_builder(catalog, document_id=record.document_id)
        for record in guide_catalog.records
    }
    config = GuideMaterializerConfig(
        max_boundaries=2,
        max_units=1,
        max_boundary_text_tokens=4,
        max_transition_text_tokens=4,
        boundary_num_queries=2,
        transition_num_queries=1,
    )
    tokenizer = _Tokenizer()

    def source_resolver(record):
        plan = plans[record.document_id]

        def decode(boundaries):
            payloads = loader.load_views_rgb_many(
                catalog.by_document_id(record.document_id).document,
                tuple(
                    {
                        "episode_frame_index": boundary.episode_frame_index,
                        "timestamp_s": boundary.timestamp_s,
                    }
                    for boundary in boundaries
                ),
            )
            return tuple(np.stack(views, axis=0) for views in payloads)

        return materialize_guide(
            plan,
            boundary_decoder=lambda boundary: decode((boundary,))[0],
            boundaries_decoder=decode,
            boundary_tokenizer=tokenizer,
            transition_tokenizer=tokenizer,
            config=config,
        )

    ensure_guide_materialization_cache(
        cache_root=cache_root,
        catalog_digest=guide_catalog.catalog_digest,
        guide_records=guide_catalog.records,
        document_catalog=catalog,
        plans_by_document=plans,
        materializer_config=config,
        boundary_tokenizer=tokenizer,
        transition_tokenizer=tokenizer,
        source_resolver=source_resolver,
    )
    build_calls = len(loader.calls)
    session = TaskGuideSession(
        documents_root=documents,
        guide_materialization_cache_root=cache_root,
        max_boundaries=2,
        max_units=1,
        max_boundary_text_tokens=4,
        max_transition_text_tokens=4,
        boundary_num_queries=2,
        transition_num_queries=1,
        catalog=catalog,
        boundary_tokenizer=_Tokenizer(),
        transition_tokenizer=_Tokenizer(),
        plan_builder=_plan_builder,
    )
    return session, loader, build_calls


def test_session_materializes_once_and_reuses_first_task_guide(tmp_path):
    session, loader, build_calls = _session(tmp_path)

    first = session.bind_instruction("toast bread")
    second = session.bind_instruction("toast bread")

    assert isinstance(first, GuideInput)
    assert first is second
    assert session.materialization_count == 1
    assert session.identity == {
        "catalog_digest": "catalog-digest",
        "guide_materialization_cache_digest": session.identity[
            "guide_materialization_cache_digest"
        ],
        "document_id": "doc-3",
        "source_episode_index": 3,
        "task_index": 4,
    }
    assert len(loader.calls) == build_calls


def test_session_rejects_task_switch_until_cleared(tmp_path):
    session, _, _ = _session(tmp_path)
    session.bind_instruction("toast bread")

    with pytest.raises(ValueError, match="cannot switch"):
        session.bind_instruction("put cup away")

    session.clear()
    assert session.guide is None
    assert session.guidance is None


def test_session_rejects_missing_task_guidance(tmp_path):
    session, _, _ = _session(tmp_path)
    with pytest.raises(ValueError, match="No accepted Guidance"):
        session.bind_instruction("unknown task")
    with pytest.raises(ValueError, match="non-empty string"):
        session.bind_instruction("")


def test_session_rejects_instruction_shared_by_multiple_tasks(tmp_path):
    catalog = _Catalog()
    ambiguous = SimpleNamespace(
        document_id="doc-9",
        document_sha256="sha-9",
        task_index=9,
        task_instruction="toast bread",
        source_episode_index=9,
        document=catalog.guidance.document,
    )
    catalog.documents = (*catalog.documents, ambiguous)
    session, _, _ = _session(tmp_path, catalog=catalog)

    with pytest.raises(ValueError, match=r"ambiguous.*\[4, 9\]"):
        session.bind_instruction("toast bread")
