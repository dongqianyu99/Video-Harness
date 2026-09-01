from dataclasses import dataclass
from pathlib import Path
import pickle
from types import MappingProxyType
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training import robodojo_guide_resolver as _resolver_module
from openpi.training.guide_dataset import GuideRecord
from openpi.training.robodojo_guide_resolver import GuideDocumentSnapshot
from openpi.training.robodojo_guide_resolver import RoboDojoGuideResolverFactory
from openpi.training.robodojo_guide_resolver import VideoHarnessGuideResolver
from openpi.training.robodojo_guide_resolver import preflight_guide_media


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
    document_id: str
    source_episode_index: int
    task_index: int
    task_instruction: str
    boundaries: tuple[_Boundary, ...]
    units: tuple[_Unit, ...]


@dataclass(frozen=True)
class _Source:
    document_id: str
    source_episode_index: int
    task_index: int
    task_instruction: str
    document: MappingProxyType


class _Catalog:
    def __init__(self, source: _Source):
        self.source = source

    def by_document_id(self, document_id: str) -> _Source:
        if document_id != self.source.document_id:
            raise ValueError(document_id)
        return self.source


class _CatalogMany:
    def __init__(self, sources: tuple[_Source, ...]):
        self.sources = {source.document_id: source for source in sources}

    def by_document_id(self, document_id: str) -> _Source:
        return self.sources[document_id]


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


class _FrameLoader:
    def __init__(self):
        self.calls = []

    def load_views_rgb_many(self, document, frame_refs):
        refs = tuple(frame_refs)
        self.calls.append((document, refs))
        return tuple(
            tuple(np.full((4, 6, 3), 20 * view + ref["episode_frame_index"], dtype=np.uint8) for view in range(3))
            for ref in refs
        )


def _setup():
    record = GuideRecord(0, "doc", 10, 3, "stack blocks")
    source = _Source(
        "doc",
        10,
        3,
        "stack blocks",
        MappingProxyType({"document_id": "doc", "source": MappingProxyType({})}),
    )
    plan = _Plan(
        "doc",
        10,
        3,
        "stack blocks",
        (
            _Boundary("b0000", 0, 0, 0, 0.0, ("h0", "l0", "r0")),
            _Boundary("b0001", 1, 1, 10, 0.4, ("h1", "l1", "r1")),
        ),
        (_Unit("u0000", 0, 0, 1, "move then place"),),
    )
    return record, _Catalog(source), plan


def _config(**overrides):
    values = {
        "max_boundaries": 2,
        "max_units": 1,
        "max_boundary_text_tokens": 4,
        "max_transition_text_tokens": 5,
        "boundary_num_queries": 2,
        "transition_num_queries": 1,
    }
    values.update(overrides)
    return GuideMaterializerConfig(**values)


def test_resolver_builds_three_view_guide_from_document_record() -> None:
    record, catalog, plan = _setup()
    frame_loader = _FrameLoader()
    boundary_tokenizer = _Tokenizer(4)
    transition_tokenizer = _Tokenizer(5)
    calls = []

    def plan_builder(catalog_arg, *, document_id):
        calls.append((catalog_arg, document_id))
        return plan

    resolver = VideoHarnessGuideResolver(
        document_catalog=catalog,
        guide_records=(record,),
        dataset_root=Path("/dataset"),
        boundary_tokenizer=boundary_tokenizer,
        transition_tokenizer=transition_tokenizer,
        materializer_config=_config(),
        frame_loader=frame_loader,
        plan_builder=plan_builder,
    )

    guide = resolver(record)

    assert isinstance(guide, GuideInput)
    assert guide.boundary_images.shape == (1, 2, 3, 224, 224, 3)
    assert guide.boundary_text_tokens.shape == (1, 2, 3, 4)
    assert guide.transition_text_tokens.shape == (1, 1, 5)
    assert calls == [(catalog, "doc")]
    assert boundary_tokenizer.calls == ["h0", "l0", "r0", "h1", "l1", "r1"]
    assert transition_tokenizer.calls == ["move then place"]
    assert frame_loader.calls[0][1] == (
        {"episode_frame_index": 0, "timestamp_s": 0.0},
        {"episode_frame_index": 10, "timestamp_s": 0.4},
    )
    assert all(not isinstance(leaf, str) for leaf in jax.tree_util.tree_leaves(guide))


def test_worker_factory_uses_pickled_parent_snapshot_without_catalog_reload(
    monkeypatch,
) -> None:
    record, catalog, plan = _setup()
    frame_loader = _FrameLoader()
    imports = []

    def import_module(name):
        imports.append(name)
        if name == "video_harness.reader":
            raise AssertionError("worker must not reload the Document catalog")
        if name == "openpi.models.tokenizer":
            return SimpleNamespace(PaligemmaTokenizer=_Tokenizer)
        raise AssertionError(f"unexpected import {name!r}")

    monkeypatch.setattr(_resolver_module.importlib, "import_module", import_module)
    monkeypatch.setattr(
        _resolver_module,
        "_default_frame_loader",
        lambda _root: frame_loader,
    )
    factory = RoboDojoGuideResolverFactory(
        dataset_root=Path("/dataset"),
        guide_records=(record,),
        document_snapshots=(
            GuideDocumentSnapshot.from_guide_document(catalog.source),
        ),
        guide_plans=(plan,),
        materializer_config=_config(),
    )
    assert factory.document_snapshots[0].document == {
        "document_id": "doc",
        "source": {},
    }

    resolver = pickle.loads(pickle.dumps(factory))()
    guide = resolver(record)

    assert isinstance(guide, GuideInput)
    assert imports == ["openpi.models.tokenizer"]
    assert frame_loader.calls[0][0]["document_id"] == "doc"


def test_resolver_uses_shared_maximum_shape() -> None:
    record, catalog, plan = _setup()
    resolver = VideoHarnessGuideResolver(
        document_catalog=catalog,
        guide_records=(record,),
        dataset_root=Path("/dataset"),
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        materializer_config=_config(max_boundaries=3, max_units=2),
        frame_loader=_FrameLoader(),
        plan_builder=lambda *_args, **_kwargs: plan,
    )

    guide = resolver(record)

    assert guide.boundary_images.shape[:3] == (1, 3, 3)
    assert guide.unit_mask.shape == (1, 2)


def test_resolver_rejects_unregistered_record() -> None:
    record, catalog, plan = _setup()
    resolver = VideoHarnessGuideResolver(
        document_catalog=catalog,
        guide_records=(record,),
        dataset_root=Path("/dataset"),
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        materializer_config=_config(),
        frame_loader=_FrameLoader(),
        plan_builder=lambda *_args, **_kwargs: plan,
    )

    with pytest.raises(ValueError, match="immutable record"):
        resolver(GuideRecord(0, "other", 10, 3, "stack blocks"))


@pytest.mark.parametrize("field", ["document_id", "source_episode_index", "task_index", "task_instruction"])
def test_resolver_rejects_plan_identity_drift(field: str) -> None:
    record, catalog, plan = _setup()
    values = dict(plan.__dict__)
    values[field] = "other" if isinstance(values[field], str) else values[field] + 1
    invalid = _Plan(**values)
    resolver = VideoHarnessGuideResolver(
        document_catalog=catalog,
        guide_records=(record,),
        dataset_root=Path("/dataset"),
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        materializer_config=_config(),
        frame_loader=_FrameLoader(),
        plan_builder=lambda *_args, **_kwargs: invalid,
    )

    with pytest.raises(ValueError, match=f"GuidePlan {field} mismatch"):
        resolver(record)


def test_resolver_rejects_non_rgb_or_missing_view_payload() -> None:
    record, catalog, plan = _setup()

    class _BadLoader:
        def load_views_rgb_many(self, _document, _refs):
            return ((np.zeros((4, 6), dtype=np.uint8),) * 3,) * 2

    resolver = VideoHarnessGuideResolver(
        document_catalog=catalog,
        guide_records=(record,),
        dataset_root=Path("/dataset"),
        boundary_tokenizer=_Tokenizer(4),
        transition_tokenizer=_Tokenizer(5),
        materializer_config=_config(),
        frame_loader=_BadLoader(),
        plan_builder=lambda *_args, **_kwargs: plan,
    )

    with pytest.raises(ValueError, match="RGB"):
        resolver(record)


def _preflight_setup():
    record_a, _, plan_a = _setup()
    record_b = GuideRecord(1, "doc-b", 11, 4, "close drawer")
    source_a = _Source(
        "doc",
        10,
        3,
        "stack blocks",
        MappingProxyType({"document_id": "doc", "source": MappingProxyType({})}),
    )
    source_b = _Source(
        "doc-b",
        11,
        4,
        "close drawer",
        MappingProxyType({"document_id": "doc-b", "source": MappingProxyType({})}),
    )
    plan_b = _Plan(
        "doc-b",
        11,
        4,
        "close drawer",
        (_Boundary("b0000", 0, 0, 5, 0.2, ("h", "l", "r")),),
        (_Unit("u0000", 0, 0, 0, "close"),),
    )
    return (
        (record_b, record_a),
        _CatalogMany((source_a, source_b)),
        {"doc": plan_a, "doc-b": plan_b},
    )


def test_preflight_decodes_all_guides_in_stable_order_and_returns_counts() -> None:
    records, catalog, plans = _preflight_setup()
    frame_loader = _FrameLoader()

    counts = preflight_guide_media(
        document_catalog=catalog,
        guide_records=records,
        plans_by_document=plans,
        dataset_root=Path("/dataset"),
        frame_loader=frame_loader,
    )

    assert counts == {"documents": 2, "boundaries": 3, "camera_frames": 9}
    assert [call[0]["document_id"] for call in frame_loader.calls] == ["doc", "doc-b"]
    assert frame_loader.calls[0][1] == (
        {"episode_frame_index": 0, "timestamp_s": 0.0},
        {"episode_frame_index": 10, "timestamp_s": 0.4},
    )
    assert frame_loader.calls[1][1] == ({"episode_frame_index": 5, "timestamp_s": 0.2},)


def test_preflight_late_decode_failure_has_guide_context() -> None:
    records, catalog, plans = _preflight_setup()

    class _FailingLoader(_FrameLoader):
        def load_views_rgb_many(self, document, frame_refs):
            if document["document_id"] == "doc-b":
                self.calls.append((document, tuple(frame_refs)))
                raise RuntimeError("corrupt media")
            return super().load_views_rgb_many(document, frame_refs)

    frame_loader = _FailingLoader()
    with pytest.raises(
        ValueError,
        match=("media preflight failed for guide_index=1, document_id='doc-b', source_episode_index=11: corrupt media"),
    ):
        preflight_guide_media(
            document_catalog=catalog,
            guide_records=records,
            plans_by_document=plans,
            dataset_root=Path("/dataset"),
            frame_loader=frame_loader,
        )

    assert [call[0]["document_id"] for call in frame_loader.calls] == ["doc", "doc-b"]


@pytest.mark.parametrize("failure", ["short", "malformed"])
def test_preflight_rejects_short_or_malformed_boundary_payloads(failure: str) -> None:
    record, catalog, plan = _setup()

    class _BadLoader:
        def load_views_rgb_many(self, _document, refs):
            if failure == "short":
                return tuple((np.zeros((4, 6, 3), dtype=np.uint8),) * 3 for _ in tuple(refs)[:-1])
            return tuple((np.zeros((4, 6), dtype=np.uint8),) * 3 for _ in refs)

    expected = "unexpected number of Boundaries" if failure == "short" else "RGB shape"
    with pytest.raises(ValueError, match=expected):
        preflight_guide_media(
            document_catalog=catalog,
            guide_records=(record,),
            plans_by_document={"doc": plan},
            dataset_root=Path("/dataset"),
            frame_loader=_BadLoader(),
        )
