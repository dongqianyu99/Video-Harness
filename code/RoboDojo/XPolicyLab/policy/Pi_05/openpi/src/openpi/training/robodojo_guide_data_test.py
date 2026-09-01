from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.training import robodojo_guide_data as _guide_data
from openpi.training import robodojo_guide_resolver as _guide_resolver
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig
from openpi.training.robodojo_guide_data import create_robodojo_guided_data_loader


@dataclasses.dataclass(frozen=True)
class _Group:
    inputs: tuple = ()


@dataclasses.dataclass(frozen=True)
class _DataConfig:
    repo_id: str = "native"
    norm_stats: object | None = None
    repack_transforms: _Group = _Group()
    data_transforms: _Group = _Group()
    model_transforms: _Group = _Group()
    use_quantile_norm: bool = False
    asset_id: str = "fake"


class _DataFactory:
    def create(self, *_args):
        return _DataConfig()


class _Normalize:
    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, data):
        return data


_TRANSFORMS = SimpleNamespace(Normalize=_Normalize)


class _Dataset:
    def __init__(self):
        self.samples = [
            {
                "image": {
                    key: np.full((2, 2, 3), index, dtype=np.float32)
                    for key in _model.IMAGE_KEYS
                },
                "image_mask": {
                    key: np.asarray(np.bool_(1), dtype=np.bool_) for key in _model.IMAGE_KEYS
                },
                "state": np.asarray([index, index + 1], dtype=np.float32),
                "actions": np.full((50, 32), index, dtype=np.float32),
                "episode_index": np.asarray(10 if index < 4 else 20, dtype=np.int64),
                "task_index": np.asarray(0 if index < 4 else 1, dtype=np.int64),
            }
            for index in range(8)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _Catalog:
    def __init__(self):
        self.build_id = "build"
        self.catalog_digest = "catalog-digest"
        self.exclusions = (SimpleNamespace(reason="quarantined"),)
        documents = []
        for task in range(2):
            episode_index = 10 if task == 0 else 20
            views = {
                alias: {
                    "camera_key": camera_key,
                    "video_path": f"videos/{episode_index}-{alias}.mp4",
                    "video_from_timestamp": float(task),
                }
                for alias, camera_key in {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                }.items()
            }
            documents.append(
                SimpleNamespace(
                    document_id=f"doc-{task}",
                    document_sha256=f"sha-{task}",
                    source_episode_index=episode_index,
                    task_index=task,
                    task_instruction=f"task {task}",
                    document={
                        "document_id": f"doc-{task}",
                        "source": {
                            "episode_index": episode_index,
                            "episode_length": 4,
                            "task_index": task,
                            "dataset_from_index": 0 if task == 0 else 4,
                            "dataset_to_index": 4 if task == 0 else 8,
                            "data_path": f"data/{episode_index}.parquet",
                            "fps": 25,
                            "views": views,
                            "camera_key": "observation.images.cam_high",
                            "video_path": views["cam_high"]["video_path"],
                            "video_from_timestamp": views["cam_high"]["video_from_timestamp"],
                        },
                    },
                )
            )
        self.documents = tuple(documents)
        self._by_id = {document.document_id: document for document in self.documents}

    def by_document_id(self, document_id):
        return self._by_id[document_id]


def _plan(document_id: str):
    task = int(document_id.rsplit("-", 1)[-1])
    source_episode = 10 if task == 0 else 20
    boundaries = tuple(
        SimpleNamespace(
            boundary_id=f"b{index}",
            order=index,
            slot=index,
            episode_frame_index=index,
            timestamp_s=index / 25,
            view_texts=tuple(
                f"{document_id} boundary {index} view {view}" for view in range(3)
            ),
        )
        for index in range(2)
    )
    return SimpleNamespace(
        document_id=document_id,
        source_episode_index=source_episode,
        task_index=task,
        task_instruction=f"task {task}",
        boundaries=boundaries,
        units=(
            SimpleNamespace(
                unit_id="u0",
                order=0,
                before_slot=0,
                after_slot=1,
                transition_text=f"{document_id} transition",
            ),
        ),
    )


class _Tokenizer:
    def __init__(self, size):
        self.size = size
        self.calls = []

    @property
    def cache_digest(self):
        return f"test-tokenizer-{self.size}"

    def tokenize_text(self, text):
        self.calls.append(text)
        tokens = np.zeros(self.size, dtype=np.int32)
        mask = np.zeros(self.size, dtype=np.bool_)
        tokens[:2] = (len(text), 1)
        mask[:2] = True
        return tokens, mask


class _FrameLoader:
    def __init__(self):
        self.calls = []

    def load_views_rgb_many(self, document, refs):
        self.calls.append((document["document_id"], tuple(refs)))
        marker = int(document["document_id"].rsplit("-", 1)[-1]) * 20
        return tuple(
            tuple(
                np.full((3, 4, 3), marker + frame * 3 + view, dtype=np.uint8)
                for view in range(3)
            )
            for frame, _ref in enumerate(refs)
        )


def _episodes():
    records = []
    for task, episode_index in enumerate((10, 20)):
        videos = tuple(
            SimpleNamespace(
                key=camera_key,
                path=f"videos/{episode_index}-{alias}.mp4",
                from_timestamp=float(task),
            )
            for alias, camera_key in {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            }.items()
        )
        records.append(
            SimpleNamespace(
                episode_index=episode_index,
                task_index=task,
                task_instruction=f"task {task}",
                length=4,
                dataset_from_index=0 if task == 0 else 4,
                dataset_to_index=4 if task == 0 else 8,
                data_path=f"data/{episode_index}.parquet",
                videos=videos,
            )
        )
    return records


def _config(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    documents_root = tmp_path / "documents"
    dataset_root.mkdir()
    documents_root.mkdir()
    return RoboDojoGuidedDataConfig(
        repo_id="fake",
        dataset_root=dataset_root,
        documents_root=documents_root,
        guide_materialization_cache_root=tmp_path / "guide-cache",
        guides_per_batch=2,
        queries_per_guide=2,
        seed=5,
        max_boundaries=2,
        max_units=1,
        max_boundary_text_tokens=8,
        max_transition_text_tokens=6,
        guide_boundary_num_queries=8,
        guide_transition_num_queries=4,
        guide_cache_entries=2,
    )


def _native_train_config():
    return SimpleNamespace(
        data=_DataFactory(),
        assets_dirs=(),
        model=SimpleNamespace(action_horizon=50),
    )


def test_config_requires_cache_root_disjoint_from_source_roots(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        dataclasses.replace(
            config,
            guide_materialization_cache_root=config.dataset_root / "guide-cache",
        )


def test_real_repo_id_must_resolve_to_dataset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "data"))
    _guide_data._validate_repo_id_root(  # noqa: SLF001
        "RoboDojo_lerobot_v30_video",
        tmp_path / "data" / "RoboDojo_lerobot_v30_video",
    )
    with pytest.raises(ValueError, match="different datasets"):
        _guide_data._validate_repo_id_root(  # noqa: SLF001
            "another-repo",
            tmp_path / "data" / "RoboDojo_lerobot_v30_video",
        )


def test_factory_builds_global_guidance_first_batch_and_three_view_input(tmp_path):
    config = _config(tmp_path)
    catalog = _Catalog()
    boundary_tokenizer = _Tokenizer(8)
    transition_tokenizer = _Tokenizer(6)
    frame_loader = _FrameLoader()
    loader = create_robodojo_guided_data_loader(
        _native_train_config(),
        config,
        num_batches=1,
        skip_norm_stats=True,
        dataset_factory=lambda *_args: _Dataset(),
        catalog_loader=lambda _root: catalog,
        episode_reader=lambda _root: _episodes(),
        boundary_tokenizer=boundary_tokenizer,
        transition_tokenizer=transition_tokenizer,
        frame_loader=frame_loader,
        plan_builder=lambda _catalog, *, document_id: _plan(document_id),
        transforms_module=_TRANSFORMS,
    )
    batch = next(iter(loader))

    assert batch.actions.shape == (2, 2, 50, 32)
    assert batch.guide.boundary_images.shape == (2, 2, 3, 224, 224, 3)
    assert batch.guide.boundary_text_tokens.shape == (2, 2, 3, 8)
    assert batch.guide.transition_text_tokens.shape == (2, 1, 6)
    assert batch.guide.memory_mask.shape == (2, 20)
    # Cache build materializes each Guide once; batch collation performs no decode.
    assert len(frame_loader.calls) == 2
    assert len(boundary_tokenizer.calls) == 12
    assert len(transition_tokenizer.calls) == 2
    assert loader.guide_catalog.catalog_digest == "catalog-digest"
    assert loader.host_metadata["accepted_guides"] == 2
    assert loader.host_metadata["excluded_guides"] == 1
    assert loader.host_metadata["guide_max_units"] == 1
    assert loader.host_metadata["guide_max_boundaries"] == 2
    assert loader.host_metadata["guide_materialization_cache"]["documents"] == 2
    assert loader.host_metadata["guide_materialization_cache"]["built"] == 2
    assert loader.host_metadata["source_media_validation"] == "cache_only"
    assert len(loader.host_metadata["task_sample_digest"]) == 64
    assert len(loader.host_metadata["guide_representation_digest"]) == 64
    # Each task has only its source episode in the native pool; successful
    # batching therefore proves source-episode queries are allowed.
    task_groups = {
        bool(np.all(batch.observation.state[group, :, 0] >= 4))
        for group in range(2)
    }
    assert task_groups == {False, True}


def test_factory_preflights_every_accepted_guide_before_returning(tmp_path):
    class _LateFailureLoader(_FrameLoader):
        def load_views_rgb_many(self, document, refs):
            refs = tuple(refs)
            self.calls.append((document["document_id"], refs))
            if document["document_id"] == "doc-1":
                raise RuntimeError("corrupt right-wrist video")
            return tuple(
                tuple(
                    np.zeros((3, 4, 3), dtype=np.uint8)
                    for _view in range(3)
                )
                for _ref in refs
            )

    frame_loader = _LateFailureLoader()
    plan_calls = []

    with pytest.raises(
        ValueError,
        match=r"Guide resolution failed.*doc-1.*source_episode_index=20",
    ):
        create_robodojo_guided_data_loader(
            _native_train_config(),
            _config(tmp_path),
            skip_norm_stats=True,
            dataset_factory=lambda *_args: _Dataset(),
            catalog_loader=lambda _root: _Catalog(),
            episode_reader=lambda _root: _episodes(),
            boundary_tokenizer=_Tokenizer(8),
            transition_tokenizer=_Tokenizer(6),
            frame_loader=frame_loader,
            plan_builder=lambda _catalog, *, document_id: (
                plan_calls.append(document_id) or _plan(document_id)
            ),
            transforms_module=_TRANSFORMS,
        )

    assert plan_calls == ["doc-0", "doc-1"]
    assert [document_id for document_id, _refs in frame_loader.calls] == [
        "doc-0",
        "doc-1",
    ]


def test_shared_maximum_shape_rejects_oversized_guide_plan():
    document_catalog = _Catalog()
    guide_catalog = GuideCatalog.from_document_catalog(document_catalog)
    oversized = _plan("doc-0")
    oversized.units = (*oversized.units, oversized.units[0])

    with pytest.raises(ValueError, match="exceeds the shared maximum shape"):
        _guide_data._build_guide_plans(  # noqa: SLF001
            document_catalog,
            guide_catalog,
            max_units=1,
            max_boundaries=2,
            plan_builder=lambda *_args, **_kwargs: oversized,
        )


def test_worker_loader_preflights_in_parent_before_workers_start(
    tmp_path,
    monkeypatch,
):
    catalog = _Catalog()
    frame_loader = _FrameLoader()
    monkeypatch.setattr(
        _guide_data,
        "_load_document_catalog",
        lambda *_args: catalog,
    )
    monkeypatch.setattr(
        _guide_data,
        "_read_episode_records",
        lambda *_args: _episodes(),
    )
    monkeypatch.setattr(
        _guide_data,
        "_build_guide_plans",
        lambda _catalog, guide_catalog, **_kwargs: (
            {
                record.document_id: _plan(record.document_id)
                for record in guide_catalog.records
            },
            {record.document_id: (1, 2) for record in guide_catalog.records},
        ),
    )
    monkeypatch.setattr(
        _guide_resolver,
        "_default_frame_loader",
        lambda _root: frame_loader,
    )
    monkeypatch.setattr(
        _guide_data,
        "_make_tokenizer",
        lambda _tokenizer, size: _Tokenizer(size),
    )

    loader = create_robodojo_guided_data_loader(
        _native_train_config(),
        dataclasses.replace(_config(tmp_path), num_workers=1),
        skip_norm_stats=True,
        dataset_factory=lambda *_args: _Dataset(),
        transforms_module=_TRANSFORMS,
    )

    assert loader.host_metadata["guide_materialization_cache"]["documents"] == 2
    assert [call[0] for call in frame_loader.calls] == [
        "doc-0",
        "doc-1",
    ]


def test_factory_preserves_capacity_fields_in_shared_materialization(tmp_path):
    config = dataclasses.replace(
        _config(tmp_path),
        guides_per_batch=1,
        guide_boundary_num_queries=12,
        guide_transition_num_queries=8,
    )
    loader = create_robodojo_guided_data_loader(
        _native_train_config(),
        config,
        num_batches=1,
        skip_norm_stats=True,
        dataset_factory=lambda *_args: _Dataset(),
        catalog_loader=lambda _root: _Catalog(),
        episode_reader=lambda _root: _episodes(),
        boundary_tokenizer=_Tokenizer(8),
        transition_tokenizer=_Tokenizer(6),
        frame_loader=_FrameLoader(),
        plan_builder=lambda _catalog, *, document_id: _plan(document_id),
        transforms_module=_TRANSFORMS,
    )
    batch = next(iter(loader))

    assert batch.guide.memory_mask.shape[-1] == 32
    assert loader.host_metadata["guide_boundary_num_queries"] == 12
    assert loader.host_metadata["guide_transition_num_queries"] == 8


def test_factory_rejects_catalog_dataset_task_mismatch(tmp_path):
    config = _config(tmp_path)
    catalog = _Catalog()
    catalog.documents[0].task_index = 99
    with pytest.raises(ValueError, match="source task mismatch"):
        create_robodojo_guided_data_loader(
            _native_train_config(),
            config,
            skip_norm_stats=True,
            dataset_factory=lambda *_args: _Dataset(),
            catalog_loader=lambda _root: catalog,
            episode_reader=lambda _root: _episodes(),
            boundary_tokenizer=_Tokenizer(8),
            transition_tokenizer=_Tokenizer(6),
            frame_loader=_FrameLoader(),
            plan_builder=lambda _catalog, *, document_id: _plan(document_id),
            transforms_module=_TRANSFORMS,
        )


@pytest.mark.parametrize(
    ("case", "pattern"),
    [
        ("episode_length", "episode_length mismatch"),
        ("fps", "fps mismatch"),
        ("dataset_range", "dataset_from_index mismatch"),
        ("data_path", "data_path mismatch"),
        ("view_path", "video_path mismatch"),
        ("view_timestamp", "video_from_timestamp mismatch"),
    ],
)
def test_factory_cross_checks_document_source_against_episode_metadata(
    tmp_path, case, pattern
):
    config = _config(tmp_path)
    catalog = _Catalog()
    source = catalog.documents[0].document["source"]
    if case == "episode_length":
        source["episode_length"] = 5
    elif case == "fps":
        source["fps"] = 24
    elif case == "dataset_range":
        source["dataset_from_index"] = 1
    elif case == "data_path":
        source["data_path"] = "wrong.parquet"
    elif case == "view_path":
        source["views"]["cam_left_wrist"]["video_path"] = "wrong.mp4"
    else:
        source["views"]["cam_right_wrist"]["video_from_timestamp"] = 9.0

    with pytest.raises(ValueError, match=pattern):
        create_robodojo_guided_data_loader(
            _native_train_config(),
            config,
            skip_norm_stats=True,
            dataset_factory=lambda *_args: _Dataset(),
            catalog_loader=lambda _root: catalog,
            episode_reader=lambda _root: _episodes(),
            boundary_tokenizer=_Tokenizer(8),
            transition_tokenizer=_Tokenizer(6),
            frame_loader=_FrameLoader(),
            plan_builder=lambda _catalog, *, document_id: _plan(document_id),
            transforms_module=_TRANSFORMS,
        )


def test_factory_rejects_custom_worker_dependencies(tmp_path):
    config = dataclasses.replace(_config(tmp_path), num_workers=1)
    with pytest.raises(ValueError, match="require num_workers=0"):
        create_robodojo_guided_data_loader(
            _native_train_config(),
            config,
            skip_norm_stats=True,
            dataset_factory=lambda *_args: _Dataset(),
            catalog_loader=lambda _root: _Catalog(),
            episode_reader=lambda _root: _episodes(),
            transforms_module=_TRANSFORMS,
        )
