from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.training import robodojo_guide_data as _guide_data
from openpi.training.guide_buckets import GuideLengthBucket
from openpi.training.guide_sampler import QueryEpisodeRange
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig
from openpi.training.robodojo_guide_data import create_robodojo_guided_data_loader


@dataclass(frozen=True)
class _Group:
    inputs: tuple[Any, ...] = ()


class _Normalize:
    def __init__(self, _norm_stats: Any, *, use_quantiles: bool):
        self.use_quantiles = use_quantiles

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


_TRANSFORMS = SimpleNamespace(Normalize=_Normalize)


@dataclass(frozen=True)
class _DataConfig:
    repo_id: str
    norm_stats: dict[str, Any] | None
    asset_id: str | None = "robodojo-test"
    repack_transforms: _Group = _Group()
    data_transforms: _Group = _Group()
    model_transforms: _Group = _Group()
    use_quantile_norm: bool = False


@dataclass(frozen=True)
class _ModelConfig:
    action_horizon: int = 50


class _NativeDataFactory:
    def __init__(self, data_config: _DataConfig):
        self.data_config = data_config
        self.calls: list[tuple[Path, _ModelConfig]] = []

    def create(self, assets_dir: Path, model_config: _ModelConfig) -> _DataConfig:
        self.calls.append((assets_dir, model_config))
        return self.data_config


@dataclass(frozen=True)
class _NativeTrainConfig:
    data: _NativeDataFactory
    model: _ModelConfig
    assets_dirs: Path


class _NativeDataset:
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.accessed.append(index)
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.samples[index].items()
        }


@dataclass(frozen=True)
class _Video:
    key: str
    path: str


@dataclass(frozen=True)
class _Episode:
    episode_index: int
    task_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    videos: tuple[_Video, ...]


@dataclass(frozen=True)
class _ArtifactBinding:
    query_episode_index: int
    support_episode_index: int
    task_index: int
    support_document_id: str


@dataclass(frozen=True)
class _Source:
    document_id: str
    episode_index: int
    task_index: int
    document: dict[str, Any]


@dataclass(frozen=True)
class _Bundle:
    build_id: str
    dataset: dict[str, Any]
    documents: tuple[_Source, ...]
    support_bindings: tuple[_ArtifactBinding, ...]


@dataclass(frozen=True)
class _Frame:
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
    frames: tuple[_Frame, ...]
    units: tuple[_Unit, ...]


def _rgb(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def _sample(index: int, *, episode_index: int, task_index: int) -> dict[str, Any]:
    return {
        "image": {
            key: np.full((2, 2, 3), index, dtype=np.float32)
            for key in _model.IMAGE_KEYS
        },
        "image_mask": {
            key: np.asarray(np.bool_(1), dtype=np.bool_)
            for key in _model.IMAGE_KEYS
        },
        "state": np.asarray([index, index + 1, index + 2, index + 3], dtype=np.float32),
        "actions": np.full((50, 32), index, dtype=np.float32),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "task_index": np.asarray(task_index, dtype=np.int64),
    }


def _make_setup(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    artifact_paths = []
    for name in ("dataset.json", "documents.jsonl", "pairs.jsonl"):
        path = tmp_path / name
        path.write_text("test\n", encoding="utf-8")
        artifact_paths.append(path)

    bindings = (
        _ArtifactBinding(100, 101, 4, "doc-0"),
        _ArtifactBinding(200, 201, 5, "doc-1"),
    )
    documents = tuple(
        _Source(
            document_id=f"doc-{binding_index}",
            episode_index=101 + binding_index * 100,
            task_index=4 + binding_index,
            document={
                "source": {
                    "episode_index": 101 + binding_index * 100,
                    "task_index": 4 + binding_index,
                    "episode_length": 4,
                    "fps": 25,
                    "camera_key": "observation.images.cam_high",
                    "video_path": f"videos/support-{binding_index}.mp4",
                }
            },
        )
        for binding_index in (0, 1)
    )
    bundle = _Bundle(
        build_id="build-test",
        dataset={
            "schema_version": "video-harness.robodojo-source",
            "fps": 25,
            "document_camera": "observation.images.cam_high",
        },
        documents=documents,
        support_bindings=bindings,
    )

    episodes = (
        _Episode(100, 4, 4, 0, 4, (_Video("observation.images.cam_high", "query-0.mp4"),)),
        _Episode(101, 4, 4, 8, 12, (_Video("observation.images.cam_high", "videos/support-0.mp4"),)),
        _Episode(200, 5, 4, 4, 8, (_Video("observation.images.cam_high", "query-1.mp4"),)),
        _Episode(201, 5, 4, 12, 16, (_Video("observation.images.cam_high", "videos/support-1.mp4"),)),
    )
    samples = [
        _sample(index, episode_index=100 if index < 4 else 200, task_index=4 if index < 4 else 5)
        for index in range(8)
    ]
    native_dataset = _NativeDataset(samples)
    native_factory = _NativeDataFactory(
        _DataConfig(repo_id="native-repo", norm_stats={"state": "stats"})
    )
    train_config = _NativeTrainConfig(native_factory, _ModelConfig(), tmp_path / "assets")

    plans = {
        100: _Plan(
            100,
            "doc-0",
            101,
            4,
            (_Frame("doc-0", 101, 0, 0.0), _Frame("doc-0", 101, 3, 0.12)),
            (_Unit(0, 1, "guide-0"),),
        ),
        200: _Plan(
            200,
            "doc-1",
            201,
            5,
            (_Frame("doc-1", 201, 0, 0.0), _Frame("doc-1", 201, 3, 0.12)),
            (_Unit(0, 1, "guide-1"),),
        ),
    }

    class _Tokenizer:
        def __init__(self):
            self.calls: list[str] = []

        def tokenize_text(self, text: str):
            self.calls.append(text)
            return np.asarray([1, 2, 0, 0], dtype=np.int32), np.asarray(
                [True, True, False, False], dtype=np.bool_
            )

    class _FrameLoader:
        def __init__(self):
            self.calls: list[tuple[Any, dict[str, Any]]] = []

        def load_rgb(self, document, frame_ref):
            self.calls.append((document, frame_ref))
            return _rgb(80 + frame_ref["episode_frame_index"])

    tokenizer = _Tokenizer()
    frame_loader = _FrameLoader()

    config = RoboDojoGuidedDataConfig(
        repo_id="fake",
        dataset_root=dataset_root,
        dataset_artifact_path=artifact_paths[0],
        documents_artifact_path=artifact_paths[1],
        pairs_artifact_path=artifact_paths[2],
        batch_size=2,
        seed=7,
        profile="actuator",
        max_frames=2,
        max_units=1,
        max_text_tokens=4,
    )
    return (
        config,
        train_config,
        native_dataset,
        bundle,
        episodes,
        plans,
        tokenizer,
        frame_loader,
        dataset_root,
    )


def test_factory_builds_real_shape_guided_batches_from_episode_ranges(tmp_path, monkeypatch):
    (
        config,
        train_config,
        native_dataset,
        bundle,
        episodes,
        plans,
        tokenizer,
        frame_loader,
        _dataset_root,
    ) = _make_setup(tmp_path)

    def dataset_factory(data_config, _action_horizon, _model_config):
        assert data_config.repo_id == "fake"
        return native_dataset

    loader = create_robodojo_guided_data_loader(
        train_config,
        config,
        num_batches=2,
        skip_norm_stats=True,
        dataset_factory=dataset_factory,
        artifact_loader=lambda **_paths: bundle,
        episode_reader=lambda _root: episodes,
        tokenizer=tokenizer,
        frame_loader=frame_loader,
        plan_builder=lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        transforms_module=_TRANSFORMS,
    )

    batches = list(loader)

    assert len(batches) == 2
    assert all(isinstance(batch, GuideConditionedBatch) for batch in batches)
    assert all(batch.actions.shape == (1, 2, 50, 32) for batch in batches)
    assert all(batch.observation.state.shape == (1, 2, 4) for batch in batches)
    assert len(tokenizer.calls) == 2
    assert len(frame_loader.calls) == 4
    assert native_dataset.accessed
    assert train_config.data.data_config.repo_id == "native-repo"
    assert all(not isinstance(leaf, str) for batch in batches for leaf in jax.tree_util.tree_leaves(batch))


def test_factory_query_subset_uses_explicit_episode_and_not_dataset_order(tmp_path):
    setup = _make_setup(tmp_path)
    config, train_config, _, bundle, episodes, plans, tokenizer, frame_loader, _ = setup
    config = dataclasses.replace(config, query_episode_indices=(200,))

    loader = create_robodojo_guided_data_loader(
        train_config,
        config,
        num_batches=1,
        skip_norm_stats=True,
        dataset_factory=lambda *_args: setup[2],
        artifact_loader=lambda **_paths: bundle,
        episode_reader=lambda _root: episodes,
        tokenizer=tokenizer,
        frame_loader=frame_loader,
        plan_builder=lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        transforms_module=_TRANSFORMS,
    )

    batch = next(iter(loader))

    assert np.all(np.asarray(batch.observation.state[0, :, 0]) >= 4)
    assert tokenizer.calls == ["guide-1"]


def test_factory_mixes_two_tasks_and_two_guides_in_one_grouped_batch(tmp_path):
    setup = _make_setup(tmp_path)
    config, train_config, _, bundle, episodes, plans, tokenizer, frame_loader, _ = setup
    config = dataclasses.replace(config, batch_size=4, guides_per_batch=2)

    loader = create_robodojo_guided_data_loader(
        train_config,
        config,
        num_batches=1,
        skip_norm_stats=True,
        dataset_factory=lambda *_args: setup[2],
        artifact_loader=lambda **_paths: bundle,
        episode_reader=lambda _root: episodes,
        tokenizer=tokenizer,
        frame_loader=frame_loader,
        plan_builder=lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        transforms_module=_TRANSFORMS,
    )

    batch = next(iter(loader))

    assert batch.actions.shape == (2, 2, 50, 32)
    assert batch.observation.state.shape == (2, 2, 4)
    assert batch.guide.images.shape[0] == 2
    assert set(tokenizer.calls) == {"guide-0", "guide-1"}
    assert len(frame_loader.calls) == 4
    assert loader.groups_per_batch == 2
    assert loader.queries_per_guide == 2


def test_factory_materializes_each_length_bucket_with_its_own_shape(tmp_path):
    setup = _make_setup(tmp_path)
    config, train_config, _, bundle, episodes, plans, tokenizer, frame_loader, _ = setup
    config = dataclasses.replace(
        config,
        max_frames=4,
        max_units=2,
        guide_length_buckets=(
            GuideLengthBucket(1, 2),
            GuideLengthBucket(2, 4),
        ),
    )
    plans = dict(plans)
    plans[200] = _Plan(
        200,
        "doc-1",
        201,
        5,
        (
            _Frame("doc-1", 201, 0, 0.0),
            _Frame("doc-1", 201, 2, 0.08),
            _Frame("doc-1", 201, 3, 0.12),
        ),
        (_Unit(0, 1, "guide-1a"), _Unit(1, 2, "guide-1b")),
    )

    loader = create_robodojo_guided_data_loader(
        train_config,
        config,
        num_batches=4,
        skip_norm_stats=True,
        dataset_factory=lambda *_args: setup[2],
        artifact_loader=lambda **_paths: bundle,
        episode_reader=lambda _root: episodes,
        tokenizer=tokenizer,
        frame_loader=frame_loader,
        plan_builder=lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        transforms_module=_TRANSFORMS,
    )

    batches = list(loader)

    assert {
        (batch.guide.images.shape[1], batch.guide.unit_mask.shape[1])
        for batch in batches
    } == {(2, 1), (4, 2)}
    assert loader.host_metadata["guide_binding_bucket_counts"] == (
        ("u1-f2", 1),
        ("u2-f4", 1),
    )
    assert loader.host_metadata["guide_document_bucket_counts"] == (
        ("u1-f2", 1),
        ("u2-f4", 1),
    )
    assert loader.host_metadata["guide_length_summary"]["documents"] == 2
    assert loader.host_metadata["guide_length_summary"]["units_max"] == 2


def test_factory_uses_split_manifest_query_scope_before_optional_debug_subset(
    tmp_path, monkeypatch
):
    setup = _make_setup(tmp_path)
    config, train_config, _, bundle, episodes, plans, tokenizer, frame_loader, _ = setup
    split_path = tmp_path / "training-split.json"
    split_path.write_text("{}", encoding="utf-8")
    config = dataclasses.replace(
        config,
        split_manifest_path=split_path,
        require_all_tasks=False,
    )
    calls = []

    def fake_validate(path, *, bundle, episode_records, require_all_tasks):
        calls.append((path, bundle, tuple(episode_records), require_all_tasks))
        return SimpleNamespace(
            split_id="split-test",
            query_episode_indices=(200,),
        )

    monkeypatch.setattr(_guide_data, "load_and_validate_training_split", fake_validate)
    loader = create_robodojo_guided_data_loader(
        train_config,
        config,
        num_batches=1,
        skip_norm_stats=True,
        dataset_factory=lambda *_args: setup[2],
        artifact_loader=lambda **_paths: bundle,
        episode_reader=lambda _root: episodes,
        tokenizer=tokenizer,
        frame_loader=frame_loader,
        plan_builder=lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        transforms_module=_TRANSFORMS,
    )

    batch = next(iter(loader))

    assert np.all(np.asarray(batch.observation.state[0, :, 0]) >= 4)
    assert loader.host_metadata["training_split_id"] == "split-test"
    assert calls == [(split_path, bundle, episodes, False)]


def test_factory_rejects_unknown_query_episode_and_bad_ranges(tmp_path):
    setup = _make_setup(tmp_path)
    config, train_config, _, bundle, episodes, plans, tokenizer, frame_loader, _ = setup
    unknown_config = dataclasses.replace(config, query_episode_indices=(999,))
    common = {
        "dataset_factory": lambda *_args: setup[2],
        "artifact_loader": lambda **_paths: bundle,
        "episode_reader": lambda _root: episodes,
        "tokenizer": tokenizer,
        "frame_loader": frame_loader,
        "plan_builder": lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        "transforms_module": _TRANSFORMS,
        "skip_norm_stats": True,
    }

    with pytest.raises(ValueError, match="not bound"):
        create_robodojo_guided_data_loader(train_config, unknown_config, **common)

    overlapping = (
        episodes[0],
        episodes[1],
        _Episode(200, 5, 4, 7, 11, episodes[2].videos),
        episodes[3],
    )
    with pytest.raises(ValueError, match="overlap"):
        create_robodojo_guided_data_loader(
            train_config,
            config,
            **{**common, "episode_reader": lambda _root: overlapping},
        )


def test_factory_rejects_process_count_workers_paths_and_missing_norm_stats(tmp_path, monkeypatch):
    setup = _make_setup(tmp_path)
    config, train_config, _, bundle, episodes, plans, tokenizer, frame_loader, _ = setup
    common = {
        "dataset_factory": lambda *_args: setup[2],
        "artifact_loader": lambda **_paths: bundle,
        "episode_reader": lambda _root: episodes,
        "tokenizer": tokenizer,
        "frame_loader": frame_loader,
        "plan_builder": lambda _bundle, *, query_episode_index, profile: plans[query_episode_index],
        "transforms_module": _TRANSFORMS,
    }

    monkeypatch.setattr(_guide_data.jax, "process_count", lambda: 2)
    with pytest.raises(ValueError, match="process_count"):
        create_robodojo_guided_data_loader(
            train_config,
            config,
            skip_norm_stats=True,
            **common,
        )

    monkeypatch.setattr(_guide_data.jax, "process_count", lambda: 1)
    worker_config = dataclasses.replace(config, num_workers=1)
    assert worker_config.num_workers == 1
    with pytest.raises(ValueError, match=r"custom.*num_workers|worker processes"):
        create_robodojo_guided_data_loader(
            train_config,
            worker_config,
            skip_norm_stats=True,
            **common,
        )

    missing_stats_train = dataclasses.replace(
        train_config,
        data=_NativeDataFactory(_DataConfig(repo_id="native-repo", norm_stats=None)),
    )
    real_config = dataclasses.replace(config, repo_id="real")
    with pytest.raises(ValueError, match=r"Normalization stats|asset_id"):
        create_robodojo_guided_data_loader(
            missing_stats_train,
            real_config,
            **common,
        )

    missing_path = dataclasses.replace(
        config,
        dataset_artifact_path=tmp_path / "missing.json",
    )
    with pytest.raises(ValueError, match="dataset_artifact_path"):
        create_robodojo_guided_data_loader(train_config, missing_path, **common)


def test_config_accepts_workers_and_rejects_invalid_parallelism_or_duplicate_query_indices(tmp_path):
    setup = _make_setup(tmp_path)
    config = setup[0]

    assert dataclasses.replace(config, num_workers=8).num_workers == 8
    with pytest.raises(ValueError, match="num_workers"):
        dataclasses.replace(config, num_workers=-1)
    with pytest.raises(ValueError, match="divisible"):
        dataclasses.replace(config, batch_size=3, guides_per_batch=2)
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(config, query_episode_indices=(100, 100))


def test_factory_range_helper_retains_half_open_episode_contract():
    records = [SimpleNamespace(episode_index=3, dataset_from_index=10, dataset_to_index=14)]

    ranges = _guide_data._make_episode_ranges(records)  # noqa: SLF001

    assert ranges == (QueryEpisodeRange(3, 10, 14),)
