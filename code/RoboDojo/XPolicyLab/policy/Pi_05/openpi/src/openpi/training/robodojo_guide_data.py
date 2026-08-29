from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
import importlib
from pathlib import Path
import statistics
from typing import Any

import jax
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training.guide_buckets import GuideLengthBucket, assign_guide_length_buckets, normalize_guide_length_buckets
from openpi.training.guide_cache import ConstantResolverFactory, ProcessLocalGuideResolver
from openpi.training.guide_collator import MultiGuideBatchCollator
from openpi.training.guide_data_loader import GuidedDataLoader
from openpi.training.guide_dataset import GuideBindingIndex, GuideBoundDataset
from openpi.training.guide_native_dataset import transform_dataset_preserving_identity
from openpi.training.guide_sampler import GroupedBindingBatchSampler, QueryEpisodeRange, build_binding_to_sample_indices
from openpi.training.guide_split import load_and_validate_training_split
from openpi.training.robodojo_guide_resolver import RoboDojoGuideResolverFactory, VideoHarnessGuideResolver


@dataclass(frozen=True)
class RoboDojoGuidedDataConfig:
    """Explicit paths and fixed budgets for the guided RoboDojo data path."""

    repo_id: str
    dataset_root: Path
    dataset_artifact_path: Path
    documents_root: Path
    pairs_artifact_path: Path
    batch_size: int
    seed: int
    profile: str
    max_frames: int
    max_units: int
    max_text_tokens: int
    query_episode_indices: tuple[int, ...] | None = None
    num_workers: int = 0
    guides_per_batch: int = 1
    prefetch_factor: int = 2
    persistent_workers: bool = True
    worker_timeout_s: float = 0.0
    worker_torch_threads: int = 1
    guide_cache_entries: int = 2
    guide_cache_max_bytes: int = 256 * 1024 * 1024
    device_prefetch_size: int = 2
    split_manifest_path: Path | None = None
    require_all_tasks: bool = True
    guide_length_buckets: tuple[GuideLengthBucket, ...] | None = None
    remainder_strategy: str = "drop"
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id.strip():
            raise ValueError("repo_id must be a non-empty string")
        for name in (
            "dataset_root",
            "dataset_artifact_path",
            "documents_root",
            "pairs_artifact_path",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be an explicit pathlib.Path")
        if self.split_manifest_path is not None and not isinstance(self.split_manifest_path, Path):
            raise ValueError("split_manifest_path must be pathlib.Path or None")

        for name in (
            "batch_size",
            "max_frames",
            "max_units",
            "max_text_tokens",
            "guides_per_batch",
            "prefetch_factor",
            "device_prefetch_size",
            "worker_torch_threads",
            "gradient_accumulation_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if self.batch_size % self.guides_per_batch != 0:
            raise ValueError(
                f"batch_size must be divisible by guides_per_batch: {self.batch_size} % {self.guides_per_batch} != 0"
            )
        if not isinstance(self.persistent_workers, bool):
            raise ValueError("persistent_workers must be bool")
        if not isinstance(self.require_all_tasks, bool):
            raise ValueError("require_all_tasks must be bool")
        if self.remainder_strategy not in {"drop", "pad_mask"}:
            raise ValueError("remainder_strategy must be 'drop' or 'pad_mask'")
        if (
            isinstance(self.worker_timeout_s, bool)
            or not isinstance(self.worker_timeout_s, (int, float))
            or self.worker_timeout_s < 0
        ):
            raise ValueError("worker_timeout_s must be non-negative")
        for name in ("guide_cache_entries", "guide_cache_max_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

        if self.guide_length_buckets is not None:
            buckets = normalize_guide_length_buckets(self.guide_length_buckets)
            if buckets[-1].max_units > self.max_units or buckets[-1].max_frames > self.max_frames:
                raise ValueError("the largest Guide bucket must fit within max_units/max_frames")
            object.__setattr__(self, "guide_length_buckets", buckets)

        if self.query_episode_indices is not None:
            indices = tuple(self.query_episode_indices)
            if not indices:
                raise ValueError("query_episode_indices must not be empty")
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
                raise ValueError("query_episode_indices must contain non-negative integers")
            if len(set(indices)) != len(indices):
                raise ValueError("query_episode_indices must be unique")
            object.__setattr__(self, "query_episode_indices", indices)

    @property
    def queries_per_guide(self) -> int:
        return self.batch_size // self.guides_per_batch


def _require_path(path: Path, *, name: str, directory: bool) -> None:
    if not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    if directory and not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")
    if not directory and not path.is_file():
        raise ValueError(f"{name} must be a file: {path}")


def _replace_repo_id(data_config: Any, repo_id: str) -> Any:
    if dataclasses.is_dataclass(data_config):
        return dataclasses.replace(data_config, repo_id=repo_id)

    copied = copy.copy(data_config)
    try:
        copied.repo_id = repo_id
    except (AttributeError, dataclasses.FrozenInstanceError) as exc:
        raise ValueError("native DataConfig must support a non-mutating repo_id override") from exc
    return copied


def _load_artifact_bundle(
    config: RoboDojoGuidedDataConfig,
    artifact_loader: Any | None,
) -> Any:
    if artifact_loader is None:
        reader = importlib.import_module("video_harness.reader")
        artifact_loader = reader.load_guide_artifact_bundle
    return artifact_loader(
        dataset_path=config.dataset_artifact_path,
        documents_path=config.documents_root,
        pairs_path=config.pairs_artifact_path,
    )


def _read_episode_records(dataset_root: Path, episode_reader: Any | None) -> list[Any]:
    if episode_reader is None:
        robodojo = importlib.import_module("video_harness.robodojo")
        episode_reader = robodojo.read_episodes
    records = list(episode_reader(dataset_root))
    if not records:
        raise ValueError(f"no RoboDojo episode records found under {dataset_root}")
    return records


def _make_episode_ranges(records: list[Any]) -> tuple[QueryEpisodeRange, ...]:
    return tuple(
        QueryEpisodeRange(
            episode_index=record.episode_index,
            dataset_from_index=record.dataset_from_index,
            dataset_to_index=record.dataset_to_index,
        )
        for record in records
    )


def _record_by_episode(records: list[Any]) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for record in records:
        episode_index = int(record.episode_index)
        if episode_index in result:
            raise ValueError(f"duplicate RoboDojo episode_index={episode_index}")
        result[episode_index] = record
    return result


def _camera_path(record: Any, camera_key: str) -> str:
    videos = getattr(record, "videos", ())
    matches = [video for video in videos if video.key == camera_key]
    if len(matches) != 1:
        raise ValueError(f"episode_index={record.episode_index} must provide exactly one {camera_key} video")
    return str(matches[0].path)


def _validate_artifact_dataset_contract(bundle: Any, records: list[Any]) -> None:
    dataset = getattr(bundle, "dataset", None)
    if not isinstance(dataset, dict) and not hasattr(dataset, "get"):
        raise ValueError("GuideArtifactBundle.dataset must be a mapping")
    if dataset.get("schema_version") != "video-harness.robodojo-source":
        raise ValueError("VideoHarness dataset artifact does not use the RoboDojo schema")
    if dataset.get("fps") != 25:
        raise ValueError(f"artifact/dataset FPS mismatch: expected 25, got {dataset.get('fps')!r}")
    if dataset.get("document_camera") != "observation.images.cam_high":
        raise ValueError("artifact document_camera must be observation.images.cam_high")

    records_by_episode = _record_by_episode(records)
    documents_by_id = {source.document_id: source for source in getattr(bundle, "documents", ())}
    for binding in getattr(bundle, "support_bindings", ()):
        query_record = records_by_episode.get(binding.query_episode_index)
        support_record = records_by_episode.get(binding.support_episode_index)
        if query_record is None:
            raise ValueError(f"artifact query episode {binding.query_episode_index} is absent from dataset metadata")
        if support_record is None:
            raise ValueError(
                f"artifact support episode {binding.support_episode_index} is absent from dataset metadata"
            )
        if query_record.task_index != binding.task_index:
            raise ValueError(
                f"query episode {binding.query_episode_index} task mismatch: "
                f"dataset={query_record.task_index}, binding={binding.task_index}"
            )
        if support_record.task_index != binding.task_index:
            raise ValueError(
                f"support episode {binding.support_episode_index} task mismatch: "
                f"dataset={support_record.task_index}, binding={binding.task_index}"
            )

        source = documents_by_id.get(binding.support_document_id)
        if source is None:
            raise ValueError(f"binding support document {binding.support_document_id!r} is absent")
        if source.episode_index != support_record.episode_index:
            raise ValueError(f"support document {source.document_id!r} episode mismatch")
        if source.task_index != support_record.task_index:
            raise ValueError(f"support document {source.document_id!r} task mismatch")

        source_metadata = source.document.get("source")
        if not isinstance(source_metadata, dict) and not hasattr(source_metadata, "get"):
            raise ValueError(f"support document {source.document_id!r} has no source metadata")
        if source_metadata.get("episode_index") != support_record.episode_index:
            raise ValueError(f"support document {source.document_id!r} source episode mismatch")
        if source_metadata.get("task_index") != support_record.task_index:
            raise ValueError(f"support document {source.document_id!r} source task mismatch")
        if source_metadata.get("episode_length") != support_record.length:
            raise ValueError(f"support document {source.document_id!r} episode length mismatch")
        if source_metadata.get("fps") != dataset.get("fps"):
            raise ValueError(f"support document {source.document_id!r} FPS mismatch")
        if source_metadata.get("camera_key") != dataset.get("document_camera"):
            raise ValueError(f"support document {source.document_id!r} camera mismatch")
        expected_video_path = _camera_path(support_record, dataset["document_camera"])
        if source_metadata.get("video_path") != expected_video_path:
            raise ValueError(f"support document {source.document_id!r} video path mismatch")


def _select_bindings(bundle: Any, query_episode_indices: tuple[int, ...] | None) -> tuple[Any, ...]:
    bindings = tuple(getattr(bundle, "support_bindings", ()))
    if not bindings:
        raise ValueError("GuideArtifactBundle contains no support bindings")
    if query_episode_indices is None:
        return bindings

    selected = tuple(binding for binding in bindings if binding.query_episode_index in query_episode_indices)
    found = {binding.query_episode_index for binding in selected}
    missing = sorted(set(query_episode_indices) - found)
    if missing:
        raise ValueError(f"requested query episodes are not bound: {missing}")
    return selected


def _guide_length_summary(
    document_lengths: dict[str, tuple[int, int]] | Any,
) -> dict[str, float | int]:
    lengths = tuple(document_lengths.values())
    units = sorted(value[0] for value in lengths)
    frames = sorted(value[1] for value in lengths)

    def percentile(values: list[int], fraction: float) -> int:
        index = round((len(values) - 1) * fraction)
        return values[index]

    return {
        "documents": len(lengths),
        "units_min": min(units),
        "units_mean": statistics.fmean(units),
        "units_p50": percentile(units, 0.50),
        "units_p95": percentile(units, 0.95),
        "units_max": max(units),
        "frames_min": min(frames),
        "frames_mean": statistics.fmean(frames),
        "frames_p50": percentile(frames, 0.50),
        "frames_p95": percentile(frames, 0.95),
        "frames_max": max(frames),
    }


def _make_tokenizer(tokenizer: Any | None, max_text_tokens: int) -> Any:
    if tokenizer is not None:
        return tokenizer
    tokenizer_module = importlib.import_module("openpi.models.tokenizer")
    return tokenizer_module.PaligemmaTokenizer(max_text_tokens)


def create_robodojo_guided_data_loader(
    native_train_config: Any,
    guided_data_config: RoboDojoGuidedDataConfig,
    *,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    dataset_factory: Any | None = None,
    artifact_loader: Any | None = None,
    episode_reader: Any | None = None,
    tokenizer: Any | None = None,
    frame_loader: Any | None = None,
    plan_builder: Any | None = None,
    transforms_module: Any | None = None,
) -> GuidedDataLoader:
    """Create the guided-only RoboDojo data path without touching stock loaders."""

    if jax.process_count() != 1:
        raise ValueError("RoboDojo guided data requires jax.process_count() == 1")
    if num_batches is not None and (
        isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches < 0
    ):
        raise ValueError(f"num_batches must be a non-negative integer or None, got {num_batches!r}")
    if skip_norm_stats and guided_data_config.repo_id != "fake":
        raise ValueError("skip_norm_stats=True is allowed only for repo_id='fake' unit tests")

    _require_path(guided_data_config.dataset_root, name="dataset_root", directory=True)
    _require_path(guided_data_config.dataset_artifact_path, name="dataset_artifact_path", directory=False)
    _require_path(
        guided_data_config.documents_root,
        name="documents_root",
        directory=True,
    )
    _require_path(guided_data_config.pairs_artifact_path, name="pairs_artifact_path", directory=False)
    if guided_data_config.split_manifest_path is not None:
        _require_path(
            guided_data_config.split_manifest_path,
            name="split_manifest_path",
            directory=False,
        )

    native_data_config = native_train_config.data.create(
        native_train_config.assets_dirs,
        native_train_config.model,
    )
    guided_native_config = _replace_repo_id(native_data_config, guided_data_config.repo_id)
    if not skip_norm_stats and guided_data_config.repo_id != "fake" and guided_native_config.norm_stats is None:
        raise ValueError(
            "Normalization stats are required for the real guided dataset: "
            f"asset_id={guided_native_config.asset_id!r}, "
            f"dataset_artifact={guided_data_config.dataset_artifact_path}"
        )

    if dataset_factory is None:
        data_loader_module = importlib.import_module("openpi.training.data_loader")
        dataset_factory = data_loader_module.create_torch_dataset
    native_dataset = dataset_factory(
        guided_native_config,
        native_train_config.model.action_horizon,
        native_train_config.model,
    )
    transformed_dataset = transform_dataset_preserving_identity(
        native_dataset,
        guided_native_config,
        skip_norm_stats=skip_norm_stats,
        transforms_module=transforms_module,
    )

    bundle = _load_artifact_bundle(guided_data_config, artifact_loader)
    episode_records = _read_episode_records(guided_data_config.dataset_root, episode_reader)
    _validate_artifact_dataset_contract(bundle, episode_records)

    validated_split = None
    if guided_data_config.split_manifest_path is not None:
        validated_split = load_and_validate_training_split(
            guided_data_config.split_manifest_path,
            bundle=bundle,
            episode_records=episode_records,
            require_all_tasks=guided_data_config.require_all_tasks,
        )
        manifest_queries = set(validated_split.query_episode_indices)
        if guided_data_config.query_episode_indices is None:
            query_episode_indices = validated_split.query_episode_indices
        else:
            requested = set(guided_data_config.query_episode_indices)
            missing = sorted(requested - manifest_queries)
            if missing:
                raise ValueError(f"debug query episodes are absent from the training split: {missing}")
            query_episode_indices = guided_data_config.query_episode_indices
    else:
        query_episode_indices = guided_data_config.query_episode_indices

    selected_bindings = _select_bindings(
        bundle,
        query_episode_indices,
    )
    binding_index = GuideBindingIndex.from_bindings(selected_bindings)
    guide_buckets = (
        guided_data_config.guide_length_buckets
        if guided_data_config.guide_length_buckets is not None
        else (
            GuideLengthBucket(
                max_units=guided_data_config.max_units,
                max_frames=guided_data_config.max_frames,
            ),
        )
    )
    bucket_assignment = assign_guide_length_buckets(
        artifact_bundle=bundle,
        binding_index=binding_index,
        buckets=guide_buckets,
        max_text_tokens=guided_data_config.max_text_tokens,
        profile=guided_data_config.profile,
        plan_builder=plan_builder,
    )
    binding_to_samples = build_binding_to_sample_indices(
        _make_episode_ranges(episode_records),
        binding_index=binding_index,
    )
    bound_dataset = GuideBoundDataset(transformed_dataset, binding_index)
    sampler = GroupedBindingBatchSampler(
        binding_to_samples,
        binding_index=binding_index,
        guides_per_batch=guided_data_config.guides_per_batch,
        queries_per_guide=guided_data_config.queries_per_guide,
        seed=guided_data_config.seed,
        binding_to_bucket=bucket_assignment.binding_to_bucket,
        remainder_strategy=guided_data_config.remainder_strategy,
        batch_block_size=guided_data_config.gradient_accumulation_steps,
    )

    materializer_config = GuideMaterializerConfig(
        max_frames=guided_data_config.max_frames,
        max_units=guided_data_config.max_units,
        max_text_tokens=guided_data_config.max_text_tokens,
    )

    custom_worker_dependencies = any(
        dependency is not None for dependency in (artifact_loader, tokenizer, frame_loader, plan_builder)
    )
    if guided_data_config.num_workers > 0 and custom_worker_dependencies:
        raise ValueError(
            "custom artifact/tokenizer/frame/plan dependencies are only supported "
            "with num_workers=0; worker processes require the standard path factory"
        )

    if guided_data_config.num_workers > 0:
        resolver_factory = RoboDojoGuideResolverFactory(
            dataset_artifact_path=guided_data_config.dataset_artifact_path,
            documents_root=guided_data_config.documents_root,
            pairs_artifact_path=guided_data_config.pairs_artifact_path,
            dataset_root=guided_data_config.dataset_root,
            binding_records=binding_index.records,
            materializer_config=materializer_config,
            materializer_configs_by_binding=tuple(bucket_assignment.binding_to_materializer_config.items()),
            profile=guided_data_config.profile,
        )
    else:
        resolver = VideoHarnessGuideResolver(
            artifact_bundle=bundle,
            binding_index=binding_index,
            dataset_root=guided_data_config.dataset_root,
            tokenizer=_make_tokenizer(tokenizer, guided_data_config.max_text_tokens),
            materializer_config=materializer_config,
            materializer_configs_by_binding=(bucket_assignment.binding_to_materializer_config),
            profile=guided_data_config.profile,
            frame_loader=frame_loader,
            plan_builder=plan_builder,
        )
        resolver_factory = ConstantResolverFactory(resolver)

    cached_resolver = ProcessLocalGuideResolver(
        resolver_factory=resolver_factory,
        max_entries=guided_data_config.guide_cache_entries,
        max_bytes=guided_data_config.guide_cache_max_bytes,
    )
    collator = MultiGuideBatchCollator(
        binding_index=binding_index,
        guide_input_resolver=cached_resolver,
        guides_per_batch=guided_data_config.guides_per_batch,
        queries_per_guide=guided_data_config.queries_per_guide,
    )
    return GuidedDataLoader(
        bound_dataset,
        batch_sampler=sampler,
        collator=collator,
        num_batches=num_batches,
        num_workers=guided_data_config.num_workers,
        prefetch_factor=guided_data_config.prefetch_factor,
        persistent_workers=guided_data_config.persistent_workers,
        worker_timeout_s=guided_data_config.worker_timeout_s,
        worker_torch_threads=guided_data_config.worker_torch_threads,
        data_config=guided_native_config,
        binding_index=binding_index,
        host_metadata={
            "artifact_build_id": getattr(bundle, "build_id", None),
            "training_split_id": (None if validated_split is None else validated_split.split_id),
            "training_split_path": (
                None if guided_data_config.split_manifest_path is None else str(guided_data_config.split_manifest_path)
            ),
            "repo_id": guided_data_config.repo_id,
            "guides_per_batch": guided_data_config.guides_per_batch,
            "queries_per_guide": guided_data_config.queries_per_guide,
            "num_workers": guided_data_config.num_workers,
            "worker_torch_threads": guided_data_config.worker_torch_threads,
            "prefetch_factor": guided_data_config.prefetch_factor,
            "guide_cache_entries": guided_data_config.guide_cache_entries,
            "guide_cache_max_bytes": guided_data_config.guide_cache_max_bytes,
            "sampler_stats": dataclasses.asdict(sampler.stats),
            "guide_binding_bucket_counts": bucket_assignment.bucket_counts,
            "guide_document_bucket_counts": (bucket_assignment.document_bucket_counts),
            "guide_length_summary": _guide_length_summary(bucket_assignment.document_lengths),
            "remainder_strategy": guided_data_config.remainder_strategy,
            "gradient_accumulation_steps": (guided_data_config.gradient_accumulation_steps),
        },
    )
