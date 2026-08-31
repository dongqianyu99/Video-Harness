from __future__ import annotations

from collections.abc import Mapping
import copy
import dataclasses
from dataclasses import dataclass
import importlib
from pathlib import Path
import statistics
from typing import Any

import jax

from openpi.models.guide_inputs import GUIDE_REPRESENTATION_DIGEST
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training.guide_buckets import GuideLengthBucket
from openpi.training.guide_buckets import assign_guide_length_buckets
from openpi.training.guide_buckets import normalize_guide_length_buckets
from openpi.training.guide_cache import ConstantResolverFactory
from openpi.training.guide_cache import ProcessLocalGuideResolver
from openpi.training.guide_collator import GuidanceBatchCollator
from openpi.training.guide_data_loader import GuidedDataLoader
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_dataset import GuidedDataset
from openpi.training.guide_dataset import TaskSampleIndex
from openpi.training.guide_native_dataset import transform_dataset_preserving_identity
from openpi.training.guide_sampler import GuidanceFirstBatchSampler
from openpi.training.robodojo_guide_resolver import GuideDocumentSnapshot
from openpi.training.robodojo_guide_resolver import RoboDojoGuideResolverFactory
from openpi.training.robodojo_guide_resolver import VideoHarnessGuideResolver
from openpi.training.robodojo_guide_resolver import preflight_guide_media

_ROBOTDOJO_FPS = 25
_VIEW_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}


@dataclass(frozen=True)
class RoboDojoGuidedDataConfig:
    """Task-level Guidance Pool and fixed materialization budgets."""

    repo_id: str
    dataset_root: Path
    documents_root: Path
    guides_per_batch: int
    queries_per_guide: int
    seed: int
    max_boundaries: int
    max_units: int
    max_boundary_text_tokens: int
    max_transition_text_tokens: int
    guide_boundary_num_queries: int = 8
    guide_transition_num_queries: int = 4
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    worker_timeout_s: float = 0.0
    worker_torch_threads: int = 1
    guide_cache_entries: int = 2
    guide_cache_max_bytes: int = 256 * 1024 * 1024
    device_prefetch_size: int = 2
    require_all_tasks: bool = True
    guide_length_buckets: tuple[GuideLengthBucket, ...] | None = None
    remainder_strategy: str = "drop"
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id.strip():
            raise ValueError("repo_id must be a non-empty string")
        for name in ("dataset_root", "documents_root"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be an explicit pathlib.Path")
        for name in (
            "guides_per_batch",
            "queries_per_guide",
            "max_boundaries",
            "max_units",
            "max_boundary_text_tokens",
            "max_transition_text_tokens",
            "guide_boundary_num_queries",
            "guide_transition_num_queries",
            "prefetch_factor",
            "device_prefetch_size",
            "worker_torch_threads",
            "gradient_accumulation_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
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
                raise ValueError(f"{name} must be a non-negative integer")
        if self.guide_length_buckets is not None:
            buckets = normalize_guide_length_buckets(self.guide_length_buckets)
            if (
                buckets[-1].max_units > self.max_units
                or buckets[-1].max_boundaries > self.max_boundaries
            ):
                raise ValueError(
                    "largest Guide bucket must fit max_units/max_boundaries"
                )
            object.__setattr__(self, "guide_length_buckets", buckets)

    @property
    def batch_size(self) -> int:
        return self.guides_per_batch * self.queries_per_guide


def _require_path(path: Path, *, name: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing directory: {path}")


def _replace_repo_id(data_config: Any, repo_id: str) -> Any:
    if dataclasses.is_dataclass(data_config):
        return dataclasses.replace(data_config, repo_id=repo_id)
    copied = copy.copy(data_config)
    try:
        copied.repo_id = repo_id
    except (AttributeError, dataclasses.FrozenInstanceError) as exc:
        raise ValueError(
            "native DataConfig must support a non-mutating repo_id override"
        ) from exc
    return copied


def _load_document_catalog(documents_root: Path, catalog_loader: Any | None) -> Any:
    if catalog_loader is None:
        reader = importlib.import_module("video_harness.reader")
        catalog_loader = reader.load_guide_document_catalog
    return catalog_loader(documents_root)


def _read_episode_records(dataset_root: Path, episode_reader: Any | None) -> list[Any]:
    if episode_reader is None:
        robodojo = importlib.import_module("video_harness.robodojo")
        episode_reader = robodojo.read_episodes
    records = list(episode_reader(dataset_root))
    if not records:
        raise ValueError(f"no RoboDojo episode records found under {dataset_root}")
    return records


def _validate_catalog_dataset_contract(
    document_catalog: Any,
    guide_catalog: GuideCatalog,
    episode_records: list[Any],
    *,
    require_all_tasks: bool,
) -> None:
    by_episode: dict[int, Any] = {}
    dataset_tasks: set[int] = set()
    for episode in episode_records:
        episode_index = int(episode.episode_index)
        if episode_index in by_episode:
            raise ValueError(f"duplicate RoboDojo episode_index={episode_index}")
        by_episode[episode_index] = episode
        dataset_tasks.add(int(episode.task_index))

    for record in guide_catalog.records:
        episode = by_episode.get(record.source_episode_index)
        if episode is None:
            raise ValueError(
                f"Guide {record.document_id!r} source episode is absent from dataset"
            )
        if int(episode.task_index) != record.task_index:
            raise ValueError(f"Guide {record.document_id!r} source task mismatch")
        if str(episode.task_instruction) != record.task_instruction:
            raise ValueError(
                f"Guide {record.document_id!r} task instruction mismatch"
            )
        source_document = document_catalog.by_document_id(record.document_id)
        document = getattr(source_document, "document", None)
        if not isinstance(document, Mapping):
            raise ValueError(f"Guide {record.document_id!r} has no document mapping")
        source = document.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"Guide {record.document_id!r} has no source mapping")
        expected_scalars = {
            "episode_index": int(episode.episode_index),
            "episode_length": int(episode.length),
            "task_index": int(episode.task_index),
            "dataset_from_index": int(episode.dataset_from_index),
            "dataset_to_index": int(episode.dataset_to_index),
            "fps": _ROBOTDOJO_FPS,
            "data_path": str(episode.data_path),
        }
        for field, expected in expected_scalars.items():
            if source.get(field) != expected:
                raise ValueError(
                    f"Guide {record.document_id!r} source {field} mismatch: "
                    f"expected {expected!r}, got {source.get(field)!r}"
                )
        views = source.get("views")
        if not isinstance(views, Mapping) or set(views) != set(_VIEW_KEYS):
            raise ValueError(
                f"Guide {record.document_id!r} must provide exactly three source views"
            )
        videos = tuple(getattr(episode, "videos", ()))
        for alias, camera_key in _VIEW_KEYS.items():
            matches = [video for video in videos if video.key == camera_key]
            if len(matches) != 1:
                raise ValueError(
                    f"episode_index={episode.episode_index} must provide exactly one {camera_key} video"
                )
            expected_video = matches[0]
            view = views.get(alias)
            if not isinstance(view, Mapping):
                raise ValueError(
                    f"Guide {record.document_id!r} source view {alias} is invalid"
                )
            expected_view = {
                "camera_key": camera_key,
                "video_path": str(expected_video.path),
                "video_from_timestamp": expected_video.from_timestamp,
            }
            for field, expected in expected_view.items():
                if view.get(field) != expected:
                    raise ValueError(
                        f"Guide {record.document_id!r} source view {alias} "
                        f"{field} mismatch"
                    )
        high = views["cam_high"]
        if (
            source.get("camera_key") != _VIEW_KEYS["cam_high"]
            or source.get("video_path") != high["video_path"]
            or source.get("video_from_timestamp")
            != high["video_from_timestamp"]
        ):
            raise ValueError(
                f"Guide {record.document_id!r} top-level camera source mismatch"
            )

    guide_tasks = set(guide_catalog.task_indices)
    unknown_tasks = sorted(guide_tasks - dataset_tasks)
    if unknown_tasks:
        raise ValueError(f"Guide catalog contains unknown tasks: {unknown_tasks}")
    if require_all_tasks and guide_tasks != dataset_tasks:
        raise ValueError(
            "every RoboDojo task must have accepted Guidance documents: "
            f"missing={sorted(dataset_tasks - guide_tasks)}"
        )


def _guide_length_summary(document_lengths: Any) -> dict[str, float | int]:
    lengths = tuple(document_lengths.values())
    units = sorted(value[0] for value in lengths)
    boundaries = sorted(value[1] for value in lengths)

    def percentile(values: list[int], fraction: float) -> int:
        return values[round((len(values) - 1) * fraction)]

    return {
        "documents": len(lengths),
        "units_min": min(units),
        "units_mean": statistics.fmean(units),
        "units_p50": percentile(units, 0.50),
        "units_p95": percentile(units, 0.95),
        "units_max": max(units),
        "boundaries_min": min(boundaries),
        "boundaries_mean": statistics.fmean(boundaries),
        "boundaries_p50": percentile(boundaries, 0.50),
        "boundaries_p95": percentile(boundaries, 0.95),
        "boundaries_max": max(boundaries),
    }


def _make_tokenizer(tokenizer: Any | None, max_tokens: int) -> Any:
    if tokenizer is not None:
        return tokenizer
    module = importlib.import_module("openpi.models.tokenizer")
    return module.PaligemmaTokenizer(max_tokens)


def create_robodojo_guided_data_loader(
    native_train_config: Any,
    guided_data_config: RoboDojoGuidedDataConfig,
    *,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    dataset_factory: Any | None = None,
    catalog_loader: Any | None = None,
    episode_reader: Any | None = None,
    boundary_tokenizer: Any | None = None,
    transition_tokenizer: Any | None = None,
    frame_loader: Any | None = None,
    plan_builder: Any | None = None,
    transforms_module: Any | None = None,
) -> GuidedDataLoader:
    """Create the document-first task-level RoboDojo Guidance data path."""

    if jax.process_count() != 1:
        raise ValueError("RoboDojo guided data requires jax.process_count() == 1")
    if num_batches is not None and (
        isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches < 0
    ):
        raise ValueError("num_batches must be a non-negative integer or None")
    if skip_norm_stats and guided_data_config.repo_id != "fake":
        raise ValueError("skip_norm_stats=True is allowed only for repo_id='fake'")
    _require_path(guided_data_config.dataset_root, name="dataset_root")
    _require_path(guided_data_config.documents_root, name="documents_root")
    custom_dependencies = any(
        dependency is not None
        for dependency in (
            catalog_loader,
            boundary_tokenizer,
            transition_tokenizer,
            frame_loader,
            plan_builder,
        )
    )
    if guided_data_config.num_workers > 0 and custom_dependencies:
        raise ValueError(
            "custom catalog/tokenizer/frame/plan dependencies require num_workers=0"
        )

    native_data_config = native_train_config.data.create(
        native_train_config.assets_dirs, native_train_config.model
    )
    guided_native_config = _replace_repo_id(
        native_data_config, guided_data_config.repo_id
    )
    if (
        not skip_norm_stats
        and guided_data_config.repo_id != "fake"
        and guided_native_config.norm_stats is None
    ):
        raise ValueError("normalization stats are required for real guided data")

    if dataset_factory is None:
        module = importlib.import_module("openpi.training.data_loader")
        dataset_factory = module.create_torch_dataset
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

    document_catalog = _load_document_catalog(
        guided_data_config.documents_root, catalog_loader
    )
    guide_catalog = GuideCatalog.from_document_catalog(document_catalog)
    episode_records = _read_episode_records(
        guided_data_config.dataset_root, episode_reader
    )
    _validate_catalog_dataset_contract(
        document_catalog,
        guide_catalog,
        episode_records,
        require_all_tasks=guided_data_config.require_all_tasks,
    )
    task_samples = TaskSampleIndex.from_episode_records(
        episode_records, dataset_length=len(transformed_dataset)
    )

    assignment = assign_guide_length_buckets(
        document_catalog=document_catalog,
        guide_catalog=guide_catalog,
        buckets=guided_data_config.guide_length_buckets,
        max_units=guided_data_config.max_units,
        max_boundaries=guided_data_config.max_boundaries,
        max_boundary_text_tokens=guided_data_config.max_boundary_text_tokens,
        max_transition_text_tokens=guided_data_config.max_transition_text_tokens,
        boundary_num_queries=guided_data_config.guide_boundary_num_queries,
        transition_num_queries=guided_data_config.guide_transition_num_queries,
        minimum_bucket_guides=guided_data_config.guides_per_batch,
        plan_builder=plan_builder,
    )
    media_preflight = preflight_guide_media(
        document_catalog=document_catalog,
        guide_records=guide_catalog.records,
        plans_by_document=assignment.plans_by_document,
        dataset_root=guided_data_config.dataset_root,
        frame_loader=frame_loader,
    )
    sampler = GuidanceFirstBatchSampler(
        guide_catalog=guide_catalog,
        task_sample_index=task_samples,
        guides_per_batch=guided_data_config.guides_per_batch,
        queries_per_guide=guided_data_config.queries_per_guide,
        seed=guided_data_config.seed,
        guide_to_bucket=assignment.guide_to_bucket,
        remainder_strategy=guided_data_config.remainder_strategy,
        batch_block_size=guided_data_config.gradient_accumulation_steps,
    )
    dataset = GuidedDataset(transformed_dataset, guide_catalog, task_samples)
    materializer_config = GuideMaterializerConfig(
        max_boundaries=guided_data_config.max_boundaries,
        max_units=guided_data_config.max_units,
        max_boundary_text_tokens=guided_data_config.max_boundary_text_tokens,
        max_transition_text_tokens=guided_data_config.max_transition_text_tokens,
        boundary_num_queries=guided_data_config.guide_boundary_num_queries,
        transition_num_queries=guided_data_config.guide_transition_num_queries,
    )

    if guided_data_config.num_workers > 0:
        document_snapshots = tuple(
            GuideDocumentSnapshot.from_guide_document(
                document_catalog.by_document_id(record.document_id)
            )
            for record in guide_catalog.records
        )
        guide_plans = tuple(
            assignment.plans_by_document[record.document_id]
            for record in guide_catalog.records
        )
        resolver_factory = RoboDojoGuideResolverFactory(
            dataset_root=guided_data_config.dataset_root,
            guide_records=guide_catalog.records,
            document_snapshots=document_snapshots,
            guide_plans=guide_plans,
            materializer_config=materializer_config,
            materializer_configs_by_guide=dict(
                assignment.guide_to_materializer_config
            ),
        )
    else:
        resolver = VideoHarnessGuideResolver(
            document_catalog=document_catalog,
            guide_records=guide_catalog.records,
            dataset_root=guided_data_config.dataset_root,
            boundary_tokenizer=_make_tokenizer(
                boundary_tokenizer,
                guided_data_config.max_boundary_text_tokens,
            ),
            transition_tokenizer=_make_tokenizer(
                transition_tokenizer,
                guided_data_config.max_transition_text_tokens,
            ),
            materializer_config=materializer_config,
            materializer_configs_by_guide=assignment.guide_to_materializer_config,
            plans_by_document=assignment.plans_by_document,
            frame_loader=frame_loader,
        )
        resolver_factory = ConstantResolverFactory(resolver)

    cached_resolver = ProcessLocalGuideResolver(
        resolver_factory=resolver_factory,
        max_entries=guided_data_config.guide_cache_entries,
        max_bytes=guided_data_config.guide_cache_max_bytes,
    )
    collator = GuidanceBatchCollator(
        guide_catalog=guide_catalog,
        guide_input_resolver=cached_resolver,
        guides_per_batch=guided_data_config.guides_per_batch,
        queries_per_guide=guided_data_config.queries_per_guide,
    )
    exclusions = tuple(getattr(document_catalog, "exclusions", ()))
    return GuidedDataLoader(
        dataset,
        batch_sampler=sampler,
        collator=collator,
        num_batches=num_batches,
        num_workers=guided_data_config.num_workers,
        prefetch_factor=guided_data_config.prefetch_factor,
        persistent_workers=guided_data_config.persistent_workers,
        worker_timeout_s=guided_data_config.worker_timeout_s,
        worker_torch_threads=guided_data_config.worker_torch_threads,
        data_config=guided_native_config,
        guide_catalog=guide_catalog,
        host_metadata={
            "catalog_build_id": getattr(document_catalog, "build_id", None),
            "catalog_digest": guide_catalog.catalog_digest,
            "task_sample_digest": task_samples.digest,
            "guide_bucket_digest": assignment.assignment_digest,
            "guide_representation_digest": GUIDE_REPRESENTATION_DIGEST,
            "accepted_guides": len(guide_catalog.records),
            "excluded_guides": len(exclusions),
            "tasks": len(guide_catalog.task_indices),
            "repo_id": guided_data_config.repo_id,
            "guides_per_batch": guided_data_config.guides_per_batch,
            "queries_per_guide": guided_data_config.queries_per_guide,
            "num_workers": guided_data_config.num_workers,
            "worker_torch_threads": guided_data_config.worker_torch_threads,
            "prefetch_factor": guided_data_config.prefetch_factor,
            "guide_cache_entries": guided_data_config.guide_cache_entries,
            "guide_cache_max_bytes": guided_data_config.guide_cache_max_bytes,
            "guide_media_preflight": media_preflight,
            "sampler_stats": dataclasses.asdict(sampler.stats),
            "guide_length_buckets": tuple(
                bucket.bucket_id for bucket in assignment.effective_buckets
            ),
            "guide_bucket_counts": assignment.bucket_counts,
            "guide_length_summary": _guide_length_summary(
                assignment.document_lengths
            ),
            "guide_boundary_num_queries": (
                guided_data_config.guide_boundary_num_queries
            ),
            "guide_transition_num_queries": (
                guided_data_config.guide_transition_num_queries
            ),
            "remainder_strategy": guided_data_config.remainder_strategy,
            "gradient_accumulation_steps": (
                guided_data_config.gradient_accumulation_steps
            ),
        },
    )
