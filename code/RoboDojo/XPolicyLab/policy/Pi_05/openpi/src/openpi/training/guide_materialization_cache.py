from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.guide_tokens import validate_materialized_guide_map
from openpi.training.guide_dataset import GuideRecord

_ARRAY_NAMES = (
    "boundary_images",
    "boundary_text_tokens",
    "boundary_text_mask",
    "transition_text_tokens",
    "transition_text_mask",
    "memory_source_kind",
    "memory_source_index",
    "memory_source_offset",
)


@dataclass(frozen=True, slots=True)
class GuideArtifactRecord:
    guide_index: int
    document_id: str
    artifact_path: Path
    metadata: Mapping[str, Any]
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class GuideMaterializationCache:
    records: tuple[GuideArtifactRecord, ...]
    stats: Mapping[str, Any]


def _prefix_count(mask: Any, *, name: str) -> int:
    value = np.asarray(mask)
    if value.ndim != 1 or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a one-dimensional bool mask")
    count = int(value.sum())
    if not np.all(value[:count]) or np.any(value[count:]):
        raise ValueError(f"{name} must contain only tail padding")
    return count


def _compact_guide(guide: GuideInput) -> tuple[dict[str, np.ndarray], int, int, int]:
    if any(
        not hasattr(value, "shape") or value.shape[0] != 1
        for value in jax.tree_util.tree_leaves(guide)
    ):
        raise ValueError("materialized GuideInput must have leading G=1")
    boundaries = _prefix_count(np.asarray(guide.boundary_mask)[0], name="boundary_mask")
    units = _prefix_count(np.asarray(guide.unit_mask)[0], name="unit_mask")
    memory = _prefix_count(np.asarray(guide.memory_mask)[0], name="memory_mask")
    return (
        {
            "boundary_images": np.asarray(guide.boundary_images)[0, :boundaries].copy(),
            "boundary_text_tokens": np.asarray(guide.boundary_text_tokens)[0, :boundaries].copy(),
            "boundary_text_mask": np.asarray(guide.boundary_text_mask)[0, :boundaries].copy(),
            "transition_text_tokens": np.asarray(guide.transition_text_tokens)[0, :units].copy(),
            "transition_text_mask": np.asarray(guide.transition_text_mask)[0, :units].copy(),
            "memory_source_kind": np.asarray(guide.memory_source_kind)[0, :memory].copy(),
            "memory_source_index": np.asarray(guide.memory_source_index)[0, :memory].copy(),
            "memory_source_offset": np.asarray(guide.memory_source_offset)[0, :memory].copy(),
        },
        boundaries,
        units,
        memory,
    )


def _artifact_path(cache_root: Path, record: GuideRecord) -> Path:
    return cache_root / f"task-{record.task_index:03d}-episode-{record.source_episode_index:07d}.npz"


def _metadata(source: Any, record: GuideRecord, plan: Any, config: GuideMaterializerConfig) -> dict[str, Any]:
    boundaries = len(plan.boundaries)
    units = len(plan.units)
    return {
        "document_id": record.document_id,
        "document_sha256": source.document_sha256,
        "boundary_count": boundaries,
        "unit_count": units,
        "memory_token_count": boundaries * config.boundary_num_queries + units * config.transition_num_queries,
        "image_size": list(config.image_size),
        "max_boundary_text_tokens": config.max_boundary_text_tokens,
        "max_transition_text_tokens": config.max_transition_text_tokens,
        "boundary_num_queries": config.boundary_num_queries,
        "transition_num_queries": config.transition_num_queries,
    }


def _write_artifact(path: Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8).copy(),
                **arrays,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_artifact(path: Path, expected_metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"metadata_json", *_ARRAY_NAMES}:
            raise ValueError(f"Guide cache has unexpected arrays: {path}")
        metadata = json.loads(bytes(payload["metadata_json"]).decode("utf-8"))
        if metadata != dict(expected_metadata):
            raise ValueError(f"Guide cache metadata is stale: {path}")
        return {name: np.array(payload[name], copy=True) for name in _ARRAY_NAMES}


def _metadata_matches(path: Path, expected_metadata: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(bytes(payload["metadata_json"]).decode("utf-8"))
        return metadata == dict(expected_metadata)
    except Exception:
        return False


def _check_array(value: np.ndarray, *, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"cached {name} must be {shape}/{dtype}, got {value.shape}/{value.dtype}")


def expand_cached_guide(arrays: Mapping[str, np.ndarray], *, config: GuideMaterializerConfig) -> GuideInput:
    boundaries = arrays["boundary_images"].shape[0]
    units = arrays["transition_text_tokens"].shape[0]
    memory = arrays["memory_source_kind"].shape[0]
    if boundaries > config.max_boundaries or units > config.max_units:
        raise ValueError("cached Guide exceeds the configured maximum shape")
    _check_array(arrays["boundary_images"], shape=(boundaries, 3, *config.image_size, 3), dtype=np.dtype(np.float32), name="boundary_images")
    _check_array(arrays["boundary_text_tokens"], shape=(boundaries, 3, config.max_boundary_text_tokens), dtype=np.dtype(np.int32), name="boundary_text_tokens")
    _check_array(arrays["boundary_text_mask"], shape=(boundaries, 3, config.max_boundary_text_tokens), dtype=np.dtype(np.bool_), name="boundary_text_mask")
    _check_array(arrays["transition_text_tokens"], shape=(units, config.max_transition_text_tokens), dtype=np.dtype(np.int32), name="transition_text_tokens")
    _check_array(arrays["transition_text_mask"], shape=(units, config.max_transition_text_tokens), dtype=np.dtype(np.bool_), name="transition_text_mask")
    for name in ("memory_source_kind", "memory_source_index", "memory_source_offset"):
        _check_array(arrays[name], shape=(memory,), dtype=np.dtype(np.int32), name=name)

    height, width = config.image_size
    boundary_images = np.full((1, config.max_boundaries, 3, height, width, 3), -1.0, dtype=np.float32)
    boundary_images[0, :boundaries] = arrays["boundary_images"]
    boundary_image_mask = np.zeros((1, config.max_boundaries, 3), dtype=np.bool_)
    boundary_image_mask[0, :boundaries] = True
    boundary_text_tokens = np.zeros((1, config.max_boundaries, 3, config.max_boundary_text_tokens), dtype=np.int32)
    boundary_text_mask = np.zeros(boundary_text_tokens.shape, dtype=np.bool_)
    boundary_text_tokens[0, :boundaries] = arrays["boundary_text_tokens"]
    boundary_text_mask[0, :boundaries] = arrays["boundary_text_mask"]
    transition_text_tokens = np.zeros((1, config.max_units, config.max_transition_text_tokens), dtype=np.int32)
    transition_text_mask = np.zeros(transition_text_tokens.shape, dtype=np.bool_)
    transition_text_tokens[0, :units] = arrays["transition_text_tokens"]
    transition_text_mask[0, :units] = arrays["transition_text_mask"]
    boundary_mask = np.zeros((1, config.max_boundaries), dtype=np.bool_)
    unit_mask = np.zeros((1, config.max_units), dtype=np.bool_)
    boundary_mask[0, :boundaries] = True
    unit_mask[0, :units] = True
    sequence = config.max_boundaries * config.boundary_num_queries + config.max_units * config.transition_num_queries
    source_kind = np.zeros((1, sequence), dtype=np.int32)
    source_index = np.zeros((1, sequence), dtype=np.int32)
    source_offset = np.zeros((1, sequence), dtype=np.int32)
    memory_mask = np.zeros((1, sequence), dtype=np.bool_)
    source_kind[0, :memory] = arrays["memory_source_kind"]
    source_index[0, :memory] = arrays["memory_source_index"]
    source_offset[0, :memory] = arrays["memory_source_offset"]
    memory_mask[0, :memory] = True
    guide = GuideInput(
        boundary_images=boundary_images,
        boundary_image_mask=boundary_image_mask,
        boundary_text_tokens=boundary_text_tokens,
        boundary_text_mask=boundary_text_mask,
        transition_text_tokens=transition_text_tokens,
        transition_text_mask=transition_text_mask,
        boundary_mask=boundary_mask,
        unit_mask=unit_mask,
        memory_source_kind=source_kind,
        memory_source_index=source_index,
        memory_source_offset=source_offset,
        memory_mask=memory_mask,
    )
    validate_materialized_guide_map(guide, boundary_num_queries=config.boundary_num_queries, transition_num_queries=config.transition_num_queries)
    return guide


def _record(record: GuideRecord, path: Path, metadata: Mapping[str, Any]) -> GuideArtifactRecord:
    size = path.stat().st_size if path.is_file() else 0
    return GuideArtifactRecord(record.guide_index, record.document_id, path, dict(metadata), size)


def ensure_guide_materialization_cache(
    *,
    cache_root: Path,
    guide_records: Sequence[GuideRecord],
    document_catalog: Any,
    plans_by_document: Mapping[str, Any],
    materializer_config: GuideMaterializerConfig,
    source_resolver: Callable[[GuideRecord], GuideInput],
) -> GuideMaterializationCache:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    built = reused = 0
    artifacts: list[GuideArtifactRecord] = []
    for record in guide_records:
        source = document_catalog.by_document_id(record.document_id)
        metadata = _metadata(source, record, plans_by_document[record.document_id], materializer_config)
        path = _artifact_path(cache_root, record)
        if _metadata_matches(path, metadata):
            reused += 1
        else:
            guide = source_resolver(record)
            arrays, boundaries, units, memory = _compact_guide(guide)
            if (boundaries, units, memory) != (metadata["boundary_count"], metadata["unit_count"], metadata["memory_token_count"]):
                raise ValueError(f"materialized Guide counts changed for {record.document_id!r}")
            _write_artifact(path, metadata, arrays)
            built += 1
        artifacts.append(_record(record, path, metadata))
    records = tuple(artifacts)
    return GuideMaterializationCache(
        records,
        {"documents": len(records), "reused": reused, "built": built, "bytes": sum(value.artifact_bytes for value in records), "elapsed_s": time.perf_counter() - started},
    )


def open_guide_materialization_cache(
    *,
    cache_root: Path,
    guide_records: Sequence[GuideRecord],
    document_catalog: Any,
    plans_by_document: Mapping[str, Any],
    materializer_config: GuideMaterializerConfig,
) -> GuideMaterializationCache:
    artifacts: list[GuideArtifactRecord] = []
    for record in guide_records:
        source = document_catalog.by_document_id(record.document_id)
        metadata = _metadata(source, record, plans_by_document[record.document_id], materializer_config)
        path = _artifact_path(Path(cache_root), record)
        artifacts.append(_record(record, path, metadata))
    records = tuple(artifacts)
    return GuideMaterializationCache(
        records,
        {"documents": len(records), "bytes": sum(value.artifact_bytes for value in records)},
    )


class CachedGuideResolver:
    def __init__(self, *, guide_records: Sequence[GuideRecord], artifact_records: Sequence[GuideArtifactRecord], materializer_config: GuideMaterializerConfig) -> None:
        self._config = materializer_config
        self._records = {record.guide_index: (record, artifact) for record, artifact in zip(guide_records, artifact_records, strict=True)}

    def __call__(self, record: GuideRecord) -> GuideInput:
        expected, artifact = self._records[record.guide_index]
        if expected != record or artifact.document_id != record.document_id:
            raise ValueError("cached Guide identity mismatch")
        return expand_cached_guide(_read_artifact(artifact.artifact_path, artifact.metadata), config=self._config)


@dataclass(frozen=True)
class CachedGuideResolverFactory:
    guide_records: tuple[GuideRecord, ...]
    artifact_records: tuple[GuideArtifactRecord, ...]
    materializer_config: GuideMaterializerConfig

    def __call__(self) -> CachedGuideResolver:
        return CachedGuideResolver(guide_records=self.guide_records, artifact_records=self.artifact_records, materializer_config=self.materializer_config)
