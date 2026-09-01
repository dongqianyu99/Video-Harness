from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from filelock import FileLock
import jax
import numpy as np

from openpi.models.guide_inputs import GUIDE_REPRESENTATION_DIGEST
from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.guide_tokens import validate_materialized_guide_map
from openpi.training.guide_dataset import GuideRecord

_ARTIFACT_ARRAYS = (
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
    artifact_key: str
    artifact_path: Path
    content_digest: str
    file_sha256: str
    boundary_count: int
    unit_count: int
    memory_token_count: int
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class GuideMaterializationCache:
    catalog_digest: str
    materialization_digest: str
    cache_digest: str
    records: tuple[GuideArtifactRecord, ...]
    stats: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_digest(tokenizer: Any, *, name: str) -> str:
    value = getattr(tokenizer, "cache_digest", None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must expose a non-empty cache_digest")
    return value


def _plan_payload(plan: Any) -> dict[str, Any]:
    return {
        "document_id": plan.document_id,
        "source_episode_index": plan.source_episode_index,
        "task_index": plan.task_index,
        "task_instruction": plan.task_instruction,
        "boundaries": [
            {
                "boundary_id": value.boundary_id,
                "order": value.order,
                "slot": value.slot,
                "episode_frame_index": value.episode_frame_index,
                "timestamp_s": value.timestamp_s,
                "view_texts": list(value.view_texts),
            }
            for value in plan.boundaries
        ],
        "units": [
            {
                "unit_id": value.unit_id,
                "order": value.order,
                "before_slot": value.before_slot,
                "after_slot": value.after_slot,
                "transition_text": value.transition_text,
            }
            for value in plan.units
        ],
    }


def _materialization_contract(
    config: GuideMaterializerConfig,
    *,
    boundary_tokenizer_digest: str,
    transition_tokenizer_digest: str,
) -> dict[str, Any]:
    return {
        "guide_representation_digest": GUIDE_REPRESENTATION_DIGEST,
        "image_pipeline": "float32[-1,1]:resize_with_pad_torch:224x224",
        "image_size": list(config.image_size),
        "max_boundary_text_tokens": config.max_boundary_text_tokens,
        "max_transition_text_tokens": config.max_transition_text_tokens,
        "boundary_num_queries": config.boundary_num_queries,
        "transition_num_queries": config.transition_num_queries,
        "boundary_tokenizer_digest": boundary_tokenizer_digest,
        "transition_tokenizer_digest": transition_tokenizer_digest,
    }


def _index_contract(
    artifact_contract: Mapping[str, Any],
    config: GuideMaterializerConfig,
) -> dict[str, Any]:
    return {
        **artifact_contract,
        "max_boundaries": config.max_boundaries,
        "max_units": config.max_units,
    }


def _artifact_key(
    *,
    document_sha256: str,
    plan: Any,
    artifact_contract: Mapping[str, Any],
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "document_sha256": document_sha256,
                "plan_digest": _sha256_bytes(_canonical_json(_plan_payload(plan))),
                "materialization": artifact_contract,
            }
        )
    )


def _prefix_count(mask: Any, *, name: str) -> int:
    value = np.asarray(mask)
    if value.ndim != 1 or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a one-dimensional bool mask")
    count = int(value.sum())
    if not np.all(value[:count]) or np.any(value[count:]):
        raise ValueError(f"{name} must contain only tail padding")
    return count


def _compact_guide(guide: GuideInput) -> tuple[dict[str, np.ndarray], int, int, int]:
    leaves = jax.tree_util.tree_leaves(guide)
    if any(not hasattr(value, "shape") or value.shape[0] != 1 for value in leaves):
        raise ValueError("materialized GuideInput must have leading G=1")
    boundaries = _prefix_count(np.asarray(guide.boundary_mask)[0], name="boundary_mask")
    units = _prefix_count(np.asarray(guide.unit_mask)[0], name="unit_mask")
    memory = _prefix_count(np.asarray(guide.memory_mask)[0], name="memory_mask")
    if not np.all(np.asarray(guide.boundary_image_mask)[0, :boundaries]):
        raise ValueError("every cached Boundary must contain all three image views")
    arrays = {
        "boundary_images": np.asarray(guide.boundary_images)[0, :boundaries].copy(),
        "boundary_text_tokens": np.asarray(guide.boundary_text_tokens)[0, :boundaries].copy(),
        "boundary_text_mask": np.asarray(guide.boundary_text_mask)[0, :boundaries].copy(),
        "transition_text_tokens": np.asarray(guide.transition_text_tokens)[0, :units].copy(),
        "transition_text_mask": np.asarray(guide.transition_text_mask)[0, :units].copy(),
        "memory_source_kind": np.asarray(guide.memory_source_kind)[0, :memory].copy(),
        "memory_source_index": np.asarray(guide.memory_source_index)[0, :memory].copy(),
        "memory_source_offset": np.asarray(guide.memory_source_offset)[0, :memory].copy(),
    }
    return arrays, boundaries, units, memory


def _content_digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(_canonical_json(metadata))
    for name in _ARTIFACT_ARRAYS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _artifact_path(cache_root: Path, artifact_key: str) -> Path:
    return cache_root / "objects" / artifact_key[:2] / f"{artifact_key}.npz"


def _write_artifact(
    path: Path,
    *,
    metadata: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[str, str]:
    content_digest = _content_digest(metadata, arrays)
    payload = {
        "metadata_json": np.frombuffer(_canonical_json(metadata), dtype=np.uint8).copy(),
        "content_digest": np.frombuffer(content_digest.encode("ascii"), dtype=np.uint8).copy(),
        **arrays,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        file_sha256 = _sha256_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return content_digest, file_sha256


def _read_artifact(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    hash_file: bool = True,
) -> tuple[dict[str, np.ndarray], str, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected_names = {"metadata_json", "content_digest", *_ARTIFACT_ARRAYS}
            if set(payload.files) != expected_names:
                raise ValueError("artifact contains unexpected arrays")
            metadata = json.loads(bytes(payload["metadata_json"]).decode("utf-8"))
            stored_digest = bytes(payload["content_digest"]).decode("ascii")
            arrays = {name: np.array(payload[name], copy=True) for name in _ARTIFACT_ARRAYS}
    except Exception as exc:
        raise ValueError(f"cannot read Guide artifact {path}: {exc}") from exc
    if metadata != dict(expected_metadata):
        raise ValueError("artifact metadata does not match the expected Guide contract")
    actual_digest = _content_digest(metadata, arrays)
    if stored_digest != actual_digest:
        raise ValueError("artifact content digest mismatch")
    return arrays, actual_digest, _sha256_file(path) if hash_file else None


def _validate_compact_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    boundaries: int,
    units: int,
    memory: int,
    config: GuideMaterializerConfig,
) -> None:
    expected = {
        "boundary_images": ((boundaries, 3, *config.image_size, 3), np.dtype(np.float32)),
        "boundary_text_tokens": (
            (boundaries, 3, config.max_boundary_text_tokens),
            np.dtype(np.int32),
        ),
        "boundary_text_mask": (
            (boundaries, 3, config.max_boundary_text_tokens),
            np.dtype(np.bool_),
        ),
        "transition_text_tokens": (
            (units, config.max_transition_text_tokens),
            np.dtype(np.int32),
        ),
        "transition_text_mask": (
            (units, config.max_transition_text_tokens),
            np.dtype(np.bool_),
        ),
        "memory_source_kind": ((memory,), np.dtype(np.int32)),
        "memory_source_index": ((memory,), np.dtype(np.int32)),
        "memory_source_offset": ((memory,), np.dtype(np.int32)),
    }
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"cached {name} must have shape {shape} and dtype {dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
    if boundaries <= 0 or units <= 0 or memory <= 0:
        raise ValueError("cached Guide counts must be positive")
    if boundaries > config.max_boundaries or units > config.max_units:
        raise ValueError("cached Guide exceeds the configured shared maximum shape")
    if not np.isfinite(arrays["boundary_images"]).all():
        raise ValueError("cached Boundary images contain non-finite values")
    if np.any((arrays["boundary_images"] < -1.0) | (arrays["boundary_images"] > 1.0)):
        raise ValueError("cached Boundary images must remain within [-1, 1]")


def expand_cached_guide(
    arrays: Mapping[str, np.ndarray],
    *,
    config: GuideMaterializerConfig,
) -> GuideInput:
    boundaries = arrays["boundary_images"].shape[0]
    units = arrays["transition_text_tokens"].shape[0]
    memory = arrays["memory_source_kind"].shape[0]
    _validate_compact_arrays(
        arrays,
        boundaries=boundaries,
        units=units,
        memory=memory,
        config=config,
    )
    height, width = config.image_size
    boundary_images = np.full(
        (1, config.max_boundaries, 3, height, width, 3),
        -1.0,
        dtype=np.float32,
    )
    boundary_images[0, :boundaries] = arrays["boundary_images"]
    boundary_image_mask = np.zeros((1, config.max_boundaries, 3), dtype=np.bool_)
    boundary_image_mask[0, :boundaries] = True
    boundary_text_tokens = np.zeros(
        (1, config.max_boundaries, 3, config.max_boundary_text_tokens),
        dtype=np.int32,
    )
    boundary_text_mask = np.zeros(boundary_text_tokens.shape, dtype=np.bool_)
    boundary_text_tokens[0, :boundaries] = arrays["boundary_text_tokens"]
    boundary_text_mask[0, :boundaries] = arrays["boundary_text_mask"]
    transition_text_tokens = np.zeros(
        (1, config.max_units, config.max_transition_text_tokens),
        dtype=np.int32,
    )
    transition_text_mask = np.zeros(transition_text_tokens.shape, dtype=np.bool_)
    transition_text_tokens[0, :units] = arrays["transition_text_tokens"]
    transition_text_mask[0, :units] = arrays["transition_text_mask"]
    boundary_mask = np.zeros((1, config.max_boundaries), dtype=np.bool_)
    unit_mask = np.zeros((1, config.max_units), dtype=np.bool_)
    boundary_mask[0, :boundaries] = True
    unit_mask[0, :units] = True
    sequence = (
        config.max_boundaries * config.boundary_num_queries
        + config.max_units * config.transition_num_queries
    )
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
    validate_materialized_guide_map(
        guide,
        boundary_num_queries=config.boundary_num_queries,
        transition_num_queries=config.transition_num_queries,
    )
    return guide


def _artifact_metadata(
    *,
    record: GuideRecord,
    artifact_key: str,
    document_sha256: str,
    plan_digest: str,
    materialization_digest: str,
    boundaries: int,
    units: int,
    memory: int,
) -> dict[str, Any]:
    return {
        "artifact_key": artifact_key,
        "document_id": record.document_id,
        "document_sha256": document_sha256,
        "source_episode_index": record.source_episode_index,
        "task_index": record.task_index,
        "plan_digest": plan_digest,
        "materialization_digest": materialization_digest,
        "boundary_count": boundaries,
        "unit_count": units,
        "memory_token_count": memory,
    }


def _cache_digest(
    *,
    catalog_digest: str,
    materialization_digest: str,
    records: Sequence[GuideArtifactRecord],
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "catalog_digest": catalog_digest,
                "materialization_digest": materialization_digest,
                "artifacts": [
                    {
                        "guide_index": value.guide_index,
                        "document_id": value.document_id,
                        "artifact_key": value.artifact_key,
                        "content_digest": value.content_digest,
                        "boundary_count": value.boundary_count,
                        "unit_count": value.unit_count,
                        "memory_token_count": value.memory_token_count,
                    }
                    for value in records
                ],
            }
        )
    )


def _write_index(cache_root: Path, cache: GuideMaterializationCache) -> None:
    path = (
        cache_root
        / "catalogs"
        / cache.catalog_digest
        / f"{cache.materialization_digest}.json"
    )
    payload = {
        "catalog_digest": cache.catalog_digest,
        "materialization_digest": cache.materialization_digest,
        "cache_digest": cache.cache_digest,
        "source_media_validation": "cache_only",
        "records": [
            {
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in asdict(record).items()
                }
            }
            for record in cache.records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    os.replace(temporary, path)


def ensure_guide_materialization_cache(
    *,
    cache_root: Path,
    catalog_digest: str,
    guide_records: Sequence[GuideRecord],
    document_catalog: Any,
    plans_by_document: Mapping[str, Any],
    materializer_config: GuideMaterializerConfig,
    boundary_tokenizer: Any,
    transition_tokenizer: Any,
    source_resolver: Callable[[GuideRecord], GuideInput],
) -> GuideMaterializationCache:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    artifact_contract = _materialization_contract(
        materializer_config,
        boundary_tokenizer_digest=_tokenizer_digest(
            boundary_tokenizer, name="boundary_tokenizer"
        ),
        transition_tokenizer_digest=_tokenizer_digest(
            transition_tokenizer, name="transition_tokenizer"
        ),
    )
    artifact_contract_digest = _sha256_bytes(_canonical_json(artifact_contract))
    materialization_digest = _sha256_bytes(
        _canonical_json(_index_contract(artifact_contract, materializer_config))
    )
    records = tuple(guide_records)
    started = time.perf_counter()
    validation_s = 0.0
    build_s = 0.0
    built = reused = rebuilt_corrupt = 0
    artifact_records: list[GuideArtifactRecord] = []

    # ponytail: one corpus build lock is intentionally simple; split it only if
    # concurrent first-time cache builds become an observed bottleneck.
    with FileLock(str(cache_root / ".build.lock")):
        for record in records:
            source = document_catalog.by_document_id(record.document_id)
            plan = plans_by_document[record.document_id]
            plan_digest = _sha256_bytes(_canonical_json(_plan_payload(plan)))
            key = _artifact_key(
                document_sha256=source.document_sha256,
                plan=plan,
                artifact_contract=artifact_contract,
            )
            path = _artifact_path(cache_root, key)
            boundaries = len(plan.boundaries)
            units = len(plan.units)
            memory = (
                boundaries * materializer_config.boundary_num_queries
                + units * materializer_config.transition_num_queries
            )
            metadata = _artifact_metadata(
                record=record,
                artifact_key=key,
                document_sha256=source.document_sha256,
                plan_digest=plan_digest,
                materialization_digest=artifact_contract_digest,
                boundaries=boundaries,
                units=units,
                memory=memory,
            )
            arrays: dict[str, np.ndarray]
            validation_started = time.perf_counter()
            try:
                arrays, content_digest, file_sha256 = _read_artifact(
                    path, expected_metadata=metadata
                )
                _validate_compact_arrays(
                    arrays,
                    boundaries=boundaries,
                    units=units,
                    memory=memory,
                    config=materializer_config,
                )
                expand_cached_guide(arrays, config=materializer_config)
                validation_s += time.perf_counter() - validation_started
                reused += 1
            except Exception as artifact_error:
                validation_s += time.perf_counter() - validation_started
                corrupt = path.exists()
                build_started = time.perf_counter()
                guide = source_resolver(record)
                arrays, actual_boundaries, actual_units, actual_memory = _compact_guide(guide)
                if (actual_boundaries, actual_units, actual_memory) != (
                    boundaries,
                    units,
                    memory,
                ):
                    raise ValueError(
                        f"materialized Guide counts drifted for {record.document_id!r}"
                    ) from artifact_error
                _validate_compact_arrays(
                    arrays,
                    boundaries=boundaries,
                    units=units,
                    memory=memory,
                    config=materializer_config,
                )
                content_digest, file_sha256 = _write_artifact(
                    path, metadata=metadata, arrays=arrays
                )
                build_s += time.perf_counter() - build_started
                if corrupt:
                    rebuilt_corrupt += 1
                else:
                    built += 1
            artifact_records.append(
                GuideArtifactRecord(
                    guide_index=record.guide_index,
                    document_id=record.document_id,
                    artifact_key=key,
                    artifact_path=path,
                    content_digest=content_digest,
                    file_sha256=file_sha256,
                    boundary_count=boundaries,
                    unit_count=units,
                    memory_token_count=memory,
                    artifact_bytes=path.stat().st_size,
                )
            )

        cache_records = tuple(artifact_records)
        cache = GuideMaterializationCache(
            catalog_digest=catalog_digest,
            materialization_digest=materialization_digest,
            cache_digest=_cache_digest(
                catalog_digest=catalog_digest,
                materialization_digest=materialization_digest,
                records=cache_records,
            ),
            records=cache_records,
            stats={
                "documents": len(cache_records),
                "reused": reused,
                "built": built,
                "rebuilt_corrupt": rebuilt_corrupt,
                "bytes": sum(value.artifact_bytes for value in cache_records),
                "boundaries": sum(value.boundary_count for value in cache_records),
                "camera_frames": 3 * sum(
                    value.boundary_count for value in cache_records
                ),
                "validation_s": validation_s,
                "build_s": build_s,
                "elapsed_s": time.perf_counter() - started,
                "source_media_validation": "cache_only",
            },
        )
        _write_index(cache_root, cache)
        return cache


def open_guide_materialization_cache(
    *,
    cache_root: Path,
    catalog_digest: str,
    guide_records: Sequence[GuideRecord],
    document_catalog: Any,
    plans_by_document: Mapping[str, Any],
    materializer_config: GuideMaterializerConfig,
    boundary_tokenizer: Any,
    transition_tokenizer: Any,
) -> GuideMaterializationCache:
    """Open and fully validate an existing cache without touching source media."""

    cache_root = Path(cache_root)
    artifact_contract = _materialization_contract(
        materializer_config,
        boundary_tokenizer_digest=_tokenizer_digest(
            boundary_tokenizer, name="boundary_tokenizer"
        ),
        transition_tokenizer_digest=_tokenizer_digest(
            transition_tokenizer, name="transition_tokenizer"
        ),
    )
    artifact_contract_digest = _sha256_bytes(_canonical_json(artifact_contract))
    materialization_digest = _sha256_bytes(
        _canonical_json(_index_contract(artifact_contract, materializer_config))
    )
    started = time.perf_counter()
    artifact_records: list[GuideArtifactRecord] = []
    for record in tuple(guide_records):
        source = document_catalog.by_document_id(record.document_id)
        plan = plans_by_document[record.document_id]
        plan_digest = _sha256_bytes(_canonical_json(_plan_payload(plan)))
        key = _artifact_key(
            document_sha256=source.document_sha256,
            plan=plan,
            artifact_contract=artifact_contract,
        )
        path = _artifact_path(cache_root, key)
        boundaries = len(plan.boundaries)
        units = len(plan.units)
        memory = (
            boundaries * materializer_config.boundary_num_queries
            + units * materializer_config.transition_num_queries
        )
        metadata = _artifact_metadata(
            record=record,
            artifact_key=key,
            document_sha256=source.document_sha256,
            plan_digest=plan_digest,
            materialization_digest=artifact_contract_digest,
            boundaries=boundaries,
            units=units,
            memory=memory,
        )
        arrays, content_digest, file_sha256 = _read_artifact(
            path, expected_metadata=metadata
        )
        if file_sha256 is None:
            raise ValueError(f"Guide artifact file digest is missing: {path}")
        _validate_compact_arrays(
            arrays,
            boundaries=boundaries,
            units=units,
            memory=memory,
            config=materializer_config,
        )
        expand_cached_guide(arrays, config=materializer_config)
        artifact_records.append(
            GuideArtifactRecord(
                guide_index=record.guide_index,
                document_id=record.document_id,
                artifact_key=key,
                artifact_path=path,
                content_digest=content_digest,
                file_sha256=file_sha256,
                boundary_count=boundaries,
                unit_count=units,
                memory_token_count=memory,
                artifact_bytes=path.stat().st_size,
            )
        )
    records = tuple(artifact_records)
    cache = GuideMaterializationCache(
        catalog_digest=catalog_digest,
        materialization_digest=materialization_digest,
        cache_digest=_cache_digest(
            catalog_digest=catalog_digest,
            materialization_digest=materialization_digest,
            records=records,
        ),
        records=records,
        stats={
            "documents": len(records),
            "reused": len(records),
            "built": 0,
            "rebuilt_corrupt": 0,
            "bytes": sum(value.artifact_bytes for value in records),
            "boundaries": sum(value.boundary_count for value in records),
            "camera_frames": 3 * sum(value.boundary_count for value in records),
            "validation_s": time.perf_counter() - started,
            "build_s": 0.0,
            "elapsed_s": time.perf_counter() - started,
            "source_media_validation": "cache_only",
        },
    )
    index_path = (
        cache_root
        / "catalogs"
        / catalog_digest
        / f"{materialization_digest}.json"
    )
    try:
        published = json.loads(index_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Guide cache index is missing or invalid: {index_path}") from exc
    if (
        published.get("catalog_digest") != catalog_digest
        or published.get("materialization_digest") != materialization_digest
        or published.get("cache_digest") != cache.cache_digest
        or published.get("source_media_validation") != "cache_only"
    ):
        raise ValueError("Guide cache index does not match the validated artifacts")
    return cache


class CachedGuideResolver:
    def __init__(
        self,
        *,
        guide_records: Sequence[GuideRecord],
        artifact_records: Sequence[GuideArtifactRecord],
        materializer_config: GuideMaterializerConfig,
    ) -> None:
        self._config = materializer_config
        self._records = {
            record.guide_index: (record, artifact)
            for record, artifact in zip(
                guide_records, artifact_records, strict=True
            )
        }
        if len(self._records) != len(tuple(guide_records)):
            raise ValueError("cached Guide records must have unique guide indices")

    def __call__(self, record: GuideRecord) -> GuideInput:
        try:
            expected, artifact = self._records[record.guide_index]
        except KeyError as exc:
            raise ValueError(f"unknown cached guide_index={record.guide_index}") from exc
        if expected != record or artifact.document_id != record.document_id:
            raise ValueError("cached Guide identity mismatch")
        metadata = None
        try:
            with np.load(artifact.artifact_path, allow_pickle=False) as payload:
                metadata = json.loads(bytes(payload["metadata_json"]).decode("utf-8"))
            arrays, content_digest, _ = _read_artifact(
                artifact.artifact_path,
                expected_metadata=metadata,
                hash_file=False,
            )
        except Exception as exc:
            raise ValueError(
                f"cannot load cached Guide {record.document_id!r}: {exc}"
            ) from exc
        if content_digest != artifact.content_digest:
            raise ValueError(f"cached Guide {record.document_id!r} changed after preflight")
        return expand_cached_guide(arrays, config=self._config)


@dataclass(frozen=True)
class CachedGuideResolverFactory:
    guide_records: tuple[GuideRecord, ...]
    artifact_records: tuple[GuideArtifactRecord, ...]
    materializer_config: GuideMaterializerConfig

    def __post_init__(self) -> None:
        if tuple(value.document_id for value in self.guide_records) != tuple(
            value.document_id for value in self.artifact_records
        ):
            raise ValueError("Guide records and cached artifacts must align")

    def __call__(self) -> CachedGuideResolver:
        return CachedGuideResolver(
            guide_records=self.guide_records,
            artifact_records=self.artifact_records,
            materializer_config=self.materializer_config,
        )
