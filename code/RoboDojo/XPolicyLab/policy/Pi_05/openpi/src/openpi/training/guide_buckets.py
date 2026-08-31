from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
from itertools import pairwise
import json
from numbers import Integral
from types import MappingProxyType
from typing import Any

from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training.guide_dataset import GuideCatalog


@dataclass(frozen=True, order=True, slots=True)
class GuideLengthBucket:
    max_units: int
    max_boundaries: int

    def __post_init__(self) -> None:
        for name in ("max_units", "max_boundaries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def bucket_id(self) -> str:
        return f"u{self.max_units}-b{self.max_boundaries}"


@dataclass(frozen=True, slots=True)
class GuideBucketAssignment:
    guide_to_bucket: Mapping[int, str]
    guide_to_materializer_config: Mapping[int, GuideMaterializerConfig]
    plans_by_document: Mapping[str, Any]
    effective_buckets: tuple[GuideLengthBucket, ...]
    bucket_counts: tuple[tuple[str, int], ...]
    document_lengths: Mapping[str, tuple[int, int]]
    assignment_digest: str


def parse_guide_length_bucket(spec: str) -> GuideLengthBucket:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("Guide length bucket must use MAX_UNITS:MAX_BOUNDARIES syntax")
    units, boundaries = spec.split(":")
    try:
        return GuideLengthBucket(max_units=int(units), max_boundaries=int(boundaries))
    except ValueError as exc:
        raise ValueError(f"invalid Guide length bucket {spec!r}; expected positive integers") from exc


def normalize_guide_length_buckets(
    buckets: Sequence[GuideLengthBucket],
) -> tuple[GuideLengthBucket, ...]:
    if not buckets:
        raise ValueError("at least one Guide length bucket is required")
    normalized = tuple(sorted(buckets))
    if len(set(normalized)) != len(normalized):
        raise ValueError("Guide length buckets must be unique")
    for previous, current in pairwise(normalized):
        if current.max_units < previous.max_units or current.max_boundaries < previous.max_boundaries:
            raise ValueError("Guide length buckets must grow monotonically in units and boundaries")
    return normalized


def _default_plan_builder(document_catalog: Any, *, document_id: str) -> Any:
    reader = importlib.import_module("video_harness.reader")
    return reader.build_guide_plan(document_catalog, document_id=document_id)


def _derive_auto_buckets(
    document_lengths: Mapping[str, tuple[int, int]],
    *,
    max_units: int,
    max_boundaries: int,
    minimum_bucket_guides: int,
) -> tuple[GuideLengthBucket, ...]:
    """Build bounded, near-equal document-count buckets from observed lengths."""

    for name, value in (("max_units", max_units), ("max_boundaries", max_boundaries)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    lengths = tuple(document_lengths.items())
    if len(lengths) < minimum_bucket_guides:
        raise ValueError(
            f"Guide catalog has {len(lengths)} documents, fewer than guides_per_batch={minimum_bucket_guides}"
        )
    for document_id, (unit_count, boundary_count) in lengths:
        if unit_count > max_units or boundary_count > max_boundaries:
            raise ValueError(
                f"Guide {document_id!r} has {unit_count} units/"
                f"{boundary_count} boundaries and exceeds configured hard caps "
                f"u{max_units}-b{max_boundaries}"
            )

    by_units: dict[int, list[int]] = {}
    for _, (unit_count, boundary_count) in lengths:
        by_units.setdefault(unit_count, []).append(boundary_count)

    # Every bucket is a distinct static JAX shape, so keep recompilations bounded.
    desired_groups = min(
        4,
        len(lengths) // minimum_bucket_guides,
        len(by_units),
    )
    levels = tuple(sorted(by_units.items()))
    buckets: list[GuideLengthBucket] = []
    group_count = 0
    group_max_boundaries = 0
    remaining = len(lengths)
    groups_remaining = desired_groups
    target_size = (remaining + groups_remaining - 1) // groups_remaining
    cumulative_max_boundaries = 0
    for level_index, (unit_count, boundary_counts) in enumerate(levels):
        level_count = len(boundary_counts)
        group_count += level_count
        group_max_boundaries = max(group_max_boundaries, *boundary_counts)
        remaining -= level_count

        remaining_levels = len(levels) - level_index - 1
        future_groups = min(
            groups_remaining - 1,
            remaining // minimum_bucket_guides,
            remaining_levels,
        )
        next_level_count = len(levels[level_index + 1][1]) if remaining_levels else 0
        can_close = (
            group_count >= minimum_bucket_guides
            and future_groups > 0
            and (group_count >= target_size or group_count + next_level_count > target_size)
        )
        if can_close:
            cumulative_max_boundaries = max(cumulative_max_boundaries, group_max_boundaries)
            buckets.append(GuideLengthBucket(unit_count, cumulative_max_boundaries))
            group_count = 0
            group_max_boundaries = 0
            groups_remaining = future_groups
            target_size = (remaining + groups_remaining - 1) // groups_remaining
        elif level_index == len(levels) - 1:
            cumulative_max_boundaries = max(cumulative_max_boundaries, group_max_boundaries)
            buckets.append(GuideLengthBucket(unit_count, cumulative_max_boundaries))

    return tuple(buckets)


def assign_guide_length_buckets(
    *,
    document_catalog: Any,
    guide_catalog: GuideCatalog,
    buckets: Sequence[GuideLengthBucket] | None,
    max_boundary_text_tokens: int,
    max_transition_text_tokens: int,
    boundary_num_queries: int = 8,
    transition_num_queries: int = 4,
    minimum_bucket_guides: int = 1,
    max_units: int | None = None,
    max_boundaries: int | None = None,
    plan_builder: Callable[..., Any] | None = None,
) -> GuideBucketAssignment:
    normalized = normalize_guide_length_buckets(buckets) if buckets is not None else None
    for name, value in (
        ("max_boundary_text_tokens", max_boundary_text_tokens),
        ("max_transition_text_tokens", max_transition_text_tokens),
        ("boundary_num_queries", boundary_num_queries),
        ("transition_num_queries", transition_num_queries),
        ("minimum_bucket_guides", minimum_bucket_guides),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    builder = _default_plan_builder if plan_builder is None else plan_builder

    plans_by_document: dict[str, Any] = {}
    document_lengths: dict[str, tuple[int, int]] = {}
    for record in guide_catalog.records:
        plan = builder(document_catalog, document_id=record.document_id)
        if getattr(plan, "document_id", None) != record.document_id:
            raise ValueError(f"GuidePlan document mismatch for guide_index={record.guide_index}")
        unit_count = len(plan.units)
        boundary_count = len(plan.boundaries)
        plans_by_document[record.document_id] = plan
        document_lengths[record.document_id] = (unit_count, boundary_count)

    if normalized is None:
        if max_units is None or max_boundaries is None:
            raise ValueError("automatic Guide buckets require max_units and max_boundaries")
        normalized = _derive_auto_buckets(
            document_lengths,
            max_units=max_units,
            max_boundaries=max_boundaries,
            minimum_bucket_guides=minimum_bucket_guides,
        )

    assigned_buckets: dict[int, GuideLengthBucket] = {}
    for record in guide_catalog.records:
        unit_count, boundary_count = document_lengths[record.document_id]
        bucket = next(
            (
                candidate
                for candidate in normalized
                if unit_count <= candidate.max_units and boundary_count <= candidate.max_boundaries
            ),
            None,
        )
        if bucket is None:
            raise ValueError(
                f"Guide {record.document_id!r} has {unit_count} units/"
                f"{boundary_count} boundaries and exceeds the largest bucket "
                f"{normalized[-1].bucket_id}"
            )
        assigned_buckets[record.guide_index] = bucket

    # A bucket with fewer than G documents can never form a distinct-Guide
    # batch.  Promote it wholesale so those documents keep the same sampling
    # marginal instead of being silently dropped forever.
    for bucket_index, bucket in enumerate(normalized[:-1]):
        members = [guide_index for guide_index, assigned in assigned_buckets.items() if assigned == bucket]
        if 0 < len(members) < minimum_bucket_guides:
            promoted = normalized[bucket_index + 1]
            for guide_index in members:
                assigned_buckets[guide_index] = promoted
    largest_members = sum(assigned == normalized[-1] for assigned in assigned_buckets.values())
    if 0 < largest_members < minimum_bucket_guides:
        raise ValueError(
            f"largest Guide bucket has {largest_members} documents, fewer than guides_per_batch={minimum_bucket_guides}"
        )

    guide_to_bucket: dict[int, str] = {}
    guide_to_config: dict[int, GuideMaterializerConfig] = {}
    counts: Counter[str] = Counter()
    for guide_index, bucket in assigned_buckets.items():
        guide_to_bucket[guide_index] = bucket.bucket_id
        guide_to_config[guide_index] = GuideMaterializerConfig(
            max_boundaries=bucket.max_boundaries,
            max_units=bucket.max_units,
            max_boundary_text_tokens=max_boundary_text_tokens,
            max_transition_text_tokens=max_transition_text_tokens,
            boundary_num_queries=boundary_num_queries,
            transition_num_queries=transition_num_queries,
        )
        counts[bucket.bucket_id] += 1

    assignment_payload = json.dumps(
        {
            "effective_buckets": [
                {
                    "max_units": bucket.max_units,
                    "max_boundaries": bucket.max_boundaries,
                }
                for bucket in normalized
            ],
            "guides": [
                {
                    "guide_index": guide_index,
                    "document_id": guide_catalog.by_guide_index(guide_index).document_id,
                    "bucket_id": guide_to_bucket[guide_index],
                    "max_units": guide_to_config[guide_index].max_units,
                    "max_boundaries": guide_to_config[guide_index].max_boundaries,
                    "max_boundary_text_tokens": guide_to_config[guide_index].max_boundary_text_tokens,
                    "max_transition_text_tokens": guide_to_config[guide_index].max_transition_text_tokens,
                    "boundary_num_queries": guide_to_config[guide_index].boundary_num_queries,
                    "transition_num_queries": guide_to_config[guide_index].transition_num_queries,
                    "document_length": document_lengths[guide_catalog.by_guide_index(guide_index).document_id],
                }
                for guide_index in sorted(guide_to_bucket)
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return GuideBucketAssignment(
        guide_to_bucket=MappingProxyType(guide_to_bucket),
        guide_to_materializer_config=MappingProxyType(guide_to_config),
        plans_by_document=MappingProxyType(plans_by_document),
        effective_buckets=normalized,
        bucket_counts=tuple(sorted(counts.items())),
        document_lengths=MappingProxyType(document_lengths),
        assignment_digest=hashlib.sha256(assignment_payload).hexdigest(),
    )
