from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
from itertools import pairwise
from numbers import Integral
from types import MappingProxyType
from typing import Any

from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.training.guide_dataset import GuideBindingIndex


@dataclass(frozen=True, order=True, slots=True)
class GuideLengthBucket:
    max_units: int
    max_frames: int

    def __post_init__(self) -> None:
        for name in ("max_units", "max_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def bucket_id(self) -> str:
        return f"u{self.max_units}-f{self.max_frames}"


@dataclass(frozen=True, slots=True)
class GuideBucketAssignment:
    binding_to_bucket: Mapping[int, str]
    binding_to_materializer_config: Mapping[int, GuideMaterializerConfig]
    bucket_counts: tuple[tuple[str, int], ...]
    document_bucket_counts: tuple[tuple[str, int], ...]
    document_lengths: Mapping[str, tuple[int, int]]


def parse_guide_length_bucket(spec: str) -> GuideLengthBucket:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError(
            "Guide length bucket must use MAX_UNITS:MAX_FRAMES syntax"
        )
    units, frames = spec.split(":")
    try:
        return GuideLengthBucket(max_units=int(units), max_frames=int(frames))
    except ValueError as exc:
        raise ValueError(
            f"invalid Guide length bucket {spec!r}; expected positive integers"
        ) from exc


def normalize_guide_length_buckets(
    buckets: Sequence[GuideLengthBucket],
) -> tuple[GuideLengthBucket, ...]:
    if not buckets:
        raise ValueError("at least one Guide length bucket is required")
    normalized = tuple(sorted(buckets))
    if len(set(normalized)) != len(normalized):
        raise ValueError("Guide length buckets must be unique")
    for previous, current in pairwise(normalized):
        if current.max_units < previous.max_units or current.max_frames < previous.max_frames:
            raise ValueError(
                "Guide length buckets must grow monotonically in units and frames"
            )
        if (
            current.max_units == previous.max_units
            and current.max_frames == previous.max_frames
        ):
            raise ValueError("Guide length buckets must grow")
    return normalized


def _default_plan_builder(bundle: Any, *, query_episode_index: int, profile: str) -> Any:
    reader = importlib.import_module("video_harness.reader")
    return reader.build_guide_plan(
        bundle,
        query_episode_index=query_episode_index,
        profile=profile,
    )


def assign_guide_length_buckets(
    *,
    artifact_bundle: Any,
    binding_index: GuideBindingIndex,
    buckets: Sequence[GuideLengthBucket],
    max_text_tokens: int,
    profile: str,
    plan_builder: Callable[..., Any] | None = None,
) -> GuideBucketAssignment:
    normalized = normalize_guide_length_buckets(buckets)
    if isinstance(max_text_tokens, bool) or not isinstance(max_text_tokens, int) or max_text_tokens <= 0:
        raise ValueError("max_text_tokens must be a positive integer")
    builder = _default_plan_builder if plan_builder is None else plan_builder

    binding_to_bucket: dict[int, str] = {}
    binding_to_config: dict[int, GuideMaterializerConfig] = {}
    document_lengths: dict[str, tuple[int, int]] = {}
    counts: Counter[str] = Counter()
    document_buckets: dict[str, str] = {}

    for record in binding_index.records:
        cached_length = document_lengths.get(record.support_document_id)
        if cached_length is None:
            plan = builder(
                artifact_bundle,
                query_episode_index=record.query_episode_index,
                profile=profile,
            )
            if plan.support_document_id != record.support_document_id:
                raise ValueError(
                    f"GuidePlan support document mismatch for binding {record.binding_index}"
                )
            frame_count = len(plan.frames)
            unit_count = len(plan.units)
            document_lengths[record.support_document_id] = (
                unit_count,
                frame_count,
            )
        else:
            unit_count, frame_count = cached_length
        bucket = next(
            (
                candidate
                for candidate in normalized
                if unit_count <= candidate.max_units
                and frame_count <= candidate.max_frames
            ),
            None,
        )
        if bucket is None:
            raise ValueError(
                f"Guide {record.support_document_id!r} has {unit_count} units/"
                f"{frame_count} frames and exceeds the largest bucket "
                f"{normalized[-1].bucket_id}"
            )
        binding_to_bucket[record.binding_index] = bucket.bucket_id
        binding_to_config[record.binding_index] = GuideMaterializerConfig(
            max_frames=bucket.max_frames,
            max_units=bucket.max_units,
            max_text_tokens=max_text_tokens,
        )
        counts[bucket.bucket_id] += 1
        previous_bucket = document_buckets.setdefault(
            record.support_document_id, bucket.bucket_id
        )
        if previous_bucket != bucket.bucket_id:
            raise ValueError(
                f"support document {record.support_document_id!r} changed buckets"
            )

    return GuideBucketAssignment(
        binding_to_bucket=MappingProxyType(binding_to_bucket),
        binding_to_materializer_config=MappingProxyType(binding_to_config),
        bucket_counts=tuple(sorted(counts.items())),
        document_bucket_counts=tuple(
            sorted(Counter(document_buckets.values()).items())
        ),
        document_lengths=MappingProxyType(document_lengths),
    )
