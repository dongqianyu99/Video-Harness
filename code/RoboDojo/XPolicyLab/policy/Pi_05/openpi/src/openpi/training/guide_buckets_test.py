from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from openpi.training.guide_buckets import GuideLengthBucket
from openpi.training.guide_buckets import assign_guide_length_buckets
from openpi.training.guide_buckets import normalize_guide_length_buckets
from openpi.training.guide_buckets import parse_guide_length_bucket
from openpi.training.guide_dataset import GuideBindingIndex


@dataclass(frozen=True)
class _Binding:
    query_episode_index: int
    support_episode_index: int
    task_index: int
    support_document_id: str


def _index() -> GuideBindingIndex:
    return GuideBindingIndex.from_bindings(
        [
            _Binding(10, 11, 0, "small-doc"),
            _Binding(20, 21, 1, "large-doc"),
        ]
    )


def test_bucket_assignment_uses_smallest_fitting_shape():
    plans = {
        10: SimpleNamespace(
            support_document_id="small-doc",
            frames=tuple(range(5)),
            units=tuple(range(4)),
        ),
        20: SimpleNamespace(
            support_document_id="large-doc",
            frames=tuple(range(17)),
            units=tuple(range(12)),
        ),
    }
    assignment = assign_guide_length_buckets(
        artifact_bundle=object(),
        binding_index=_index(),
        buckets=(GuideLengthBucket(16, 24), GuideLengthBucket(8, 10)),
        max_text_tokens=32,
        profile="actuator",
        plan_builder=lambda _bundle, *, query_episode_index, profile: plans[
            query_episode_index
        ],
    )

    assert dict(assignment.binding_to_bucket) == {0: "u8-f10", 1: "u16-f24"}
    assert assignment.binding_to_materializer_config[0].max_units == 8
    assert assignment.binding_to_materializer_config[1].max_frames == 24
    assert assignment.bucket_counts == (("u16-f24", 1), ("u8-f10", 1))
    assert assignment.document_bucket_counts == (
        ("u16-f24", 1),
        ("u8-f10", 1),
    )


def test_bucket_assignment_fails_instead_of_truncating_oversized_guide():
    plan = SimpleNamespace(
        support_document_id="small-doc",
        frames=tuple(range(20)),
        units=tuple(range(10)),
    )
    index = GuideBindingIndex.from_bindings([_Binding(10, 11, 0, "small-doc")])

    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        assign_guide_length_buckets(
            artifact_bundle=object(),
            binding_index=index,
            buckets=(GuideLengthBucket(8, 16),),
            max_text_tokens=32,
            profile="actuator",
            plan_builder=lambda *_args, **_kwargs: plan,
        )


def test_bucket_parser_and_monotonic_validation():
    assert parse_guide_length_bucket("16:24") == GuideLengthBucket(16, 24)
    with pytest.raises(ValueError, match="MAX_UNITS:MAX_FRAMES"):
        parse_guide_length_bucket("16")
    with pytest.raises(ValueError, match="positive integers"):
        parse_guide_length_bucket("zero:24")
    with pytest.raises(ValueError, match="monotonically"):
        normalize_guide_length_buckets(
            (GuideLengthBucket(8, 32), GuideLengthBucket(16, 24))
        )
