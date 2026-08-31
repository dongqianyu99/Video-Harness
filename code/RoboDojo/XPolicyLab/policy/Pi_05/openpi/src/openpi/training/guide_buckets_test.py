from types import SimpleNamespace

import pytest

from openpi.training.guide_buckets import GuideLengthBucket
from openpi.training.guide_buckets import assign_guide_length_buckets
from openpi.training.guide_buckets import normalize_guide_length_buckets
from openpi.training.guide_buckets import parse_guide_length_bucket
from openpi.training.guide_dataset import GuideCatalog


def _catalog():
    documents = (
        SimpleNamespace(document_id="small", source_episode_index=1, task_index=0, task_instruction="zero"),
        SimpleNamespace(document_id="large", source_episode_index=2, task_index=1, task_instruction="one"),
    )
    source = SimpleNamespace(catalog_digest="digest", documents=documents)
    return source, GuideCatalog.from_document_catalog(source)


def _catalog_with_lengths(lengths):
    documents = tuple(
        SimpleNamespace(
            document_id=f"doc-{index}",
            source_episode_index=index,
            task_index=0,
            task_instruction="task",
        )
        for index in range(len(lengths))
    )
    source = SimpleNamespace(catalog_digest="digest", documents=documents)
    plans = {
        document.document_id: SimpleNamespace(
            document_id=document.document_id,
            units=tuple(range(unit_count)),
            boundaries=tuple(range(boundary_count)),
        )
        for document, (unit_count, boundary_count) in zip(documents, lengths, strict=True)
    }
    return source, GuideCatalog.from_document_catalog(source), plans


def test_bucket_assignment_uses_smallest_fitting_shape_and_capacities():
    source, catalog = _catalog()
    plans = {
        "small": SimpleNamespace(document_id="small", boundaries=tuple(range(5)), units=tuple(range(4))),
        "large": SimpleNamespace(document_id="large", boundaries=tuple(range(17)), units=tuple(range(12))),
    }
    assignment = assign_guide_length_buckets(
        document_catalog=source,
        guide_catalog=catalog,
        buckets=(GuideLengthBucket(8, 10), GuideLengthBucket(16, 24)),
        max_boundary_text_tokens=32,
        max_transition_text_tokens=24,
        boundary_num_queries=12,
        transition_num_queries=8,
        plan_builder=lambda _catalog, *, document_id: plans[document_id],
    )

    assert dict(assignment.guide_to_bucket) == {0: "u8-b10", 1: "u16-b24"}
    small = assignment.guide_to_materializer_config[0]
    assert (small.max_units, small.max_boundaries) == (8, 10)
    assert (small.boundary_num_queries, small.transition_num_queries) == (12, 8)
    assert assignment.bucket_counts == (("u16-b24", 1), ("u8-b10", 1))
    assert assignment.effective_buckets == (
        GuideLengthBucket(8, 10),
        GuideLengthBucket(16, 24),
    )
    assert len(assignment.assignment_digest) == 64


def test_bucket_assignment_fails_instead_of_truncating():
    source, catalog = _catalog()
    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        assign_guide_length_buckets(
            document_catalog=source,
            guide_catalog=catalog,
            buckets=(GuideLengthBucket(8, 16),),
            max_boundary_text_tokens=32,
            max_transition_text_tokens=24,
            plan_builder=lambda _catalog, *, document_id: SimpleNamespace(
                document_id=document_id,
                boundaries=tuple(range(20)),
                units=tuple(range(10)),
            ),
        )


def test_bucket_assignment_deterministically_promotes_too_small_bucket():
    documents = tuple(
        SimpleNamespace(
            document_id=f"doc-{index}",
            source_episode_index=index,
            task_index=index,
            task_instruction=f"task {index}",
        )
        for index in range(3)
    )
    source = SimpleNamespace(catalog_digest="digest", documents=documents)
    catalog = GuideCatalog.from_document_catalog(source)
    plans = {
        "doc-0": SimpleNamespace(document_id="doc-0", boundaries=(0, 1), units=(0,)),
        "doc-1": SimpleNamespace(document_id="doc-1", boundaries=tuple(range(8)), units=tuple(range(6))),
        "doc-2": SimpleNamespace(document_id="doc-2", boundaries=tuple(range(8)), units=tuple(range(6))),
    }

    assignment = assign_guide_length_buckets(
        document_catalog=source,
        guide_catalog=catalog,
        buckets=(GuideLengthBucket(2, 3), GuideLengthBucket(8, 10)),
        max_boundary_text_tokens=8,
        max_transition_text_tokens=8,
        minimum_bucket_guides=2,
        plan_builder=lambda _catalog, *, document_id: plans[document_id],
    )

    assert set(assignment.guide_to_bucket.values()) == {"u8-b10"}
    assert assignment.guide_to_materializer_config[0].max_boundaries == 10


def test_bucket_parser_and_monotonic_validation():
    assert parse_guide_length_bucket("16:24") == GuideLengthBucket(16, 24)
    with pytest.raises(ValueError, match="MAX_UNITS:MAX_BOUNDARIES"):
        parse_guide_length_bucket("16")
    with pytest.raises(ValueError, match="monotonically"):
        normalize_guide_length_buckets((GuideLengthBucket(8, 32), GuideLengthBucket(16, 24)))


def test_auto_buckets_are_bounded_non_single_and_build_each_plan_once():
    source, catalog, plans = _catalog_with_lengths(tuple((units, units + 1) for units in range(1, 17)))
    calls = []

    assignment = assign_guide_length_buckets(
        document_catalog=source,
        guide_catalog=catalog,
        buckets=None,
        max_units=32,
        max_boundaries=40,
        max_boundary_text_tokens=8,
        max_transition_text_tokens=8,
        minimum_bucket_guides=2,
        plan_builder=lambda _catalog, *, document_id: calls.append(document_id) or plans[document_id],
    )

    assert assignment.effective_buckets == (
        GuideLengthBucket(4, 5),
        GuideLengthBucket(8, 9),
        GuideLengthBucket(12, 13),
        GuideLengthBucket(16, 17),
    )
    assert all(count >= 2 for _, count in assignment.bucket_counts)
    assert len(calls) == len(set(calls)) == 16
    assert assignment.plans_by_document == plans


def test_auto_buckets_merge_sparse_head_and_tail_without_splitting_units():
    source, catalog, plans = _catalog_with_lengths(((1, 2), *((2, 3),) * 3, *((10, 11),) * 3, (11, 12)))

    assignment = assign_guide_length_buckets(
        document_catalog=source,
        guide_catalog=catalog,
        buckets=None,
        max_units=16,
        max_boundaries=16,
        max_boundary_text_tokens=8,
        max_transition_text_tokens=8,
        minimum_bucket_guides=4,
        plan_builder=lambda _catalog, *, document_id: plans[document_id],
    )

    assert assignment.effective_buckets == (
        GuideLengthBucket(2, 3),
        GuideLengthBucket(11, 12),
    )
    assert assignment.bucket_counts == (("u11-b12", 4), ("u2-b3", 4))


def test_auto_buckets_close_a_valid_small_head_before_one_large_length_level():
    source, catalog, plans = _catalog_with_lengths(((1, 2), (2, 3), *((10, 11),) * 8))

    assignment = assign_guide_length_buckets(
        document_catalog=source,
        guide_catalog=catalog,
        buckets=None,
        max_units=16,
        max_boundaries=16,
        max_boundary_text_tokens=8,
        max_transition_text_tokens=8,
        minimum_bucket_guides=2,
        plan_builder=lambda _catalog, *, document_id: plans[document_id],
    )

    assert assignment.effective_buckets == (
        GuideLengthBucket(2, 3),
        GuideLengthBucket(10, 11),
    )
    assert assignment.bucket_counts == (("u10-b11", 8), ("u2-b3", 2))


def test_auto_buckets_keep_boundary_caps_monotonic():
    source, catalog, plans = _catalog_with_lengths(((1, 10), (2, 9), (3, 4), (4, 5)))

    assignment = assign_guide_length_buckets(
        document_catalog=source,
        guide_catalog=catalog,
        buckets=None,
        max_units=8,
        max_boundaries=12,
        max_boundary_text_tokens=8,
        max_transition_text_tokens=8,
        minimum_bucket_guides=2,
        plan_builder=lambda _catalog, *, document_id: plans[document_id],
    )

    assert assignment.effective_buckets == (
        GuideLengthBucket(2, 10),
        GuideLengthBucket(4, 10),
    )


def test_auto_buckets_fail_when_catalog_cannot_supply_distinct_guides():
    source, catalog, plans = _catalog_with_lengths(((1, 2), (2, 3), (3, 4)))

    with pytest.raises(ValueError, match="fewer than guides_per_batch=4"):
        assign_guide_length_buckets(
            document_catalog=source,
            guide_catalog=catalog,
            buckets=None,
            max_units=8,
            max_boundaries=8,
            max_boundary_text_tokens=8,
            max_transition_text_tokens=8,
            minimum_bucket_guides=4,
            plan_builder=lambda _catalog, *, document_id: plans[document_id],
        )


def test_auto_buckets_fail_closed_on_configured_hard_caps():
    source, catalog, plans = _catalog_with_lengths(((1, 2), (9, 10)))

    with pytest.raises(ValueError, match="exceeds configured hard caps"):
        assign_guide_length_buckets(
            document_catalog=source,
            guide_catalog=catalog,
            buckets=None,
            max_units=8,
            max_boundaries=10,
            max_boundary_text_tokens=8,
            max_transition_text_tokens=8,
            plan_builder=lambda _catalog, *, document_id: plans[document_id],
        )


def test_auto_bucket_assignment_digest_is_deterministic():
    source, catalog, plans = _catalog_with_lengths(tuple((units, units + 1) for units in range(1, 9)))
    kwargs = {
        "document_catalog": source,
        "guide_catalog": catalog,
        "buckets": None,
        "max_units": 16,
        "max_boundaries": 16,
        "max_boundary_text_tokens": 8,
        "max_transition_text_tokens": 8,
        "minimum_bucket_guides": 2,
        "plan_builder": lambda _catalog, *, document_id: plans[document_id],
    }

    first = assign_guide_length_buckets(**kwargs)
    second = assign_guide_length_buckets(**kwargs)

    assert first.effective_buckets == second.effective_buckets
    assert first.assignment_digest == second.assignment_digest
