from __future__ import annotations

import itertools

import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.guide_inputs import GuideConditionedBatch
from openpi.models.guide_inputs import GuideInput
from scripts.benchmark_guided_data_loader import benchmark_loader


def _batch() -> GuideConditionedBatch:
    groups, queries = 2, 3
    return GuideConditionedBatch(
        observation=_model.Observation(
            images={
                key: jnp.zeros((groups, queries, 4, 4, 3), dtype=jnp.float32)
                for key in _model.IMAGE_KEYS
            },
            image_masks={
                key: jnp.ones((groups, queries), dtype=jnp.bool_)
                for key in _model.IMAGE_KEYS
            },
            state=jnp.zeros((groups, queries, 32), dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((groups, queries, 8), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((groups, queries, 8), dtype=jnp.bool_),
            token_ar_mask=jnp.zeros((groups, queries, 8), dtype=jnp.int32),
            token_loss_mask=jnp.zeros((groups, queries, 8), dtype=jnp.bool_),
        ),
        actions=jnp.zeros((groups, queries, 50, 32), dtype=jnp.float32),
        guide=GuideInput(
            boundary_images=jnp.zeros((groups, 2, 3, 4, 4, 3), dtype=jnp.float32),
            boundary_image_mask=jnp.ones((groups, 2, 3), dtype=jnp.bool_),
            boundary_text_tokens=jnp.zeros((groups, 2, 3, 8), dtype=jnp.int32),
            boundary_text_mask=jnp.ones((groups, 2, 3, 8), dtype=jnp.bool_),
            transition_text_tokens=jnp.zeros((groups, 1, 8), dtype=jnp.int32),
            transition_text_mask=jnp.ones((groups, 1, 8), dtype=jnp.bool_),
            boundary_mask=jnp.ones((groups, 2), dtype=jnp.bool_),
            unit_mask=jnp.ones((groups, 1), dtype=jnp.bool_),
            memory_source_kind=jnp.zeros((groups, 20), dtype=jnp.int32),
            memory_source_index=jnp.zeros((groups, 20), dtype=jnp.int32),
            memory_source_offset=jnp.zeros((groups, 20), dtype=jnp.int32),
            memory_mask=jnp.ones((groups, 20), dtype=jnp.bool_),
        ),
    )


class _Loader:
    def __init__(self):
        self.host_metadata = {"sampler_stats": {"num_batches": 4}}

    def __iter__(self):
        return itertools.repeat(_batch())


def test_benchmark_loader_reports_grouped_throughput_and_bytes():
    report = benchmark_loader(_Loader(), warmup_batches=1, measured_batches=3)

    assert report["groups_per_batch"] == 2
    assert report["queries_per_guide"] == 3
    assert report["queries_per_batch"] == 6
    assert report["measured_batches"] == 3
    assert report["batches_per_s"] > 0
    assert report["query_slots_per_s"] > 0
    assert report["valid_queries_per_s"] > 0
    assert report["valid_queries"] == 18
    assert report["padded_query_slots"] == 0
    assert report["batch_bytes"] > report["guide_bytes"] > 0
    assert report["host_metadata"]["sampler_stats"]["num_batches"] == 4
