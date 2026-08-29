from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import numpy as np
from openpi.models.guide_inputs import query_mask_or_ones, validate_guide_conditioned_batch
from openpi.training.guide_buckets import parse_guide_length_bucket
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig, create_robodojo_guided_data_loader


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from no values")
    ordered = np.asarray(sorted(values), dtype=np.float64)
    return float(np.percentile(ordered, percentile))


def _tree_nbytes(tree: Any) -> int:
    return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(tree))


def benchmark_loader(
    loader: Any,
    *,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, Any]:
    if warmup_batches < 0 or measured_batches <= 0:
        raise ValueError("warmup_batches must be non-negative and measured_batches positive")

    iterator = iter(loader)
    for _ in range(warmup_batches):
        next(iterator)

    wait_ms: list[float] = []
    first_batch = None
    valid_queries = 0
    start_total = time.perf_counter()
    for _ in range(measured_batches):
        start = time.perf_counter()
        batch = next(iterator)
        wait_ms.append((time.perf_counter() - start) * 1000.0)
        if first_batch is None:
            first_batch = batch
        valid_queries += int(np.sum(np.asarray(query_mask_or_ones(batch))))
    elapsed = time.perf_counter() - start_total

    assert first_batch is not None
    groups, queries = validate_guide_conditioned_batch(first_batch)
    total_query_slots = measured_batches * groups * queries
    return {
        "schema_version": "openpi.guided-data-benchmark.v0",
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "groups_per_batch": groups,
        "queries_per_guide": queries,
        "queries_per_batch": groups * queries,
        "elapsed_s": elapsed,
        "batches_per_s": measured_batches / elapsed,
        "query_slots_per_s": total_query_slots / elapsed,
        "valid_queries_per_s": valid_queries / elapsed,
        "valid_queries": valid_queries,
        "padded_query_slots": total_query_slots - valid_queries,
        "data_wait_ms": {
            "mean": statistics.fmean(wait_ms),
            "p50": _percentile(wait_ms, 50),
            "p95": _percentile(wait_ms, 95),
            "max": max(wait_ms),
        },
        "batch_bytes": _tree_nbytes(first_batch),
        "guide_bytes": _tree_nbytes(first_batch.guide),
        "host_metadata": getattr(loader, "host_metadata", {}),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the real grouped Guide data path without loading Pi0.5.")
    parser.add_argument("--native-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-artifact", type=Path, required=True)
    parser.add_argument("--documents-root", type=Path, required=True)
    parser.add_argument("--pairs-artifact", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--guides-per-batch", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--guide-cache-entries", type=int, default=2)
    parser.add_argument("--guide-cache-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--remainder-strategy", choices=("drop", "pad_mask"), default="drop")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--guide-length-bucket",
        action="append",
        default=[],
        metavar="MAX_UNITS:MAX_FRAMES",
    )
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--max-units", type=int, required=True)
    parser.add_argument("--max-text-tokens", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--measured-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="actuator")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    native_config = importlib.import_module("openpi.training.config").get_config(args.native_config_name)
    config = RoboDojoGuidedDataConfig(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        dataset_artifact_path=args.dataset_artifact,
        documents_root=args.documents_root,
        pairs_artifact_path=args.pairs_artifact,
        batch_size=args.batch_size,
        guides_per_batch=args.guides_per_batch,
        seed=args.seed,
        profile=args.profile,
        max_frames=args.max_frames,
        max_units=args.max_units,
        max_text_tokens=args.max_text_tokens,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        worker_torch_threads=args.worker_torch_threads,
        guide_cache_entries=args.guide_cache_entries,
        guide_cache_max_bytes=args.guide_cache_max_bytes,
        split_manifest_path=args.split_manifest,
        require_all_tasks=True,
        guide_length_buckets=(tuple(parse_guide_length_bucket(spec) for spec in args.guide_length_bucket) or None),
        remainder_strategy=args.remainder_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    loader = create_robodojo_guided_data_loader(native_config, config)
    report = benchmark_loader(
        loader,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
    )
    report["config"] = dataclasses.asdict(config)
    text = json.dumps(report, indent=2, default=str, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
