from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig
from openpi.training.robodojo_guide_data import create_robodojo_guided_data_loader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read one real task-level RoboDojo Guidance batch."
    )
    parser.add_argument("--native-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--documents-root", type=Path, required=True)
    parser.add_argument("--guides-per-batch", type=int, required=True)
    parser.add_argument("--queries-per-guide", type=int, required=True)
    parser.add_argument("--max-boundaries", type=int, required=True)
    parser.add_argument("--max-units", type=int, required=True)
    parser.add_argument("--max-boundary-text-tokens", type=int, required=True)
    parser.add_argument("--max-transition-text-tokens", type=int, required=True)
    parser.add_argument("--guide-boundary-num-queries", type=int, default=8)
    parser.add_argument("--guide-transition-num-queries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    return parser


def _shape(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    native_config = importlib.import_module("openpi.training.config").get_config(
        args.native_config_name
    )
    config = RoboDojoGuidedDataConfig(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        documents_root=args.documents_root,
        guides_per_batch=args.guides_per_batch,
        queries_per_guide=args.queries_per_guide,
        seed=args.seed,
        max_boundaries=args.max_boundaries,
        max_units=args.max_units,
        max_boundary_text_tokens=args.max_boundary_text_tokens,
        max_transition_text_tokens=args.max_transition_text_tokens,
        guide_boundary_num_queries=args.guide_boundary_num_queries,
        guide_transition_num_queries=args.guide_transition_num_queries,
    )
    loader = create_robodojo_guided_data_loader(
        native_config, config, num_batches=args.num_batches
    )
    batch = next(iter(loader))
    catalog = loader.guide_catalog
    assert catalog is not None
    values = np.asarray(batch.guide.boundary_images)
    return {
        "native_config": args.native_config_name,
        "repo_id": args.repo_id,
        "catalog_digest": catalog.catalog_digest,
        "accepted_guides": len(catalog.records),
        "catalog_examples": [dataclasses.asdict(record) for record in catalog.records[:5]],
        "G": int(batch.actions.shape[0]),
        "Q": int(batch.actions.shape[1]),
        "observation_image_shapes": {
            key: _shape(value) for key, value in batch.observation.images.items()
        },
        "state_shape": _shape(batch.observation.state),
        "action_shape": _shape(batch.actions),
        "boundary_image_shape": _shape(batch.guide.boundary_images),
        "boundary_text_shape": _shape(batch.guide.boundary_text_tokens),
        "transition_text_shape": _shape(batch.guide.transition_text_tokens),
        "valid_boundaries": int(np.asarray(batch.guide.boundary_mask).sum()),
        "valid_units": int(np.asarray(batch.guide.unit_mask).sum()),
        "valid_memory_tokens": int(np.asarray(batch.guide.memory_mask).sum()),
        "image_min": float(values.min()),
        "image_max": float(values.max()),
        "sampler_stats": dataclasses.asdict(loader.batch_sampler.stats),
        "host_metadata": loader.host_metadata,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(_run(args), indent=2, sort_keys=True, default=str))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
