from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from openpi.training.robodojo_guide_data import RoboDojoGuidedDataConfig, create_robodojo_guided_data_loader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read one real RoboDojo Guide-conditioned Pi0.5 batch.")
    parser.add_argument("--native-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-artifact", type=Path, required=True)
    parser.add_argument("--documents-root", type=Path, required=True)
    parser.add_argument("--pairs-artifact", type=Path, required=True)
    parser.add_argument("--query-episode-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--max-units", type=int, required=True)
    parser.add_argument("--max-text-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", default="actuator")
    parser.add_argument("--num-batches", type=int, default=1)
    return parser


def _shape(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def _valid_count(mask: Any) -> int:
    return int(np.asarray(mask, dtype=np.bool_).sum())


def _slot_summary(batch: Any) -> dict[str, int | None]:
    unit_mask = np.asarray(batch.guide.unit_mask)[0].astype(bool)
    before = np.asarray(batch.guide.before_slot)[0][unit_mask]
    after = np.asarray(batch.guide.after_slot)[0][unit_mask]
    if before.size == 0:
        return {
            "before_min": None,
            "before_max": None,
            "after_min": None,
            "after_max": None,
        }
    return {
        "before_min": int(before.min()),
        "before_max": int(before.max()),
        "after_min": int(after.min()),
        "after_max": int(after.max()),
    }


def _support_summary(
    config: RoboDojoGuidedDataConfig,
    query_episode_index: int,
) -> dict[str, Any]:
    reader = importlib.import_module("video_harness.reader")
    bundle = reader.load_guide_artifact_bundle(
        dataset_path=config.dataset_artifact_path,
        documents_path=config.documents_root,
        pairs_path=config.pairs_artifact_path,
    )
    matches = [binding for binding in bundle.support_bindings if binding.query_episode_index == query_episode_index]
    if len(matches) != 1:
        raise ValueError(
            f"expected one support binding for query_episode_index={query_episode_index}, found {len(matches)}"
        )
    binding = matches[0]
    return {
        "support_episode": binding.support_episode_index,
        "support_document": binding.support_document_id,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_module = importlib.import_module("openpi.training.config")
    native_config = config_module.get_config(args.native_config_name)
    guided_config = RoboDojoGuidedDataConfig(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        dataset_artifact_path=args.dataset_artifact,
        documents_root=args.documents_root,
        pairs_artifact_path=args.pairs_artifact,
        batch_size=args.batch_size,
        seed=args.seed,
        profile=args.profile,
        max_frames=args.max_frames,
        max_units=args.max_units,
        max_text_tokens=args.max_text_tokens,
        query_episode_indices=(args.query_episode_index,),
    )
    loader = create_robodojo_guided_data_loader(
        native_config,
        guided_config,
        num_batches=args.num_batches,
    )
    batch = next(iter(loader))

    sampler = loader.batch_sampler
    stats = [asdict(item) for item in getattr(sampler, "stats", ())]
    image_values = np.asarray(batch.guide.images)
    return {
        "native_config": args.native_config_name,
        "repo_id": args.repo_id,
        "query_episode": args.query_episode_index,
        **_support_summary(guided_config, args.query_episode_index),
        "G": int(batch.actions.shape[0]),
        "Q": int(batch.actions.shape[1]),
        "observation_image_shapes": {key: _shape(value) for key, value in batch.observation.images.items()},
        "state_shape": _shape(batch.observation.state),
        "action_shape": _shape(batch.actions),
        "guide_image_shape": _shape(batch.guide.images),
        "guide_text_shape": _shape(batch.guide.text_tokens),
        "guide_unit_shape": _shape(batch.guide.unit_mask),
        "valid_frame_count": _valid_count(batch.guide.image_mask),
        "valid_unit_count": _valid_count(batch.guide.unit_mask),
        "valid_token_count": _valid_count(batch.guide.text_mask),
        "slot_range": _slot_summary(batch),
        "image_min": float(image_values.min()),
        "image_max": float(image_values.max()),
        "sampler_stats": stats,
        "sampler_total_samples": int(getattr(sampler, "total_samples", 0)),
        "sampler_used_samples": int(getattr(sampler, "used_samples", 0)),
        "sampler_dropped_samples": int(getattr(sampler, "dropped_samples", 0)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(_run(args), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
