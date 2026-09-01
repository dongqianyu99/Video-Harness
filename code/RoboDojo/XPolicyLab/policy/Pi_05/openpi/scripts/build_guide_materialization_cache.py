from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import robodojo_defaults as _defaults
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_materialization_cache import ensure_guide_materialization_cache
from openpi.training.robodojo_guide_data import _build_guide_plans
from openpi.training.robodojo_guide_data import _validate_catalog_dataset_contract
from openpi.training.robodojo_guide_resolver import VideoHarnessGuideResolver


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or fully validate the persistent RoboDojo GuideInput cache."
    )
    parser.add_argument("--dataset-root", type=Path, default=_defaults.ROBODOJO_DATASET_ROOT)
    parser.add_argument("--documents-root", type=Path, default=_defaults.GUIDE_DOCUMENTS_ROOT)
    parser.add_argument(
        "--guide-materialization-cache-root",
        type=Path,
        default=_defaults.GUIDE_MATERIALIZATION_CACHE_ROOT,
    )
    parser.add_argument("--max-boundaries", type=int, default=_defaults.MAX_BOUNDARIES)
    parser.add_argument("--max-units", type=int, default=_defaults.MAX_UNITS)
    parser.add_argument(
        "--max-boundary-text-tokens",
        type=int,
        default=_defaults.MAX_BOUNDARY_TEXT_TOKENS,
    )
    parser.add_argument(
        "--max-transition-text-tokens",
        type=int,
        default=_defaults.MAX_TRANSITION_TEXT_TOKENS,
    )
    parser.add_argument("--guide-boundary-num-queries", type=int, default=8)
    parser.add_argument("--guide-transition-num-queries", type=int, default=4)
    return parser


def build_cache(args: argparse.Namespace):
    cache_root = args.guide_materialization_cache_root.resolve()
    for name in ("dataset_root", "documents_root"):
        source_root = getattr(args, name).resolve()
        if (
            cache_root == source_root
            or cache_root in source_root.parents
            or source_root in cache_root.parents
        ):
            raise ValueError(
                f"guide-materialization-cache-root must be disjoint from {name}"
            )
    reader = importlib.import_module("video_harness.reader")
    robodojo = importlib.import_module("video_harness.robodojo")
    document_catalog = reader.load_guide_document_catalog(args.documents_root)
    guide_catalog = GuideCatalog.from_document_catalog(document_catalog)
    episodes = list(robodojo.read_episodes(args.dataset_root))
    _validate_catalog_dataset_contract(
        document_catalog,
        guide_catalog,
        episodes,
        require_all_tasks=True,
    )
    plans, lengths = _build_guide_plans(
        document_catalog,
        guide_catalog,
        max_units=args.max_units,
        max_boundaries=args.max_boundaries,
        plan_builder=reader.build_guide_plan,
    )
    config = GuideMaterializerConfig(
        max_boundaries=args.max_boundaries,
        max_units=args.max_units,
        max_boundary_text_tokens=args.max_boundary_text_tokens,
        max_transition_text_tokens=args.max_transition_text_tokens,
        boundary_num_queries=args.guide_boundary_num_queries,
        transition_num_queries=args.guide_transition_num_queries,
    )
    boundary_tokenizer = PaligemmaTokenizer(args.max_boundary_text_tokens)
    transition_tokenizer = PaligemmaTokenizer(args.max_transition_text_tokens)
    resolver = VideoHarnessGuideResolver(
        document_catalog=document_catalog,
        guide_records=guide_catalog.records,
        dataset_root=args.dataset_root,
        boundary_tokenizer=boundary_tokenizer,
        transition_tokenizer=transition_tokenizer,
        materializer_config=config,
        plans_by_document=plans,
    )
    cache = ensure_guide_materialization_cache(
        cache_root=args.guide_materialization_cache_root,
        catalog_digest=guide_catalog.catalog_digest,
        guide_records=guide_catalog.records,
        document_catalog=document_catalog,
        plans_by_document=plans,
        materializer_config=config,
        boundary_tokenizer=boundary_tokenizer,
        transition_tokenizer=transition_tokenizer,
        source_resolver=resolver,
    )
    return {
        "catalog_digest": cache.catalog_digest,
        "materialization_digest": cache.materialization_digest,
        "cache_digest": cache.cache_digest,
        "cache_root": str(args.guide_materialization_cache_root),
        "guide_lengths": {
            document_id: {"units": value[0], "boundaries": value[1]}
            for document_id, value in sorted(lengths.items())
        },
        **dict(cache.stats),
    }


def main(argv: list[str] | None = None) -> int:
    result = build_cache(_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
