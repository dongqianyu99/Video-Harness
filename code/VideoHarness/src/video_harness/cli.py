from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .annotations import make_backends
from .config import HarnessConfig
from .evidence import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    boundary_state_is_usable,
    evidence_is_trainable,
    validate_boundary_state_record,
    validate_evidence_record,
)
from .media import FFmpegFrameLoader
from .pairing import build_pairs
from .pipeline import EvidenceUnitPipeline
from .reconciliation import reconcile_document, sequence_projection_sha256
from .robodojo import EpisodeRecord, load_info, read_episodes, summarize, validate_info
from .sampling import plan_document, unit_boundary_states, validate_document
from .temporal_media import TemporalMediaBuilder
from .training_split import build_training_split, episode_record_from_dict


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _require_new_or_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
    return values


def _inspect(args: argparse.Namespace) -> int:
    info = load_info(args.dataset_root)
    validate_info(info)
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_root": str(args.dataset_root.resolve()),
                "codebase_version": info["codebase_version"],
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
                "tasks": info["total_tasks"],
                "fps": info["fps"],
            },
            indent=2,
        )
    )
    return 0


def _select_records(
    records: list[EpisodeRecord], max_tasks: int | None, episodes_per_task: int | None
) -> list[EpisodeRecord]:
    if max_tasks is None and episodes_per_task is None:
        return records
    by_task: dict[int, list[EpisodeRecord]] = {}
    for record in records:
        by_task.setdefault(record.task_index, []).append(record)
    task_indices = sorted(by_task)
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("--max-tasks must be positive")
        task_indices = task_indices[:max_tasks]
    selected: list[EpisodeRecord] = []
    for task_index in task_indices:
        task_records = sorted(by_task[task_index], key=lambda item: item.episode_index)
        if episodes_per_task is not None:
            if episodes_per_task < 2:
                raise ValueError(
                    "--episodes-per-task must be at least two for support/query pairing"
                )
            task_records = task_records[:episodes_per_task]
        selected.extend(task_records)
    return selected


def _build(args: argparse.Namespace) -> int:
    _require_new_or_empty_directory(args.output_root)
    source_records = read_episodes(args.dataset_root)
    records = _select_records(source_records, args.max_tasks, args.episodes_per_task)
    build_id = (
        f"robodojo-main__benchmark-34__hz-{args.sample_hz:g}__"
        f"supports-{args.supports_per_query}__seed-{args.seed}__"
        f"tasks-{args.max_tasks if args.max_tasks is not None else 'all'}__"
        f"episodes-{args.episodes_per_task if args.episodes_per_task is not None else 'all'}"
    )
    summary = summarize(records)
    summary.update(
        {
            "build_id": build_id,
            "source_dataset": "RoboDojo-Benchmark/RoboDojo/data/RoboDojo_lerobot_v30_video",
            "source_revision": "main",
            "sample_hz": args.sample_hz,
            "supports_per_query": args.supports_per_query,
            "document_camera": "observation.images.cam_high",
            "benchmark_source_episodes": len(source_records),
            "selection": {
                "max_tasks": args.max_tasks,
                "episodes_per_task": args.episodes_per_task,
            },
        }
    )
    _write_json(args.output_root / "dataset.json", summary)
    _write_jsonl(
        args.output_root / "episodes.jsonl", [record.to_dict() for record in records]
    )
    documents = [
        plan_document(record, build_id=build_id, sample_hz=args.sample_hz)
        for record in records
    ]
    for document in documents:
        validate_document(document)
    _write_jsonl(args.output_root / "documents.jsonl", documents)
    pairs = build_pairs(
        records,
        build_id=build_id,
        supports_per_query=args.supports_per_query,
        seed=args.seed,
    )
    _write_jsonl(args.output_root / "pairs.jsonl", pairs)
    print(
        json.dumps(
            {**summary, "documents": len(documents), "pairs": len(pairs)}, indent=2
        )
    )
    return 0


def _annotate(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"Annotation output already exists: {args.output}")
    documents = _read_jsonl(args.documents)
    inspection_backend, evidence_backend, repair_backend = make_backends(
        args.provider,
        args.model,
    )
    debug_root = None
    if args.debug:
        debug_root = args.debug_root or args.output.with_suffix(
            args.output.suffix + ".debug"
        )
    config = HarnessConfig(
        debug=args.debug,
        debug_root=debug_root,
        inspection_retries=args.inspection_retries,
        repair_max_attempts=getattr(args, "repair_max_attempts", 2),
        sequence_audit_max_attempts=getattr(
            args,
            "sequence_audit_max_attempts",
            2,
        ),
        sequence_repair_rounds=getattr(args, "sequence_repair_rounds", 2),
    )
    pipeline = EvidenceUnitPipeline(
        inspection_backend=inspection_backend,
        evidence_backend=evidence_backend,
        media_builder=TemporalMediaBuilder(args.dataset_root),
        config=config,
        repair_backend=repair_backend,
    )
    annotated_units = 0
    annotated_documents = 0
    failed_units = 0
    document_quality_counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []

    for document_index, original in enumerate(documents):
        if args.limit_documents is not None and document_index >= args.limit_documents:
            break
        validate_document(original)
        document = copy.deepcopy(original)
        unit_budget = args.limit_units_per_document
        document_annotations = 0
        for unit in document["evidence_units"]:
            if unit_budget is not None and document_annotations >= unit_budget:
                break
            annotation = unit["annotation"]
            if annotation.get("status") in {"complete", "mock"} and annotation.get(
                "record"
            ):
                continue
            try:
                result = pipeline.run(document, unit)
            except Exception as exc:  # noqa: BLE001 - isolate one failed Unit
                annotation["schema_version"] = EVIDENCE_SCHEMA_VERSION
                annotation["status"] = "failed"
                annotation["record"] = None
                annotation["provenance"] = None
                for boundary in unit_boundary_states(document, unit):
                    boundary_annotation = boundary["annotation"]
                    if boundary_annotation.get("status") not in {"complete", "mock"}:
                        boundary_annotation["schema_version"] = (
                            BOUNDARY_STATE_SCHEMA_VERSION
                        )
                        boundary_annotation["status"] = "failed"
                        boundary_annotation["record"] = None
                        boundary_annotation["provenance"] = None
                document_annotations += 1
                failed_units += 1
                failures.append(
                    {
                        "document_id": document["document_id"],
                        "unit_id": unit["unit_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            output_status = "mock" if result.evidence.provider == "mock" else "complete"
            annotation["schema_version"] = EVIDENCE_SCHEMA_VERSION
            annotation["status"] = output_status
            annotation["record"] = result.evidence.evidence
            call2_model = (
                result.initial_evidence.trace.response_model
                or result.initial_evidence.requested_model
            )
            call2_provenance = {
                "provider": result.initial_evidence.provider,
                "model": call2_model,
                "prompt_version": result.initial_evidence.prompt_version,
            }
            annotation["provenance"] = {
                "call1": {
                    "provider": result.inspection.provider,
                    "model": (
                        result.inspection.trace.response_model
                        or result.inspection.requested_model
                    ),
                    "prompt_version": result.inspection.prompt_version,
                },
                "call2": call2_provenance,
                "repair": (
                    None
                    if result.repair is None
                    else {
                        "provider": result.repair.provider,
                        "model": (
                            result.repair.trace.response_model
                            or result.repair.requested_model
                        ),
                        "prompt_version": result.repair.prompt_version,
                        "attempts": result.repair_attempts,
                        "reason": result.repair.repair["reason"],
                    }
                ),
            }
            before_boundary, after_boundary = unit_boundary_states(document, unit)
            boundary_provenance_base = call2_provenance
            if (
                result.repair is not None
                and result.repair.repair["evidence_sufficient"]
            ):
                boundary_provenance_base = {
                    "provider": result.repair.provider,
                    "model": (
                        result.repair.trace.response_model
                        or result.repair.requested_model
                    ),
                    "prompt_version": result.repair.prompt_version,
                }
            generated_boundaries = (
                (
                    before_boundary,
                    "before",
                    result.before_boundary_record,
                ),
                (
                    after_boundary,
                    "after",
                    result.after_boundary_record,
                ),
            )
            for boundary, role, record in generated_boundaries:
                if record is None:
                    continue
                boundary["annotation"] = {
                    "schema_version": BOUNDARY_STATE_SCHEMA_VERSION,
                    "status": output_status,
                    "record": record,
                    "provenance": {
                        **boundary_provenance_base,
                        "source_unit_id": unit["unit_id"],
                        "boundary_role": role,
                    },
                }
            boundary_by_role = {
                "before": before_boundary,
                "after": after_boundary,
            }
            for role in result.conflicted_boundary_roles:
                boundary_annotation = boundary_by_role[role]["annotation"]
                boundary_record = boundary_annotation.get("record")
                if isinstance(boundary_record, dict):
                    boundary_record["quality_status"] = "quarantined"
            document_annotations += 1
            annotated_units += 1

        statuses = {
            item["annotation"]["status"]
            for key in ("boundary_states", "evidence_units")
            for item in document[key]
        }
        if statuses == {"pending"}:
            document["status"] = "planned"
        elif statuses == {"complete"}:
            document["status"] = "annotated"
        elif statuses == {"mock"}:
            document["status"] = "mock-annotated"
        else:
            document["status"] = "partially-annotated"
        document["quality_status"] = "pending"
        document["quality_provenance"] = None
        if document["status"] in {"annotated", "mock-annotated"}:
            reconciliation = reconcile_document(
                document,
                backend=repair_backend,
                pipeline=pipeline,
                config=config,
            )
            document = reconciliation.document
        elif any(
            unit["annotation"]["status"] == "failed"
            for unit in document["evidence_units"]
        ):
            failed_issue_list = [
                {
                    "unit_id": unit["unit_id"],
                    "reason": "Evidence Unit provider or schema processing failed.",
                }
                for unit in document["evidence_units"]
                if unit["annotation"]["status"] == "failed"
            ]
            document["quality_status"] = "quarantined"
            document["quality_provenance"] = {
                "provider": repair_backend.provider,
                "model": repair_backend.model,
                "prompt_version": "video-harness.sequence-audit",
                "audit_attempts": 0,
                "repair_rounds": 0,
                "sequence_sha256": sequence_projection_sha256(document),
                "issues": failed_issue_list,
            }
        documents[document_index] = document
        validate_document(document)
        document_quality_counts[document["quality_status"]] += 1
        annotated_documents += 1

    for document in documents:
        validate_document(document)
    _write_jsonl(args.output, documents)
    print(
        json.dumps(
            {
                "provider": evidence_backend.provider,
                "model": evidence_backend.model,
                "documents_touched": annotated_documents,
                "units_annotated": annotated_units,
                "units_failed": failed_units,
                "document_quality_status": dict(
                    sorted(document_quality_counts.items())
                ),
                "failure_examples": failures[:20],
                "debug_root": None if debug_root is None else str(debug_root),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def _make_training_split(args: argparse.Namespace) -> int:
    _require_new_or_empty_directory(args.output_root)
    dataset = json.loads(args.dataset_artifact.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise TypeError("--dataset-artifact must contain one JSON object")
    build_id = dataset.get("build_id")
    if not isinstance(build_id, str) or not build_id.strip():
        raise ValueError("dataset artifact has no valid build_id")

    records = [episode_record_from_dict(value) for value in _read_jsonl(args.episodes)]
    documents = _read_jsonl(args.documents)
    manifest, pairs = build_training_split(
        records,
        documents,
        build_id=build_id,
        support_documents_per_task=args.support_documents_per_task,
        heldout_documents_per_task=args.heldout_documents_per_task,
        query_episodes_per_task=args.query_episodes_per_task,
        min_trainable_units=args.min_trainable_units,
        seed=args.seed,
    )
    _write_json(args.output_root / "training-split.json", manifest)
    _write_jsonl(args.output_root / "train-pairs.jsonl", pairs)
    print(
        json.dumps(
            {
                "split_id": manifest["split_id"],
                **manifest["totals"],
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    documents = _read_jsonl(args.documents)
    document_quality_status: Counter[str] = Counter()
    boundary_annotation_status: Counter[str] = Counter()
    boundary_quality_status: Counter[str] = Counter()
    annotation_status: Counter[str] = Counter()
    quality_status: Counter[str] = Counter()
    causal_validation: Counter[str] = Counter()
    detail_observation: Counter[str] = Counter()
    trainable_units = 0
    trainable_documents = 0
    invalid: list[dict[str, str]] = []
    total_units = 0
    total_boundaries = 0

    for document in documents:
        try:
            validate_document(document)
        except (TypeError, ValueError) as exc:
            invalid.append(
                {
                    "document_id": str(document.get("document_id"))
                    if isinstance(document, dict)
                    else "<invalid-document>",
                    "unit_id": "<document>",
                    "error": str(exc),
                }
            )
            continue
        document_quality_status[document["quality_status"]] += 1
        if document["quality_status"] == "accepted":
            trainable_documents += 1
        for boundary in document["boundary_states"]:
            total_boundaries += 1
            annotation = boundary["annotation"]
            status = str(annotation["status"])
            boundary_annotation_status[status] += 1
            if status in {"complete", "mock"}:
                boundary = validate_boundary_state_record(annotation["record"])
                boundary_quality_status[boundary["quality_status"]] += 1
        for unit in document.get("evidence_units", []):
            total_units += 1
            annotation = unit.get("annotation")
            if not isinstance(annotation, dict):
                invalid.append(
                    {
                        "document_id": str(document.get("document_id")),
                        "unit_id": str(unit.get("unit_id")),
                        "error": "missing annotation object",
                    }
                )
                continue
            if set(annotation) != {"schema_version", "status", "record", "provenance"}:
                invalid.append(
                    {
                        "document_id": str(document.get("document_id")),
                        "unit_id": str(unit.get("unit_id")),
                        "error": "annotation must contain exact schema/status/record/provenance fields",
                    }
                )
                continue
            status = str(annotation.get("status"))
            annotation_status[status] += 1
            record = annotation.get("record")
            try:
                if annotation.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
                    raise ValueError(
                        f"unexpected evidence schema {annotation.get('schema_version')!r}"
                    )
                if status in {"pending", "failed"}:
                    if record is not None or annotation.get("provenance") is not None:
                        raise ValueError(
                            f"{status} annotation requires null record and provenance"
                        )
                    continue
                evidence = validate_evidence_record(record)
                if status in {"complete", "mock"}:
                    provenance = annotation.get("provenance")
                    if not isinstance(provenance, dict):
                        raise ValueError("complete/mock evidence requires provenance")
                elif status not in {"pending", "failed"}:
                    raise ValueError(f"unsupported annotation status {status!r}")
            except (TypeError, ValueError) as exc:
                invalid.append(
                    {
                        "document_id": str(document.get("document_id")),
                        "unit_id": str(unit.get("unit_id")),
                        "error": str(exc),
                    }
                )
                continue

            quality_status[evidence["quality_status"]] += 1
            causal_validation[evidence["causal_validation"]["status"]] += 1
            detail_observation[
                "present" if evidence["detail_observation"] is not None else "absent"
            ] += 1
            boundaries = unit_boundary_states(document, unit)
            usable_boundaries = all(
                boundary["annotation"]["status"] == "complete"
                and boundary_state_is_usable(boundary["annotation"]["record"])
                for boundary in boundaries
            )
            if (
                status == "complete"
                and evidence_is_trainable(evidence)
                and usable_boundaries
                and document["quality_status"] == "accepted"
            ):
                trainable_units += 1

    report = {
        "documents": len(documents),
        "document_quality_status": dict(sorted(document_quality_status.items())),
        "trainable_documents_default": trainable_documents,
        "boundary_states": total_boundaries,
        "boundary_annotation_status": dict(sorted(boundary_annotation_status.items())),
        "boundary_quality_status": dict(sorted(boundary_quality_status.items())),
        "units": total_units,
        "annotation_status": dict(sorted(annotation_status.items())),
        "quality_status": dict(sorted(quality_status.items())),
        "causal_validation": dict(sorted(causal_validation.items())),
        "detail_observation": dict(sorted(detail_observation.items())),
        "trainable_units_default": trainable_units,
        "invalid_units": len(invalid),
        "invalid_examples": invalid[:20],
    }
    print(json.dumps(report, indent=2))
    return 1 if invalid else 0


def _decode_smoke(args: argparse.Namespace) -> int:
    documents = _read_jsonl(args.documents)
    loader = FFmpegFrameLoader(args.dataset_root)
    decoded: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for document in documents:
        validate_document(document)
        for boundary in document["boundary_states"]:
            frame_ref = boundary["frame"]
            key = (document["document_id"], int(frame_ref["episode_frame_index"]))
            if key in seen:
                continue
            seen.add(key)
            image = loader.load(document, frame_ref)
            decoded.append(
                {
                    "document_id": document["document_id"],
                    "boundary_id": boundary["boundary_id"],
                    "episode_frame_index": frame_ref["episode_frame_index"],
                    "jpeg_bytes": len(image),
                }
            )
            if len(decoded) >= args.limit_frames:
                print(json.dumps({"decoded": decoded}, indent=2))
                return 0
    print(json.dumps({"decoded": decoded}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multiview temporal Video Harness compiler"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect", help="validate the public RoboDojo Pi_05 source"
    )
    inspect.add_argument("--dataset-root", type=Path, required=True)
    inspect.set_defaults(handler=_inspect)

    build = subparsers.add_parser(
        "build", help="build inventory, draft documents, and pairs"
    )
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--sample-hz", type=float, default=1.0)
    build.add_argument("--supports-per-query", type=int, default=1)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--max-tasks", type=int)
    build.add_argument("--episodes-per-task", type=int)
    build.set_defaults(handler=_build)

    annotate = subparsers.add_parser(
        "annotate",
        help="compile structured transition evidence with a mock or VLM provider",
    )
    annotate.add_argument("--documents", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--dataset-root", type=Path, required=True)
    annotate.add_argument(
        "--provider", choices=("mock", "openai", "anthropic"), required=True
    )
    annotate.add_argument("--model")
    annotate.add_argument("--limit-documents", type=int)
    annotate.add_argument("--limit-units-per-document", type=int)
    annotate.add_argument("--debug", action="store_true")
    annotate.add_argument("--debug-root", type=Path)
    annotate.add_argument("--inspection-retries", type=int, default=1)
    annotate.add_argument("--repair-max-attempts", type=int, default=2)
    annotate.add_argument("--sequence-audit-max-attempts", type=int, default=2)
    annotate.add_argument("--sequence-repair-rounds", type=int, default=2)
    annotate.set_defaults(handler=_annotate)

    split = subparsers.add_parser(
        "make-training-split",
        help="build a role-disjoint training split and balanced static Guide bindings",
    )
    split.add_argument("--dataset-artifact", type=Path, required=True)
    split.add_argument("--episodes", type=Path, required=True)
    split.add_argument("--documents", type=Path, required=True)
    split.add_argument("--output-root", type=Path, required=True)
    split.add_argument("--support-documents-per-task", type=int, required=True)
    split.add_argument("--heldout-documents-per-task", type=int, required=True)
    split.add_argument("--query-episodes-per-task", type=int)
    split.add_argument("--min-trainable-units", type=int, default=1)
    split.add_argument("--seed", type=int, default=0)
    split.set_defaults(handler=_make_training_split)

    report = subparsers.add_parser(
        "report", help="validate evidence records and summarize annotation usability"
    )
    report.add_argument("--documents", type=Path, required=True)
    report.set_defaults(handler=_report)

    decode = subparsers.add_parser(
        "decode-smoke", help="decode referenced frames without saving images"
    )
    decode.add_argument("--documents", type=Path, required=True)
    decode.add_argument("--dataset-root", type=Path, required=True)
    decode.add_argument("--limit-frames", type=int, default=2)
    decode.set_defaults(handler=_decode_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
