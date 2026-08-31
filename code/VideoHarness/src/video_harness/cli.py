from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotations import RepairBackend, make_backends
from .config import HarnessConfig
from .evidence import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    validate_boundary_state_record,
    validate_evidence_record,
)
from .hdf5_source import (
    HDF5_SOURCE_DATASET,
    hdf5_document_source,
    inspect_hdf5_episode,
    load_hdf5_jpeg,
)
from .media import FFmpegFrameLoader
from .pipeline import EvidenceUnitPipeline
from .reconciliation import reconcile_document
from .robodojo import EpisodeRecord, load_info, read_episodes, summarize, validate_info
from .run_tracking import (
    ApiCallBudgetExceeded,
    RunTracker,
    TrackingBackend,
    summarize_run,
)
from .sampling import (
    plan_document,
    plan_document_from_source,
    unit_boundary_states,
    validate_document,
)
from .temporal_media import TemporalMediaBuilder

CHECKPOINT_SCHEMA_VERSION = "video-harness.document-checkpoint"
CHECKPOINT_RUN_SCHEMA_VERSION = "video-harness.checkpoint-run"


@dataclass(frozen=True)
class DocumentAnnotationResult:
    document: dict[str, Any]
    annotated_units: int
    failed_units: int
    failures: tuple[dict[str, str], ...]
    reused: bool = False


@dataclass(frozen=True)
class _WorkerContext:
    pipeline: EvidenceUnitPipeline
    repair_backend: RepairBackend


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _path_slug(value: str, *, max_length: int = 56) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "unnamed"


def _document_task_name(document: dict[str, Any]) -> str:
    source = document["source"]
    return str(source.get("task_name") or document["task_instruction"])


def _final_document_path(root: Path, document: dict[str, Any]) -> Path:
    source = document["source"]
    task_folder = _path_slug(_document_task_name(document))
    episode_index = source.get("episode_index")
    if isinstance(episode_index, int) and not isinstance(episode_index, bool):
        episode_name = f"episode-{episode_index:07d}"
    else:
        episode_name = _path_slug(document["document_id"])
    return root / task_folder / f"{episode_name}.document.jsonl"


class _TaskProgress:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        completed_documents: list[dict[str, Any]],
    ) -> None:
        self.totals = Counter(_document_task_name(document) for document in documents)
        self.completed = Counter(
            _document_task_name(document) for document in completed_documents
        )
        self._tty = sys.stderr.isatty()
        self._rendered_lines = 0

    @staticmethod
    def _line(task_name: str, completed: int, total: int) -> str:
        width = 20
        filled = width if total == 0 else round(width * completed / total)
        bar = "#" * filled + "-" * (width - filled)
        label = task_name if len(task_name) <= 48 else task_name[:45] + "..."
        return f"[{bar}] {completed:3d}/{total:<3d} {label}"

    def render(self, changed_task: str | None = None) -> None:
        if self._tty:
            if self._rendered_lines:
                sys.stderr.write(f"\x1b[{self._rendered_lines}F")
            lines = [
                self._line(task, self.completed[task], total)
                for task, total in sorted(self.totals.items())
            ]
            for line in lines:
                sys.stderr.write("\r\x1b[2K" + line + "\n")
            sys.stderr.flush()
            self._rendered_lines = len(lines)
            return
        if changed_task is not None:
            print(
                self._line(
                    changed_task,
                    self.completed[changed_task],
                    self.totals[changed_task],
                ),
                file=sys.stderr,
                flush=True,
            )

    def advance(self, document: dict[str, Any]) -> None:
        task_name = _document_task_name(document)
        self.completed[task_name] += 1
        self.render(task_name)


def _write_final_document(root: Path, document: dict[str, Any]) -> Path:
    validate_document(document)
    path = _final_document_path(root, document)
    _write_jsonl(path, [document])
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(root: Path, document_id: str) -> Path:
    name = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return root / "documents" / f"{name}.json"


def _write_document_checkpoint(
    root: Path,
    result: DocumentAnnotationResult,
) -> Path:
    path = _checkpoint_path(root, result.document["document_id"])
    _write_json(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "document_id": result.document["document_id"],
            "document": result.document,
            "failures": list(result.failures),
        },
    )
    return path


def _load_document_checkpoint(
    root: Path, document_id: str
) -> DocumentAnnotationResult | None:
    path = _checkpoint_path(root, document_id)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "document_id",
        "document",
        "failures",
    }:
        raise ValueError(f"Invalid document checkpoint: {path}")
    if value["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported document checkpoint schema: {path}")
    if value["document_id"] != document_id:
        raise ValueError(f"Document checkpoint identity mismatch: {path}")
    document = validate_document(value["document"])
    failures = value["failures"]
    if not isinstance(failures, list) or any(
        not isinstance(item, dict) for item in failures
    ):
        raise ValueError(f"Invalid checkpoint failures: {path}")
    return DocumentAnnotationResult(
        document=document,
        annotated_units=0,
        failed_units=0,
        failures=tuple(failures),
        reused=True,
    )


def _document_shard(document_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def _ensure_checkpoint_run(
    root: Path,
    *,
    documents_path: Path,
    dataset_root: Path,
    provider: str,
    model: str,
    num_shards: int,
    config: HarnessConfig,
    unit_ids: tuple[str, ...] | None = None,
) -> None:
    config_value = config.manifest()
    config_value.pop("debug_root", None)
    expected = {
        "schema_version": CHECKPOINT_RUN_SCHEMA_VERSION,
        "documents_sha256": _file_sha256(documents_path),
        "dataset_root": str(dataset_root.resolve()),
        "provider": provider,
        "model": model,
        "num_shards": num_shards,
        "unit_ids": None if unit_ids is None else list(unit_ids),
        "config": config_value,
    }
    path = root / "run.json"
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                f"Checkpoint run contract does not match this invocation: {path}"
            )
        return
    _write_json(path, expected)
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(f"Checkpoint run contract changed during creation: {path}")


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
            if episodes_per_task < 1:
                raise ValueError("--episodes-per-task must be positive")
            task_records = task_records[:episodes_per_task]
        selected.extend(task_records)
    return selected


def _build(args: argparse.Namespace) -> int:
    _require_new_or_empty_directory(args.output_root)
    source_records = read_episodes(args.dataset_root)
    records = _select_records(source_records, args.max_tasks, args.episodes_per_task)
    build_id = (
        f"robodojo-main__benchmark-34__hz-{args.sample_hz:g}__"
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
            "document_views": [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
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
    print(
        json.dumps(
            {**summary, "documents": len(documents)}, indent=2
        )
    )
    return 0


def _build_hdf5(args: argparse.Namespace) -> int:
    _require_new_or_empty_directory(args.output_root)
    episode = inspect_hdf5_episode(args.episode)
    build_id = (
        f"robodojo-hdf5-v1__task-{episode.task_name}__"
        f"episode-{episode.episode_index:07d}__hz-{args.sample_hz:g}"
    )
    document = plan_document_from_source(
        build_id=build_id,
        document_id=(
            f"robodojo-hdf5/{episode.task_name}/episode-{episode.episode_index:07d}"
        ),
        source=hdf5_document_source(episode),
        task_instruction=episode.task_instruction,
        sample_hz=args.sample_hz,
    )
    validate_document(document)
    summary = {
        "schema_version": "video-harness.hdf5-source",
        "build_id": build_id,
        "source_dataset": HDF5_SOURCE_DATASET,
        "episode": episode.path.name,
        "episode_index": episode.episode_index,
        "task_name": episode.task_name,
        "task_instruction": episode.task_instruction,
        "frames": episode.length,
        "fps": episode.fps,
        "sample_hz": args.sample_hz,
        "documents": 1,
    }
    _write_json(args.output_root / "dataset.json", summary)
    _write_jsonl(args.output_root / "documents.jsonl", [document])
    print(json.dumps({**summary, "output_root": str(args.output_root)}, indent=2))
    return 0


def _make_worker_context(
    args: argparse.Namespace,
    config: HarnessConfig,
    tracker: RunTracker,
) -> _WorkerContext:
    inspection_backend, evidence_backend, repair_backend = make_backends(
        args.provider,
        args.model,
        timeout_s=config.provider_timeout_s,
        max_retries=config.provider_max_retries,
    )
    inspection_backend = TrackingBackend(inspection_backend, tracker)
    evidence_backend = TrackingBackend(evidence_backend, tracker)
    repair_backend = TrackingBackend(repair_backend, tracker)
    return _WorkerContext(
        pipeline=EvidenceUnitPipeline(
            inspection_backend=inspection_backend,
            evidence_backend=evidence_backend,
            media_builder=TemporalMediaBuilder(
                args.dataset_root,
                timeout_s=config.ffmpeg_timeout_s,
            ),
            config=config,
            repair_backend=repair_backend,
        ),
        repair_backend=repair_backend,
    )


def _annotate_document(
    original: dict[str, Any],
    *,
    context: _WorkerContext,
    config: HarnessConfig,
    unit_budget: int | None,
    selected_unit_ids: frozenset[str] | None = None,
) -> DocumentAnnotationResult:
    validate_document(original)
    document = copy.deepcopy(original)
    annotated_units = 0
    failed_units = 0
    failures: list[dict[str, str]] = []
    document_annotations = 0
    for unit in document["evidence_units"]:
        if selected_unit_ids is not None and unit["unit_id"] not in selected_unit_ids:
            continue
        if unit_budget is not None and document_annotations >= unit_budget:
            break
        annotation = unit["annotation"]
        if (
            selected_unit_ids is None
            and annotation.get("status") in {"complete", "mock"}
            and annotation.get("record")
        ):
            continue
        try:
            result = context.pipeline.run(document, unit)
        except ApiCallBudgetExceeded:
            raise
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
            result.evidence.trace.response_model or result.evidence.requested_model
        )
        call2_provenance = {
            "provider": result.evidence.provider,
            "model": call2_model,
            "prompt_version": result.evidence.prompt_version,
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
            "repair": None,
        }
        before_boundary, after_boundary = unit_boundary_states(document, unit)
        for boundary, role, record in (
            (before_boundary, "before", result.before_boundary_record),
            (after_boundary, "after", result.after_boundary_record),
        ):
            if record is None:
                continue
            boundary["annotation"] = {
                "schema_version": BOUNDARY_STATE_SCHEMA_VERSION,
                "status": output_status,
                "record": record,
                "provenance": {
                    **call2_provenance,
                    "source_unit_id": unit["unit_id"],
                    "boundary_role": role,
                },
            }
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
            backend=context.repair_backend,
            pipeline=context.pipeline,
            config=config,
        )
        document = reconciliation.document
        failures.extend(
            {
                "document_id": document["document_id"],
                "unit_id": failure["target_id"],
                "error": failure["error"],
            }
            for failure in reconciliation.technical_failures
        )
    validate_document(document)
    return DocumentAnnotationResult(
        document=document,
        annotated_units=annotated_units,
        failed_units=failed_units,
        failures=tuple(failures),
    )


def _annotate(args: argparse.Namespace) -> int:
    resume = bool(getattr(args, "resume", False))
    workers = int(getattr(args, "workers", 1))
    num_shards = int(getattr(args, "num_shards", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    if workers < 1:
        raise ValueError("--workers must be at least one")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be within [0, --num-shards)")
    if args.output.exists() and not resume:
        raise FileExistsError(f"Annotation output already exists: {args.output}")

    unit_ids = tuple(getattr(args, "unit_id", None) or ())
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("--unit-id values must be unique")
    if unit_ids and args.limit_units_per_document is not None:
        raise ValueError("--unit-id cannot be combined with --limit-units-per-document")

    documents = _read_jsonl(args.documents)
    for document in documents:
        validate_document(document)
        available = {unit["unit_id"] for unit in document["evidence_units"]}
        missing = sorted(set(unit_ids) - available)
        if missing:
            raise ValueError(
                f"Document {document['document_id']} has no Units: {', '.join(missing)}"
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
        media_retries=getattr(args, "media_retries", 2),
        call2_retries=getattr(args, "call2_retries", 2),
        repair_max_attempts=getattr(args, "repair_max_attempts", 2),
        sequence_audit_max_attempts=getattr(args, "sequence_audit_max_attempts", 2),
        sequence_repair_rounds=getattr(args, "sequence_repair_rounds", 2),
        provider_timeout_s=getattr(args, "provider_timeout_s", 300.0),
        provider_max_retries=getattr(args, "provider_max_retries", 2),
        ffmpeg_timeout_s=getattr(args, "ffmpeg_timeout_s", 120.0),
    )
    checkpoint_root = getattr(args, "checkpoint_root", None)
    if num_shards > 1 and checkpoint_root is None:
        raise ValueError(
            "--checkpoint-root is required when --num-shards is greater than one"
        )
    checkpoint_root = (
        Path(checkpoint_root)
        if checkpoint_root is not None
        else args.output.with_name(args.output.name + ".checkpoints")
    )
    document_root = getattr(args, "document_root", None)
    document_root = (
        Path(document_root)
        if document_root is not None
        else args.output.parent / f"documents-{args.provider}"
    )
    model_name = (
        "deterministic-insufficient-evidence" if args.provider == "mock" else args.model
    )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"--model is required for provider {args.provider!r}")
    _ensure_checkpoint_run(
        checkpoint_root,
        documents_path=args.documents,
        dataset_root=args.dataset_root,
        provider=args.provider,
        model=model_name,
        num_shards=num_shards,
        config=config,
        unit_ids=unit_ids or None,
    )
    tracker = RunTracker(
        checkpoint_root,
        max_api_calls=getattr(args, "max_api_calls", None),
        context={"shard_index": shard_index, "num_shards": num_shards},
    )
    tracker.log(
        "shard_started",
        workers=workers,
        resume=resume,
        max_api_calls=tracker.max_api_calls,
    )

    assigned = [
        (index, document)
        for index, document in enumerate(documents)
        if _document_shard(document["document_id"], num_shards) == shard_index
    ]
    if args.limit_documents is not None:
        if args.limit_documents < 0:
            raise ValueError("--limit-documents must be non-negative")
        assigned = assigned[: args.limit_documents]
    task_totals = Counter(_document_task_name(document) for _, document in assigned)
    tracker.log("task_plan", tasks=dict(sorted(task_totals.items())))

    results: dict[int, DocumentAnnotationResult] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, original in assigned:
        checkpoint = _load_document_checkpoint(checkpoint_root, original["document_id"])
        if checkpoint is None:
            pending.append((index, original))
            continue
        if not resume:
            raise FileExistsError(
                "Document checkpoint already exists; rerun with --resume: "
                f"{_checkpoint_path(checkpoint_root, original['document_id'])}"
            )
        if (
            checkpoint.document["build_id"] != original["build_id"]
            or checkpoint.document["source"] != original["source"]
        ):
            raise ValueError(
                f"Checkpoint source mismatch for {original['document_id']}"
            )
        terminal_semantic_result = checkpoint.document[
            "quality_status"
        ] == "accepted" or (
            checkpoint.document["quality_status"] == "quarantined"
            and not checkpoint.failures
        )
        if terminal_semantic_result:
            results[index] = checkpoint
            _write_final_document(document_root, checkpoint.document)
            tracker.log(
                "document_reused",
                document_id=checkpoint.document["document_id"],
                task_name=_document_task_name(checkpoint.document),
                quality_status=checkpoint.document["quality_status"],
            )
        else:
            pending.append((index, checkpoint.document))

    local = threading.local()

    def process(
        item: tuple[int, dict[str, Any]],
    ) -> tuple[int, DocumentAnnotationResult]:
        index, document = item
        context = getattr(local, "context", None)
        if context is None:
            context = _make_worker_context(args, config, tracker)
            local.context = context
        task_name = _document_task_name(document)
        tracker.log(
            "document_started",
            document_id=document["document_id"],
            task_name=task_name,
        )
        try:
            result = _annotate_document(
                document,
                context=context,
                config=config,
                unit_budget=args.limit_units_per_document,
                selected_unit_ids=frozenset(unit_ids) if unit_ids else None,
            )
        except Exception as exc:
            tracker.log(
                "document_interrupted",
                document_id=document["document_id"],
                task_name=task_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        _write_document_checkpoint(checkpoint_root, result)
        final_document = result.document["quality_status"] == "accepted" or (
            result.document["quality_status"] == "quarantined" and not result.failures
        )
        if final_document:
            _write_final_document(document_root, result.document)
        for failure in result.failures:
            tracker.log("unit_failed", **failure)
        tracker.log(
            "document_completed",
            document_id=document["document_id"],
            task_name=task_name,
            quality_status=result.document["quality_status"],
            units_annotated=result.annotated_units,
            units_failed=result.failed_units,
        )
        return index, result

    progress = _TaskProgress(
        [document for _, document in assigned],
        [result.document for result in results.values()],
    )
    progress.render()

    def record(index: int, result: DocumentAnnotationResult) -> None:
        results[index] = result
        progress.advance(result.document)

    try:
        if workers == 1:
            for item in pending:
                index, result = process(item)
                record(index, result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process, item) for item in pending]
                try:
                    for future in as_completed(futures):
                        index, result = future.result()
                        record(index, result)
                except Exception:
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
    except Exception as exc:
        tracker.log(
            "shard_interrupted",
            error_type=type(exc).__name__,
            error=str(exc),
            completed_documents=len(results),
            total_documents=len(assigned),
        )
        raise

    if num_shards == 1:
        output_documents = list(documents)
        for index, result in results.items():
            output_documents[index] = result.document
    else:
        output_documents = [results[index].document for index, _ in assigned]
    for document in output_documents:
        validate_document(document)
    _write_jsonl(args.output, output_documents)

    selected_results = [results[index] for index, _ in assigned]
    annotated_units = sum(result.annotated_units for result in selected_results)
    failed_units = sum(result.failed_units for result in selected_results)
    failures = [failure for result in selected_results for failure in result.failures]
    document_quality_counts = Counter(
        result.document["quality_status"] for result in selected_results
    )
    budget = tracker.snapshot()
    tracker.log(
        "shard_completed",
        documents=len(selected_results),
        document_quality_status=dict(sorted(document_quality_counts.items())),
        **budget,
    )
    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": model_name,
                "shard": f"{shard_index}/{num_shards}",
                "workers": workers,
                "documents_touched": sum(
                    not result.reused for result in selected_results
                ),
                "documents_reused": sum(result.reused for result in selected_results),
                "units_annotated": annotated_units,
                "units_failed": failed_units,
                "document_quality_status": dict(
                    sorted(document_quality_counts.items())
                ),
                "failure_examples": failures[:20],
                "api_budget": budget,
                "events": str(tracker.events_path),
                "checkpoint_root": str(checkpoint_root),
                "document_root": str(document_root),
                "debug_root": None if debug_root is None else str(debug_root),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def _merge_checkpoints(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"Merged annotation output already exists: {args.output}")
    documents = _read_jsonl(args.documents)
    run_path = args.checkpoint_root / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Checkpoint run manifest does not exist: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        not isinstance(run, dict)
        or run.get("schema_version") != CHECKPOINT_RUN_SCHEMA_VERSION
        or run.get("documents_sha256") != _file_sha256(args.documents)
    ):
        raise ValueError("Checkpoint run does not match --documents")
    document_root = getattr(args, "document_root", None)
    document_root = (
        Path(document_root)
        if document_root is not None
        else args.output.parent / f"documents-{run['provider']}"
    )

    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    nonterminal: list[str] = []
    for original in documents:
        validate_document(original)
        checkpoint = _load_document_checkpoint(
            args.checkpoint_root, original["document_id"]
        )
        if checkpoint is None:
            missing.append(original["document_id"])
            continue
        document = checkpoint.document
        if (
            document["build_id"] != original["build_id"]
            or document["source"] != original["source"]
        ):
            raise ValueError(
                f"Checkpoint source mismatch for {original['document_id']}"
            )
        if checkpoint.failures:
            nonterminal.append(original["document_id"])
            continue
        if document["quality_status"] not in {"accepted", "quarantined"}:
            nonterminal.append(original["document_id"])
            continue
        _write_final_document(document_root, document)
        merged.append(document)
    if missing or nonterminal:
        raise ValueError(
            "Cannot merge incomplete checkpoints: "
            f"missing={len(missing)} {missing[:5]}, "
            f"nonterminal={len(nonterminal)} {nonterminal[:5]}"
        )
    _write_jsonl(args.output, merged)
    quality = Counter(document["quality_status"] for document in merged)
    print(
        json.dumps(
            {
                "documents": len(merged),
                "document_quality_status": dict(sorted(quality.items())),
                "document_root": str(document_root),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    documents = _read_jsonl(args.documents)
    document_quality_status: Counter[str] = Counter()
    boundary_annotation_status: Counter[str] = Counter()
    annotation_status: Counter[str] = Counter()
    detail_observation: Counter[str] = Counter()
    repaired_units = 0
    audit_attempts = 0
    repair_rounds = 0
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
        provenance = document.get("quality_provenance")
        if isinstance(provenance, dict):
            audit_attempts += int(provenance["audit_attempts"])
            repair_rounds += int(provenance["repair_rounds"])
        if document["quality_status"] == "accepted":
            trainable_documents += 1
        for boundary in document["boundary_states"]:
            total_boundaries += 1
            annotation = boundary["annotation"]
            status = str(annotation["status"])
            boundary_annotation_status[status] += 1
            if status in {"complete", "mock"}:
                validate_boundary_state_record(annotation["record"])
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
                    if provenance.get("repair") is not None:
                        repaired_units += 1
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

            detail_observation[
                "present" if evidence["detail_observation"] else "absent"
            ] += 1
            boundaries = unit_boundary_states(document, unit)
            usable_boundaries = all(
                boundary["annotation"]["status"] == "complete"
                for boundary in boundaries
            )
            if (
                status == "complete"
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
        "units": total_units,
        "annotation_status": dict(sorted(annotation_status.items())),
        "detail_observation": dict(sorted(detail_observation.items())),
        "audit_attempts": audit_attempts,
        "repair_rounds": repair_rounds,
        "repaired_units": repaired_units,
        "trainable_units_default": trainable_units,
        "invalid_units": len(invalid),
        "invalid_examples": invalid[:20],
    }
    print(json.dumps(report, indent=2))
    return 1 if invalid else 0


def _summarize_run(args: argparse.Namespace) -> int:
    summary = summarize_run(args.checkpoint_root)
    output = args.output or args.checkpoint_root / "run-summary.json"
    _write_json(output, summary)
    print(json.dumps({**summary, "output": str(output)}, indent=2))
    return 0


def _decode_smoke(args: argparse.Namespace) -> int:
    documents = _read_jsonl(args.documents)
    video_loader = FFmpegFrameLoader(args.dataset_root)
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
            source = document["source"]
            if source["dataset"] == HDF5_SOURCE_DATASET:
                image = load_hdf5_jpeg(
                    args.dataset_root / source["hdf5_path"],
                    source["views"]["cam_high"]["dataset_key"],
                    int(frame_ref["episode_frame_index"]),
                )
            else:
                image = video_loader.load(document, frame_ref)
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
        "build", help="build source inventory and draft Documents"
    )
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--sample-hz", type=float, default=1.0)
    build.add_argument("--max-tasks", type=int)
    build.add_argument("--episodes-per-task", type=int)
    build.set_defaults(handler=_build)

    build_hdf5 = subparsers.add_parser(
        "build-hdf5", help="build one standalone RoboDojo HDF5 document"
    )
    build_hdf5.add_argument("--episode", type=Path, required=True)
    build_hdf5.add_argument("--output-root", type=Path, required=True)
    build_hdf5.add_argument("--sample-hz", type=float, default=1.0)
    build_hdf5.set_defaults(handler=_build_hdf5)

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
    annotate.add_argument(
        "--unit-id",
        action="append",
        help="process or reprocess only this Evidence Unit; repeat for multiple Units",
    )
    annotate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent Documents in this process; Units remain sequential",
    )
    annotate.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="deterministic Document shard count",
    )
    annotate.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based shard assigned to this process",
    )
    annotate.add_argument(
        "--checkpoint-root",
        type=Path,
        help="shared atomic Document checkpoint directory",
    )
    annotate.add_argument(
        "--document-root",
        type=Path,
        help="task-grouped per-episode final Document directory",
    )
    annotate.add_argument(
        "--resume",
        action="store_true",
        help="reuse terminal checkpoints and continue nonterminal Documents",
    )
    annotate.add_argument(
        "--max-api-calls",
        type=int,
        help="optional shared call cap across this checkpoint root; disabled by default",
    )
    annotate.add_argument("--debug", action="store_true")
    annotate.add_argument("--debug-root", type=Path)
    annotate.add_argument("--inspection-retries", type=int, default=1)
    annotate.add_argument("--media-retries", type=int, default=2)
    annotate.add_argument("--call2-retries", type=int, default=2)
    annotate.add_argument("--provider-timeout-s", type=float, default=300.0)
    annotate.add_argument("--provider-max-retries", type=int, default=2)
    annotate.add_argument("--ffmpeg-timeout-s", type=float, default=120.0)
    annotate.add_argument("--repair-max-attempts", type=int, default=2)
    annotate.add_argument("--sequence-audit-max-attempts", type=int, default=2)
    annotate.add_argument("--sequence-repair-rounds", type=int, default=2)
    annotate.set_defaults(handler=_annotate)

    merge = subparsers.add_parser(
        "merge-checkpoints",
        help="merge terminal per-Document checkpoints in source order",
    )
    merge.add_argument("--documents", type=Path, required=True)
    merge.add_argument("--checkpoint-root", type=Path, required=True)
    merge.add_argument("--document-root", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=_merge_checkpoints)

    report = subparsers.add_parser(
        "report", help="validate evidence records and summarize annotation usability"
    )
    report.add_argument("--documents", type=Path, required=True)
    report.set_defaults(handler=_report)

    summary = subparsers.add_parser(
        "summarize-run",
        help="aggregate events, usage, latency, errors, and per-task progress",
    )
    summary.add_argument("--checkpoint-root", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    summary.set_defaults(handler=_summarize_run)

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
