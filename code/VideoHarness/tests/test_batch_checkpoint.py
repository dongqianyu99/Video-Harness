from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from _support import annotate_boundaries, set_document_quality

from video_harness import cli
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document, validate_document


def _planned_document(episode_index: int) -> dict:
    return plan_document(
        EpisodeRecord(
            episode_index=episode_index,
            task_index=0,
            task_instruction="Place the visible object on the target.",
            task_kind="benchmark",
            length=26,
            dataset_from_index=episode_index * 26,
            dataset_to_index=(episode_index + 1) * 26,
            data_path="data/chunk-000/file-000.parquet",
            videos=tuple(
                VideoSlice(
                    key=key,
                    path=f"videos/{episode_index:04d}.mkv",
                    from_timestamp=0.0,
                    to_timestamp=26 / 25,
                )
                for key in (
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                )
            ),
        ),
        build_id="batch-build",
    )


def _accepted_document(original: dict, evidence: dict) -> dict:
    document = copy.deepcopy(original)
    annotate_boundaries(document)
    document["evidence_units"][0]["annotation"] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "record": copy.deepcopy(evidence),
        "provenance": {
            "call1": {
                "provider": "test",
                "model": "test",
                "prompt_version": "test",
            },
            "call2": {
                "provider": "test",
                "model": "test",
                "prompt_version": "test",
            },
            "repair": None,
        },
    }
    document["status"] = "annotated"
    set_document_quality(document, "accepted")
    return validate_document(document)


def _args(
    *,
    documents: Path,
    output: Path,
    dataset_root: Path,
    checkpoint_root: Path,
    shard_index: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        documents=documents,
        output=output,
        dataset_root=dataset_root,
        provider="mock",
        model=None,
        limit_documents=None,
        limit_units_per_document=None,
        workers=2,
        num_shards=2,
        shard_index=shard_index,
        checkpoint_root=checkpoint_root,
        resume=False,
        debug=False,
        debug_root=None,
        inspection_retries=0,
        repair_max_attempts=2,
        sequence_audit_max_attempts=2,
        sequence_repair_rounds=2,
    )


def test_document_workers_shard_checkpoint_and_merge(
    tmp_path: Path,
    monkeypatch,
    changed_evidence: dict,
) -> None:
    originals = [_planned_document(index) for index in range(20)]
    accepted = {
        document["document_id"]: _accepted_document(document, changed_evidence)
        for document in originals
    }
    documents_path = tmp_path / "documents.jsonl"
    documents_path.write_text(
        "".join(json.dumps(document) + "\n" for document in originals),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_make_worker_context", lambda args, config: object())

    def fake_annotate(original, *, context, config, unit_budget):
        del context, config, unit_budget
        return cli.DocumentAnnotationResult(
            document=copy.deepcopy(accepted[original["document_id"]]),
            annotated_units=1,
            failed_units=0,
            failures=(),
        )

    monkeypatch.setattr(cli, "_annotate_document", fake_annotate)
    checkpoint_root = tmp_path / "checkpoints"
    shard_outputs = []
    for shard_index in range(2):
        output = tmp_path / f"shard-{shard_index}.jsonl"
        shard_outputs.append(output)
        assert (
            cli._annotate(
                _args(
                    documents=documents_path,
                    output=output,
                    dataset_root=tmp_path / "dataset",
                    checkpoint_root=checkpoint_root,
                    shard_index=shard_index,
                )
            )
            == 0
        )

    shard_ids = [
        {
            json.loads(line)["document_id"]
            for line in output.read_text(encoding="utf-8").splitlines()
        }
        for output in shard_outputs
    ]
    assert shard_ids[0]
    assert shard_ids[1]
    assert not shard_ids[0] & shard_ids[1]
    assert shard_ids[0] | shard_ids[1] == set(accepted)

    merged = tmp_path / "merged.jsonl"
    assert (
        cli._merge_checkpoints(
            SimpleNamespace(
                documents=documents_path,
                checkpoint_root=checkpoint_root,
                output=merged,
            )
        )
        == 0
    )
    merged_documents = [
        json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["document_id"] for item in merged_documents] == [
        item["document_id"] for item in originals
    ]


def test_resume_reuses_terminal_document_checkpoint(
    tmp_path: Path,
    monkeypatch,
    changed_evidence: dict,
) -> None:
    original = _planned_document(0)
    terminal = _accepted_document(original, changed_evidence)
    documents_path = tmp_path / "documents.jsonl"
    documents_path.write_text(json.dumps(original) + "\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    config = cli.HarnessConfig(inspection_retries=0)
    cli._ensure_checkpoint_run(
        checkpoint_root,
        documents_path=documents_path,
        dataset_root=tmp_path / "dataset",
        provider="mock",
        model="deterministic-insufficient-evidence",
        num_shards=1,
        config=config,
    )
    cli._write_document_checkpoint(
        checkpoint_root,
        cli.DocumentAnnotationResult(terminal, 1, 0, ()),
    )
    monkeypatch.setattr(
        cli,
        "_make_worker_context",
        lambda args, config: (_ for _ in ()).throw(
            AssertionError("terminal checkpoints must not create a worker")
        ),
    )
    output = tmp_path / "resumed.jsonl"
    args = _args(
        documents=documents_path,
        output=output,
        dataset_root=tmp_path / "dataset",
        checkpoint_root=checkpoint_root,
        shard_index=0,
    )
    args.num_shards = 1
    args.workers = 1
    args.resume = True
    assert cli._annotate(args) == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["quality_status"] == "accepted"
    )
