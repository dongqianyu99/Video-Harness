from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from _support import annotate_boundaries, set_document_quality

from video_harness import cli
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.run_tracking import ApiCallBudgetExceeded
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


def test_legacy_gripper_evidence_is_migrated(changed_evidence: dict) -> None:
    document = _accepted_document(_planned_document(0), changed_evidence)
    annotation = document["evidence_units"][0]["annotation"]
    annotation["schema_version"] = "video-harness.evidence.v4"
    annotation["record"]["gripper_state"] = {"samples": []}

    migrated = cli._migrate_legacy_gripper_evidence(document)

    assert annotation["schema_version"] == "video-harness.evidence.v4"
    assert migrated["evidence_units"][0]["annotation"]["schema_version"] == (
        EVIDENCE_SCHEMA_VERSION
    )
    assert "gripper_state" not in migrated["evidence_units"][0]["annotation"]["record"]
    assert migrated["quality_status"] == "pending"
    assert migrated["quality_provenance"] is None
    validate_document(migrated)


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
    capsys,
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
    monkeypatch.setattr(
        cli, "_make_worker_context", lambda args, config, tracker: object()
    )

    def fake_annotate(
        original,
        *,
        context,
        config,
        unit_budget,
        selected_unit_ids=None,
    ):
        del context, config, unit_budget, selected_unit_ids
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

    final_files = sorted((tmp_path / "documents-mock").glob("*/*.document.jsonl"))
    assert len(final_files) == len(originals)
    assert len({path.parent for path in final_files}) == 1
    assert final_files[0].parent.name == "place-the-visible-object-on-the-target"
    assert {json.loads(path.read_text())["document_id"] for path in final_files} == set(
        accepted
    )
    progress_output = capsys.readouterr().err
    assert "Place the visible object on the target." in progress_output
    assert "[####################]" in progress_output

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
        lambda args, config, tracker: (_ for _ in ()).throw(
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


def test_resume_retries_technical_quarantine(
    tmp_path: Path,
    monkeypatch,
    changed_evidence: dict,
) -> None:
    original = _planned_document(0)
    quarantined = _accepted_document(original, changed_evidence)
    set_document_quality(quarantined, "quarantined")
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
        cli.DocumentAnnotationResult(
            quarantined,
            0,
            1,
            (
                {
                    "document_id": quarantined["document_id"],
                    "unit_id": "u0000",
                    "error": "provider timeout",
                },
            ),
        ),
    )
    retried = []

    def fake_annotate(document, **_kwargs):
        retried.append(document["document_id"])
        return cli.DocumentAnnotationResult(
            _accepted_document(original, changed_evidence),
            1,
            0,
            (),
        )

    monkeypatch.setattr(cli, "_make_worker_context", lambda *_args: object())
    monkeypatch.setattr(cli, "_annotate_document", fake_annotate)
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
    assert retried == [original["document_id"]]
    assert (
        json.loads(output.read_text(encoding="utf-8"))["quality_status"] == "accepted"
    )


def test_api_budget_exhaustion_does_not_quarantine_document() -> None:
    class Pipeline:
        @staticmethod
        def run(document, unit):
            del document, unit
            raise ApiCallBudgetExceeded("budget exhausted")

    original = _planned_document(0)
    with pytest.raises(ApiCallBudgetExceeded, match="budget exhausted"):
        cli._annotate_document(
            original,
            context=SimpleNamespace(pipeline=Pipeline(), repair_backend=None),
            config=cli.HarnessConfig(),
            unit_budget=None,
        )
    assert original["quality_status"] == "pending"
    assert original["evidence_units"][0]["annotation"]["status"] == "pending"
