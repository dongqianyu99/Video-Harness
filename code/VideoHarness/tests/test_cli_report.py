import json
from types import SimpleNamespace

from _support import annotate_boundaries, set_document_quality

from video_harness.cli import _report, _require_new_or_empty_directory
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document


def _document(record: dict) -> dict:
    source = EpisodeRecord(
        episode_index=0,
        task_index=0,
        task_instruction="Put bread into the toaster.",
        task_kind="benchmark",
        length=2,
        dataset_from_index=0,
        dataset_to_index=2,
        data_path="data/chunk-000/file-000.parquet",
        videos=tuple(
            VideoSlice(
                key=key,
                path=f"videos/{key}/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=0.08,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ),
    )
    document = plan_document(source, build_id="test-build")
    annotate_boundaries(document)
    document["status"] = "annotated"
    document["evidence_units"][0]["annotation"] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "record": record,
        "provenance": {
            "call1": {
                "provider": "test",
                "model": "test-motion",
                "prompt_version": "test-inspection",
            },
            "call2": {
                "provider": "test",
                "model": "test-evidence",
                "prompt_version": "test-evidence",
            },
            "repair": None,
        },
    }
    set_document_quality(document, "accepted")
    return document


def test_report_counts_trainable_structured_evidence(
    tmp_path, capsys, changed_evidence
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(_document(changed_evidence)) + "\n")
    assert _report(SimpleNamespace(documents=path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trainable_units_default"] == 1
    assert report["document_quality_status"] == {"accepted": 1}
    assert "quality_status" not in report
    assert "causal_validation" not in report
    assert report["detail_observation"] == {"present": 1}


def test_report_fails_closed_on_unknown_annotation_field(
    tmp_path, capsys, changed_evidence
) -> None:
    document = _document(changed_evidence)
    document["evidence_units"][0]["annotation"]["extra"] = True
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(document) + "\n")
    assert _report(SimpleNamespace(documents=path)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_units"] == 1


def test_report_validates_pending_annotation_contract(tmp_path, capsys) -> None:
    from video_harness.evidence import mock_evidence_record

    document = _document(mock_evidence_record())
    document["status"] = "planned"
    annotation = document["evidence_units"][0]["annotation"]
    annotation.update(
        {
            "schema_version": "wrong-schema",
            "status": "pending",
            "record": None,
            "provenance": None,
        }
    )
    annotate_boundaries(document, status="pending")
    set_document_quality(document, "pending")
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(document) + "\n")
    assert _report(SimpleNamespace(documents=path)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_units"] == 1
    assert "unexpected evidence schema" in report["invalid_examples"][0]["error"]


def test_build_output_directory_must_be_empty(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.jsonl").write_text("stale")
    try:
        _require_new_or_empty_directory(output)
    except FileExistsError as exc:
        assert "new or empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-empty output directory was accepted")
