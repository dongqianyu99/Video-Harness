import json
from types import SimpleNamespace

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
        videos=(
            VideoSlice(
                key="observation.images.cam_high",
                path="videos/observation.images.cam_high/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=0.08,
            ),
        ),
    )
    document = plan_document(source, build_id="test-build")
    document["status"] = "annotated"
    document["guidance_units"][0]["annotation"] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "record": record,
        "provenance": {
            "provider": "test",
            "model": "test-model",
            "prompt_version": "test-prompt",
        },
    }
    return document


def test_report_counts_trainable_structured_evidence(tmp_path, capsys, changed_evidence) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(_document(changed_evidence)) + "\n")
    assert _report(SimpleNamespace(documents=path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trainable_units_default"] == 1
    assert report["entity_roles"] == {"manipulated_object": 1, "target_receptacle": 1}
    assert report["operation_labels"] == {"insert": 1}


def test_report_fails_closed_on_unknown_annotation_field(tmp_path, capsys, changed_evidence) -> None:
    document = _document(changed_evidence)
    document["guidance_units"][0]["annotation"]["extra"] = True
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(document) + "\n")
    assert _report(SimpleNamespace(documents=path)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_units"] == 1


def test_report_validates_pending_annotation_contract(tmp_path, capsys) -> None:
    document = _document({
        "change_status": "insufficient_visual_evidence",
        "visual_observation": {
            "before": None,
            "after": None,
            "change": None,
            "support": "insufficient",
        },
        "entities": [],
        "operation_hint": None,
        "visible_end_effector": "uncertain",
        "task_relevance": "uncertain",
        "visibility_limits": ["motion_path", "force", "precise_pose"],
    })
    document["status"] = "planned"
    annotation = document["guidance_units"][0]["annotation"]
    annotation.update(
        {
            "schema_version": "wrong-schema",
            "status": "pending",
            "record": None,
            "provenance": None,
        }
    )
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
