import copy

import pytest

from video_harness.evidence import EVIDENCE_SCHEMA_VERSION, mock_evidence_record
from video_harness.renderer import (
    RENDER_PROFILES,
    render_evidence_text,
    render_interleaved,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import BEHAVIOR_DOCUMENT_SCHEMA_VERSION, plan_document


def _document(records: list[dict], *, status: str = "complete") -> dict:
    length = 25 * len(records) + 1
    source = EpisodeRecord(
        episode_index=0,
        task_index=0,
        task_instruction="Put bread into the toaster.",
        task_kind="benchmark",
        length=length,
        dataset_from_index=0,
        dataset_to_index=length,
        data_path="data/chunk-000/file-000.parquet",
        videos=(
            VideoSlice(
                key="observation.images.cam_high",
                path="videos/observation.images.cam_high/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=length / 25,
            ),
        ),
    )
    document = plan_document(source, build_id="test-build")
    assert len(document["guidance_units"]) == len(records)
    for unit, record in zip(document["guidance_units"], records):
        unit["annotation"] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": status,
            "record": record,
            "provenance": {
                "provider": "test",
                "model": "test",
                "prompt_version": "test",
            },
        }
    document["status"] = "mock-annotated" if status == "mock" else "annotated"
    return document


def test_all_profiles_are_derived_without_mutating_evidence(changed_evidence) -> None:
    original = copy.deepcopy(changed_evidence)
    outputs = {profile: render_evidence_text(changed_evidence, profile) for profile in RENDER_PROFILES}
    assert outputs["brief"] == "The bread slice is now inside the toaster slot."
    assert "Operation hint" not in outputs["instructional"]
    assert "Visible result" in outputs["instructional"]
    assert "manipulated_object=bread slice" in outputs["stage-card"]
    assert "Operation hypothesis" in outputs["stage-card"]
    assert changed_evidence == original


def test_renderer_reuses_shared_boundary_image(changed_evidence) -> None:
    second = copy.deepcopy(changed_evidence)
    second["visual_observation"] = {
        "before": "The bread slice is visible inside the toaster slot.",
        "after": "The toaster lever is visibly lowered beside the bread slice.",
        "change": "The toaster lever is now visibly lowered.",
        "support": "clear",
    }
    second["operation_hint"] = {
        "label": "press",
        "description": "Press the toaster lever toward its lowered state.",
        "support": "endpoint_change",
    }
    document = _document([changed_evidence, second])
    calls: list[int] = []

    def load(_document, frame_ref):
        calls.append(frame_ref["episode_frame_index"])
        return f"frame-{frame_ref['episode_frame_index']}"

    rendered = render_interleaved(document, load, profile="state-change")
    assert [item["type"] for item in rendered] == ["image", "text", "image", "text", "image"]
    assert calls == [0, 25, 50]


def test_renderer_rejects_mock_by_default_but_can_opt_in(changed_evidence) -> None:
    document = _document([changed_evidence], status="mock")
    with pytest.raises(ValueError, match="no usable evidence"):
        render_interleaved(document, lambda *_: b"frame")
    assert render_interleaved(document, lambda *_: b"frame", allow_mock=True)


def test_renderer_rejects_insufficient_and_ambiguous_by_default(changed_evidence) -> None:
    insufficient = _document([mock_evidence_record()])
    with pytest.raises(ValueError, match="no trainable visible change"):
        render_interleaved(insufficient, lambda *_: b"frame")

    changed_evidence["visual_observation"]["support"] = "ambiguous"
    ambiguous = _document([changed_evidence])
    with pytest.raises(ValueError, match="no trainable visible change"):
        render_interleaved(ambiguous, lambda *_: b"frame")
    assert render_interleaved(ambiguous, lambda *_: b"frame", allow_ambiguous=True)


def test_renderer_fails_closed_on_document_and_annotation_schema(changed_evidence) -> None:
    document = _document([changed_evidence])
    document["schema_version"] = "old-document-schema"
    with pytest.raises(ValueError, match="unexpected behavior document schema"):
        render_interleaved(document, lambda *_: b"frame")

    document["schema_version"] = BEHAVIOR_DOCUMENT_SCHEMA_VERSION
    document["guidance_units"][0]["annotation"]["schema_version"] = "old-evidence-schema"
    with pytest.raises(ValueError, match="unexpected evidence schema"):
        render_interleaved(document, lambda *_: b"frame")


def test_actuator_v0_separates_visual_facts_from_operation_inference(changed_evidence) -> None:
    assert "actuator-v0" in RENDER_PROFILES

    output = render_evidence_text(changed_evidence, "actuator-v0")

    assert output.startswith("Observed before:")
    assert "Observed after:" in output
    assert "Visible change:" in output
    assert "Relevant entities:" in output
    assert "manipulated_object=bread slice [grounding=visual_plus_task, support=clear]" in output
    assert "target_receptacle=toaster slot [grounding=visual_plus_task, support=clear]" in output
    assert "Operation inference [support=endpoint_plus_task_context]: insert —" in output
    assert "Visible end effector: right." in output
    assert "Unobserved details: motion_path, force, precise_pose, grasp_contact." in output
    assert "Put bread into the toaster." not in output
    assert "Operation hypothesis" not in output


def test_actuator_v0_is_deterministic_and_handles_missing_operation(changed_evidence) -> None:
    first = render_evidence_text(changed_evidence, "actuator-v0")
    second = render_evidence_text(copy.deepcopy(changed_evidence), "actuator-v0")

    assert first == second

    changed_evidence["operation_hint"] = None
    output = render_evidence_text(changed_evidence, "actuator-v0")
    assert "Operation inference [support=none recorded]: none recorded." in output
