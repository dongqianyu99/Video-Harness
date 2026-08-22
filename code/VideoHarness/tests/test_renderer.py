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
        videos=tuple(
            VideoSlice(
                key=key,
                path=f"videos/{key}/chunk-000/file-000.mp4",
                from_timestamp=0.0,
                to_timestamp=length / 25,
            )
            for key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
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
                "call1": {
                    "provider": "test",
                    "model": "test-motion",
                    "prompt_version": "test-inspection",
                },
                "call2": {
                    "provider": "test",
                    "model": "test-evidence",
                    "prompt_version": "test-evidence",
                    "attempts": 1,
                    "selected_attempt": 1,
                },
            },
        }
    document["status"] = "mock-annotated" if status == "mock" else "annotated"
    return document


def test_all_profiles_are_derived_without_mutating_evidence(changed_evidence) -> None:
    original = copy.deepcopy(changed_evidence)
    outputs = {profile: render_evidence_text(changed_evidence, profile) for profile in RENDER_PROFILES}
    assert outputs["brief"] == "The robot inserts the bread slice into the toaster slot."
    assert "Action:" in outputs["instructional"]
    assert "Task role:" in outputs["instructional"]
    assert "Causal validation: pass" in outputs["stage-card"]
    assert changed_evidence == original


def test_renderer_reuses_shared_boundary_image(changed_evidence) -> None:
    second = copy.deepcopy(changed_evidence)
    second["motion_summary"] = "The gripper approaches and lowers the toaster lever."
    second["endpoint_observation"]["after"]["cam_high"] = (
        "The toaster lever is visibly lowered beside the bread slice."
    )
    second["unit_interpretation"] = {
        "action_description": "The robot presses the toaster lever downward.",
        "task_role": "This Unit starts the toaster after loading the bread.",
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


def test_renderer_rejects_unknown_operation_by_default(changed_evidence) -> None:
    unknown = _document([mock_evidence_record()])
    with pytest.raises(ValueError, match="no trainable transition evidence"):
        render_interleaved(unknown, lambda *_: b"frame")


def test_renderer_fails_closed_on_document_and_annotation_schema(changed_evidence) -> None:
    document = _document([changed_evidence])
    document["schema_version"] = "wrong-document-schema"
    with pytest.raises(ValueError, match="unexpected behavior document schema"):
        render_interleaved(document, lambda *_: b"frame")

    document["schema_version"] = BEHAVIOR_DOCUMENT_SCHEMA_VERSION
    document["guidance_units"][0]["annotation"]["schema_version"] = "wrong-evidence-schema"
    with pytest.raises(ValueError, match="unexpected evidence schema"):
        render_interleaved(document, lambda *_: b"frame")


def test_actuator_profile_separates_visual_facts_from_operation_inference(changed_evidence) -> None:
    assert "actuator" in RENDER_PROFILES

    output = render_evidence_text(changed_evidence, "actuator")

    assert output.startswith("Motion:")
    assert "Before cam_high:" in output
    assert "Before cam_left_wrist:" in output
    assert "After cam_right_wrist:" in output
    assert "Detail:" in output
    assert "Action: The robot inserts" in output
    assert "Task role:" in output
    assert "Put bread into the toaster." not in output
    assert "Causal validation" not in output


def test_actuator_profile_is_deterministic(changed_evidence) -> None:
    first = render_evidence_text(changed_evidence, "actuator")
    second = render_evidence_text(copy.deepcopy(changed_evidence), "actuator")

    assert first == second

    assert "Action: The robot inserts" in first
