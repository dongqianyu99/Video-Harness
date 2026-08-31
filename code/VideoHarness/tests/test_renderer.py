import copy

from _support import annotate_boundaries, set_document_quality
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.renderer import (
    render_boundary_view_texts,
    render_transition_text,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document


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
    annotate_boundaries(document, status=status)
    assert len(document["evidence_units"]) == len(records)
    for unit, record in zip(document["evidence_units"], records):
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
                },
                "repair": None,
            },
        }
    document["status"] = "mock-annotated" if status == "mock" else "annotated"
    set_document_quality(
        document,
        "quarantined" if status == "mock" else "accepted",
    )
    return document


def test_transition_renderer_is_complete_and_does_not_mutate_evidence(
    changed_evidence,
) -> None:
    original = copy.deepcopy(changed_evidence)
    output = render_transition_text(changed_evidence)

    assert output.startswith("Motion:")
    assert "Detail:" in output
    assert "Action:" in output
    assert "Task role:" in output
    assert "Causal validation" not in output
    assert changed_evidence == original


def test_boundary_view_texts_use_canonical_camera_order(changed_evidence) -> None:
    document = _document([changed_evidence])
    observation = document["boundary_states"][0]["annotation"]["record"][
        "observation"
    ]

    assert render_boundary_view_texts(
        document["boundary_states"][0]["annotation"]["record"]
    ) == (
        observation["cam_high"],
        observation["cam_left_wrist"],
        observation["cam_right_wrist"],
    )


def test_transition_renderer_is_deterministic(changed_evidence) -> None:
    first = render_transition_text(changed_evidence)
    second = render_transition_text(copy.deepcopy(changed_evidence))

    assert first == second

    assert "Action: The robot inserts" in first
