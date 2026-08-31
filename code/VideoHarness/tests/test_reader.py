from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
from _support import annotate_boundaries, set_document_quality
from video_harness.evidence import EVIDENCE_SCHEMA_VERSION
from video_harness.reader import (
    GuideDocumentCatalog,
    GuidePlan,
    build_guide_plan,
    load_guide_document_catalog,
)
from video_harness.robodojo import EpisodeRecord, VideoSlice
from video_harness.sampling import plan_document


def _source(
    episode_index: int,
    *,
    task_index: int = 3,
    task_instruction: str = "Put bread into the toaster.",
    length: int = 51,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=episode_index,
        task_index=task_index,
        task_instruction=task_instruction,
        task_kind="benchmark",
        length=length,
        dataset_from_index=episode_index * length,
        dataset_to_index=(episode_index + 1) * length,
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


def _document(
    episode_index: int,
    changed_evidence: dict,
    *,
    build_id: str = "catalog-build",
    task_index: int = 3,
    task_instruction: str = "Put bread into the toaster.",
    length: int = 51,
    quality_status: str = "accepted",
) -> dict:
    document = plan_document(
        _source(
            episode_index,
            task_index=task_index,
            task_instruction=task_instruction,
            length=length,
        ),
        build_id=build_id,
    )
    annotate_boundaries(document)
    for unit in document["evidence_units"]:
        unit["annotation"] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "complete",
            "record": copy.deepcopy(changed_evidence),
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
    document["status"] = "annotated"
    set_document_quality(document, quality_status)
    return document


def _write_documents(root: Path, documents: list[dict]) -> None:
    for position, document in enumerate(documents):
        task_root = root / f"arbitrary-folder-{position:02d}"
        task_root.mkdir(parents=True, exist_ok=True)
        path = task_root / (
            f"episode-{document['source']['episode_index']:07d}.document.jsonl"
        )
        path.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_catalog_is_document_only_stable_and_filters_quarantine(
    tmp_path: Path, changed_evidence
) -> None:
    accepted_late = _document(12, changed_evidence)
    accepted_early = _document(10, changed_evidence)
    quarantined = _document(
        11, changed_evidence, quality_status="quarantined"
    )
    root = tmp_path / "documents"
    _write_documents(root, [accepted_late, quarantined, accepted_early])

    catalog = load_guide_document_catalog(root)

    assert isinstance(catalog, GuideDocumentCatalog)
    assert catalog.build_id == "catalog-build"
    assert len(catalog.catalog_digest) == 64
    assert [entry.source_episode_index for entry in catalog.documents] == [10, 12]
    assert [entry.source_episode_index for entry in catalog.exclusions] == [11]
    assert catalog.documents_for_task(3) == catalog.documents
    assert {entry.task_instruction for entry in catalog.documents} == {
        "Put bread into the toaster."
    }
    assert catalog.by_document_id(
        "robodojo/episode-0000010"
    ).document_path.is_absolute()

    second_root = tmp_path / "documents-second"
    _write_documents(second_root, [accepted_early, accepted_late, quarantined])
    assert (
        load_guide_document_catalog(second_root).catalog_digest
        == catalog.catalog_digest
    )


def test_catalog_is_deeply_read_only(tmp_path: Path, changed_evidence) -> None:
    root = tmp_path / "documents"
    _write_documents(root, [_document(10, changed_evidence)])
    catalog = load_guide_document_catalog(root)

    with pytest.raises(dataclasses.FrozenInstanceError):
        catalog.build_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        catalog.documents[0].document["build_id"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("build", "build_id"),
        ("duplicate", "Duplicate document_id"),
        ("instruction", "inconsistent task instructions"),
        ("pending", "nonterminal"),
    ],
)
def test_catalog_fails_closed_on_cross_document_drift(
    tmp_path: Path, changed_evidence, mutation: str, message: str
) -> None:
    first = _document(10, changed_evidence)
    second = _document(11, changed_evidence)
    if mutation == "build":
        second["build_id"] = "other-build"
    elif mutation == "duplicate":
        second = copy.deepcopy(first)
    elif mutation == "instruction":
        second["task_instruction"] = "A different instruction."
    else:
        second = plan_document(_source(11), build_id="catalog-build")

    root = tmp_path / "documents"
    _write_documents(root, [first, second])
    with pytest.raises(ValueError, match=message):
        load_guide_document_catalog(root)


def test_catalog_allows_one_instruction_to_name_multiple_tasks(
    tmp_path: Path,
    changed_evidence,
) -> None:
    first = _document(10, changed_evidence)
    second = _document(11, changed_evidence)
    second["source"]["task_index"] = 4
    root = tmp_path / "documents"
    _write_documents(root, [first, second])

    catalog = load_guide_document_catalog(root)

    candidates = tuple(
        document
        for document in catalog.documents
        if document.task_instruction == first["task_instruction"]
    )
    assert [document.task_index for document in candidates] == [3, 4]


def test_catalog_reports_malformed_or_multi_record_files(
    tmp_path: Path, changed_evidence
) -> None:
    root = tmp_path / "documents"
    _write_documents(root, [_document(10, changed_evidence)])
    path = next(root.glob("*/*.document.jsonl"))
    path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1"):
        load_guide_document_catalog(root)

    path.write_text(
        json.dumps(_document(10, changed_evidence))
        + "\n"
        + json.dumps(_document(11, changed_evidence))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one record"):
        load_guide_document_catalog(root)


def test_catalog_requires_at_least_one_accepted_document(
    tmp_path: Path, changed_evidence
) -> None:
    root = tmp_path / "documents"
    _write_documents(
        root, [_document(10, changed_evidence, quality_status="quarantined")]
    )
    with pytest.raises(ValueError, match="no accepted"):
        load_guide_document_catalog(root)


def test_plan_projects_shared_boundaries_and_current_semantics(
    tmp_path: Path, changed_evidence
) -> None:
    document = _document(10, changed_evidence)
    original = copy.deepcopy(document)
    root = tmp_path / "documents"
    _write_documents(root, [document])
    catalog = load_guide_document_catalog(root)

    plan = build_guide_plan(
        catalog,
        document_id="robodojo/episode-0000010",
    )

    assert isinstance(plan, GuidePlan)
    assert plan.document_id == "robodojo/episode-0000010"
    assert plan.source_episode_index == 10
    assert [boundary.slot for boundary in plan.boundaries] == [0, 1, 2]
    assert [boundary.episode_frame_index for boundary in plan.boundaries] == [
        0,
        25,
        50,
    ]
    assert [boundary.timestamp_s for boundary in plan.boundaries] == [0.0, 1.0, 2.0]
    assert all(len(boundary.view_texts) == 3 for boundary in plan.boundaries)
    assert plan.boundaries[0].view_texts == (
        document["boundary_states"][0]["annotation"]["record"]["observation"][
            "cam_high"
        ],
        document["boundary_states"][0]["annotation"]["record"]["observation"][
            "cam_left_wrist"
        ],
        document["boundary_states"][0]["annotation"]["record"]["observation"][
            "cam_right_wrist"
        ],
    )
    assert [(unit.before_slot, unit.after_slot) for unit in plan.units] == [
        (0, 1),
        (1, 2),
    ]
    transition = plan.units[0].transition_text
    assert transition.startswith("Motion:")
    assert "Detail:" in transition
    assert "Action:" in transition
    assert "Task role:" in transition
    assert "Causal validation" not in transition
    assert "reason" not in transition.lower()
    assert document == original

    plan_fields = {field.name for field in dataclasses.fields(plan)}
    assert not {
        "query_episode_index",
        "support_document_id",
        "profile",
        "provenance",
    } & plan_fields
    unit_fields = {field.name for field in dataclasses.fields(plan.units[0])}
    assert "provenance" not in unit_fields


def test_plan_rejects_unknown_or_excluded_document(
    tmp_path: Path, changed_evidence
) -> None:
    root = tmp_path / "documents"
    accepted = _document(10, changed_evidence)
    excluded = _document(11, changed_evidence, quality_status="quarantined")
    _write_documents(root, [accepted, excluded])
    catalog = load_guide_document_catalog(root)

    for document_id in (excluded["document_id"], "missing"):
        with pytest.raises(ValueError, match="No accepted"):
            build_guide_plan(catalog, document_id=document_id)
