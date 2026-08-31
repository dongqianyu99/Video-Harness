from __future__ import annotations

from typing import Any

from video_harness.evidence import BOUNDARY_STATE_SCHEMA_VERSION


def boundary_observation(label: str) -> dict[str, str]:
    return {
        "cam_high": f"{label} from the fixed global view.",
        "cam_left_wrist": f"{label} from the left wrist view.",
        "cam_right_wrist": f"{label} from the right wrist view.",
    }


def annotate_boundaries(
    document: dict[str, Any],
    *,
    status: str = "complete",
) -> None:
    for order, boundary in enumerate(document["boundary_states"]):
        if status in {"pending", "failed"}:
            boundary["annotation"] = {
                "schema_version": BOUNDARY_STATE_SCHEMA_VERSION,
                "status": status,
                "record": None,
                "provenance": None,
            }
            continue
        role = "before" if order == 0 else "after"
        source_order = order if role == "before" else order - 1
        boundary["annotation"] = {
            "schema_version": BOUNDARY_STATE_SCHEMA_VERSION,
            "status": status,
            "record": {
                "observation": boundary_observation(f"Boundary {order}"),
            },
            "provenance": {
                "provider": "test",
                "model": "test-evidence",
                "prompt_version": "test-evidence",
                "source_unit_id": f"u{source_order:04d}",
                "boundary_role": role,
            },
        }


def set_document_quality(document: dict[str, Any], status: str) -> None:
    document["quality_status"] = status
    document["quality_provenance"] = (
        None
        if status == "pending"
        else {
            "provider": "test",
            "model": "test-auditor",
            "prompt_version": "video-harness.sequence-audit",
            "audit_attempts": 1,
            "repair_rounds": 0,
            "sequence_sha256": "0" * 64,
            "issues": [],
        }
    )
