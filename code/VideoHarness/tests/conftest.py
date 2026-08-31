from __future__ import annotations

import copy

import pytest

from video_harness.evidence import compose_evidence_record

_CALL2_RECORD = {
    "motion_summary": "The gripper transports the bread slice toward the toaster and releases it into the slot.",
    "before_boundary_observation": {
        "cam_high": "A bread slice rests beside the toaster.",
        "cam_left_wrist": "The bread slice is visible near the open gripper.",
        "cam_right_wrist": "The toaster remains visible beyond the right gripper.",
    },
    "after_boundary_observation": {
        "cam_high": "The bread slice is visible inside the toaster slot.",
        "cam_left_wrist": "The gripper is open after releasing the bread slice.",
        "cam_right_wrist": "The toaster slot contains the bread slice.",
    },
    "detail_observation": "The bread slice passes below the gripper tips into the slot.",
    "unit_interpretation": {
        "action_description": "The robot inserts the bread slice into the toaster slot.",
        "task_role": "This Evidence Unit loads one bread slice for the toasting task.",
    },
}

_ACCEPTED_EVIDENCE = compose_evidence_record(_CALL2_RECORD)


@pytest.fixture
def call2_record() -> dict:
    return copy.deepcopy(_CALL2_RECORD)


@pytest.fixture
def changed_evidence() -> dict:
    return copy.deepcopy(_ACCEPTED_EVIDENCE)
