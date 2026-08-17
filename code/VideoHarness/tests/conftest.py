from __future__ import annotations

import copy

import pytest


_CHANGED_EVIDENCE = {
    "change_status": "changed",
    "visual_observation": {
        "before": "A bread slice rests on the table beside the toaster.",
        "after": "The bread slice is visible inside the toaster slot.",
        "change": "The bread slice is now inside the toaster slot.",
        "support": "clear",
    },
    "entities": [
        {
            "name": "bread slice",
            "visual_description": "A light-brown rectangular slice beside the toaster.",
            "role": "manipulated_object",
            "visible_in": "both",
            "grounding": "visual_plus_task",
            "support": "clear",
        },
        {
            "name": "toaster slot",
            "visual_description": "A dark slot on top of the toaster.",
            "role": "target_receptacle",
            "visible_in": "both",
            "grounding": "visual_plus_task",
            "support": "clear",
        },
    ],
    "operation_hint": {
        "label": "insert",
        "description": "Insert the bread slice into the toaster slot.",
        "support": "endpoint_plus_task_context",
    },
    "visible_end_effector": "right",
    "task_relevance": "relevant",
    "visibility_limits": ["motion_path", "force", "precise_pose", "grasp_contact"],
}


@pytest.fixture
def changed_evidence() -> dict:
    return copy.deepcopy(_CHANGED_EVIDENCE)
