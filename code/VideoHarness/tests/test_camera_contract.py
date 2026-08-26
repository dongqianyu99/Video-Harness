import pytest

from video_harness.camera_contract import (
    CAMERA_BY_VIEW,
    CAMERA_VIEWS,
    image_label,
    system_prompt_camera_contract,
    validate_image_label,
)
from video_harness.prompts import INSPECTION_SYSTEM_PROMPT, SYSTEM_PROMPT


def test_camera_roles_are_explicit_and_stable() -> None:
    assert CAMERA_VIEWS == (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert CAMERA_BY_VIEW["cam_high"].role_id == "FIXED_GLOBAL"
    assert CAMERA_BY_VIEW["cam_left_wrist"].role_id == "WRIST_MOUNTED_LOCAL_LEFT"
    assert CAMERA_BY_VIEW["cam_right_wrist"].role_id == "WRIST_MOUNTED_LOCAL_RIGHT"


def test_every_provider_image_label_contains_view_role_and_evidence_role() -> None:
    label = image_label(
        evidence_role="BOUNDARY_BEFORE",
        view="cam_left_wrist",
        metadata="UNIT_FRAME=0",
    )
    validate_image_label(
        label,
        evidence_role="BOUNDARY_BEFORE",
        view="cam_left_wrist",
    )
    assert "CAMERA_ROLE=WRIST_MOUNTED_LOCAL_LEFT" in label
    with pytest.raises(ValueError, match="camera contract"):
        validate_image_label(
            label, evidence_role="BOUNDARY_AFTER", view="cam_left_wrist"
        )


def test_both_calls_share_the_camera_authority_contract() -> None:
    contract = system_prompt_camera_contract()
    for prompt in (contract, INSPECTION_SYSTEM_PROMPT, SYSTEM_PROMPT):
        assert "fixed, elevated, oblique view" in prompt
        assert "rigidly mounted behind the left gripper" in prompt
        assert "rigidly mounted behind the right gripper" in prompt
        assert "global scene layout" in prompt
        assert "local entity appearance and identity" in prompt
        assert "Wrist-camera pixel motion includes camera ego-motion" in prompt
        assert "Not visible is not the same as absent or unchanged" in prompt
    assert "never infer global entity movement" in SYSTEM_PROMPT
    assert "Do not treat pixel coordinates from different cameras" in SYSTEM_PROMPT
def test_shared_prompts_do_not_encode_task_specific_scene_vocabulary() -> None:
    prompts = (
        system_prompt_camera_contract(),
        INSPECTION_SYSTEM_PROMPT,
        SYSTEM_PROMPT,
    )
    banned_terms = (
        "pad occupancy",
        "tabletop",
        "button",
        "insertion",
        "toaster",
        "drawer",
        "cabinet",
    )
    for prompt in prompts:
        lowered = prompt.lower()
        assert all(term not in lowered for term in banned_terms)
