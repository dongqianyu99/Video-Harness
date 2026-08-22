from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSpec:
    view_id: str
    role_id: str
    physical_view: str
    evidence_authority: str


CAMERA_SPECS = (
    CameraSpec(
        view_id="cam_high",
        role_id="FIXED_GLOBAL",
        physical_view=(
            "a fixed, elevated, oblique view looking down toward the tabletop; "
            "it does not move with either robot arm"
        ),
        evidence_authority=(
            "global object position, pad occupancy, object count, ordering, scene "
            "state, and world-relative displacement"
        ),
    ),
    CameraSpec(
        view_id="cam_left_wrist",
        role_id="MOVING_LOCAL_LEFT_WRIST",
        physical_view=(
            "an egocentric camera rigidly mounted behind the left gripper; it moves "
            "with the left arm and the left gripper remains visible in the image"
        ),
        evidence_authority=(
            "local object identity, left-gripper opening or closing, local contact, "
            "grasp, and release"
        ),
    ),
    CameraSpec(
        view_id="cam_right_wrist",
        role_id="MOVING_LOCAL_RIGHT_WRIST",
        physical_view=(
            "an egocentric camera rigidly mounted behind the right gripper; it moves "
            "with the right arm and the right gripper remains visible in the image"
        ),
        evidence_authority=(
            "local object identity, right-gripper opening or closing, local contact, "
            "grasp, and release"
        ),
    ),
)

CAMERA_BY_VIEW = {spec.view_id: spec for spec in CAMERA_SPECS}
CAMERA_VIEWS = tuple(spec.view_id for spec in CAMERA_SPECS)


def camera_spec(view: str) -> CameraSpec:
    try:
        return CAMERA_BY_VIEW[view]
    except KeyError as exc:
        raise ValueError(f"unknown camera view {view!r}") from exc


def image_label(
    *,
    evidence_role: str,
    view: str,
    metadata: str,
) -> str:
    spec = camera_spec(view)
    return (
        f"EVIDENCE={evidence_role} | VIEW={spec.view_id} | "
        f"CAMERA_ROLE={spec.role_id} | {metadata}"
    )


def validate_image_label(label: str, *, evidence_role: str, view: str) -> None:
    spec = camera_spec(view)
    required = (
        f"EVIDENCE={evidence_role}",
        f"VIEW={spec.view_id}",
        f"CAMERA_ROLE={spec.role_id}",
    )
    if any(fragment not in label for fragment in required):
        raise ValueError(
            f"image label does not match {evidence_role}/{view} camera contract"
        )


def system_prompt_camera_contract() -> str:
    definitions = "\n".join(
        f"- {spec.view_id} [{spec.role_id}]: {spec.physical_view}. It is most "
        f"authoritative for {spec.evidence_authority}."
        for spec in CAMERA_SPECS
    )
    return f"""Camera contract:
{definitions}

All three cameras observe the same synchronized physical scene, but they do not have equal authority for every claim. Use cam_high for global world-state and displacement claims. Use the corresponding wrist camera for local identity, gripper state, and contact evidence. Wrist-camera pixel motion includes camera ego-motion: never infer global object movement, cross-arm transfer, or changed pad occupancy solely because an object changes position, scale, or visibility in a moving wrist view. Do not compare pixel coordinates across cameras. Use the views to corroborate one another while keeping per-view observations separate when evidence conflicts or remains occluded."""
