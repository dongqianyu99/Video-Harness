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
            "a fixed, elevated, oblique view looking down toward the interaction "
            "workspace; "
            "it does not move with either robot arm"
        ),
        evidence_authority=(
            "global scene layout, entity positions, spatial relations, scene state, "
            "and world-relative displacement"
        ),
    ),
    CameraSpec(
        view_id="cam_left_wrist",
        role_id="WRIST_MOUNTED_LOCAL_LEFT",
        physical_view=(
            "an egocentric camera rigidly mounted behind the left gripper; it moves "
            "with the left arm and the left gripper is normally visible in the image"
        ),
        evidence_authority=(
            "local entity appearance and identity, left-gripper configuration, "
            "proximity, contact, and local interaction state"
        ),
    ),
    CameraSpec(
        view_id="cam_right_wrist",
        role_id="WRIST_MOUNTED_LOCAL_RIGHT",
        physical_view=(
            "an egocentric camera rigidly mounted behind the right gripper; it moves "
            "with the right arm and the right gripper is normally visible in the image"
        ),
        evidence_authority=(
            "local entity appearance and identity, right-gripper configuration, "
            "proximity, contact, and local interaction state"
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

All three cameras observe the same synchronized physical scene, but they do not have equal authority for every claim. Treat camera authority as claim-specific evidence priority, not as infallibility: any view may be occluded, cropped, or unresolved. Use cam_high primarily for global world-state, spatial-relation, and displacement claims. Use the corresponding wrist camera primarily for local identity, end-effector configuration, proximity, contact, and local interaction evidence. Wrist-camera pixel motion includes camera ego-motion: never infer global entity movement, transfer between end effectors, or a changed global scene configuration solely because an entity changes position, scale, or visibility in a moving wrist view. Not visible is not the same as absent or unchanged. Images with the same episode-frame label are synchronized observations of one moment. Do not treat pixel coordinates from different cameras as a shared coordinate system; instead, corroborate views qualitatively using synchronized time, entity identity, and end-effector association. Keep per-view observations separate and preserve uncertainty when evidence conflicts or remains occluded."""
