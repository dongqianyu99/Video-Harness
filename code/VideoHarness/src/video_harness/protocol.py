from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePayload:
    """One labeled image supplied to a provider request."""

    label: str
    data: bytes
    media_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("image payload label must be non-empty")
        if not self.data:
            raise ValueError(f"image payload {self.label!r} must not be empty")
        if self.media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"unsupported image media type {self.media_type!r}")
