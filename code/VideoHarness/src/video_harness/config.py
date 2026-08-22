from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarnessConfig:
    debug: bool = False
    debug_root: Path | None = None
    inspection_retries: int = 1
    call2_max_attempts: int = 3
    fps: int = 25
    unit_frame_count: int = 26
    preprocessing_version: str = "video-harness.temporal-pack"

    def __post_init__(self) -> None:
        if self.debug and self.debug_root is None:
            raise ValueError("debug_root is required when debug mode is enabled")
        if not self.debug and self.debug_root is not None:
            raise ValueError("debug_root must be omitted when debug mode is disabled")
        for field in ("inspection_retries", "call2_max_attempts"):
            value = getattr(self, field)
            minimum = 0 if field == "inspection_retries" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if self.fps != 25 or self.unit_frame_count != 26:
            raise ValueError("Video Harness requires 25 Hz and 26-frame standard Units")
        if not self.preprocessing_version.strip():
            raise ValueError("preprocessing_version must be non-empty")

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["debug_root"] = None if self.debug_root is None else str(self.debug_root)
        return value
