from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarnessConfig:
    debug: bool = False
    debug_root: Path | None = None
    inspection_retries: int = 1
    repair_max_attempts: int = 2
    sequence_audit_max_attempts: int = 2
    sequence_repair_rounds: int = 2
    provider_timeout_s: float = 300.0
    provider_max_retries: int = 2
    ffmpeg_timeout_s: float = 120.0
    fps: int = 25
    unit_frame_count: int = 26
    preprocessing_version: str = "video-harness.temporal-pack"

    def __post_init__(self) -> None:
        if self.debug and self.debug_root is None:
            raise ValueError("debug_root is required when debug mode is enabled")
        if not self.debug and self.debug_root is not None:
            raise ValueError("debug_root must be omitted when debug mode is disabled")
        for field in (
            "inspection_retries",
            "repair_max_attempts",
            "sequence_audit_max_attempts",
            "sequence_repair_rounds",
            "provider_max_retries",
        ):
            value = getattr(self, field)
            minimum = (
                0
                if field
                in {
                    "inspection_retries",
                    "sequence_repair_rounds",
                    "provider_max_retries",
                }
                else 1
            )
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        for field in ("provider_timeout_s", "ffmpeg_timeout_s"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{field} must be a positive number")
        if self.fps != 25 or self.unit_frame_count != 26:
            raise ValueError("Video Harness requires 25 Hz and 26-frame standard Units")
        if not self.preprocessing_version.strip():
            raise ValueError("preprocessing_version must be non-empty")

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["debug_root"] = None if self.debug_root is None else str(self.debug_root)
        return value
