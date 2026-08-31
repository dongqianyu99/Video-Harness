from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

SEQUENCE_AUDIT_REPORT_SCHEMA_VERSION = "video-harness.sequence-audit-report.v1"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError("debug artifact identifier cannot be empty")
    return slug


def sequence_audit_report_path(
    *, enabled: bool, root: Path | None, document_id: str
) -> Path | None:
    if not enabled:
        return None
    if root is None:
        raise ValueError("enabled sequence audit report requires an explicit root")
    report_root = Path(root) / _slug(document_id) / "sequence-audit"
    report_root.mkdir(parents=True, exist_ok=True)
    index = 0
    while (candidate := report_root / f"run-{index:03d}.json").exists():
        index += 1
    return candidate


def write_sequence_audit_report(path: Path | None, value: Any) -> None:
    if path is None:
        return
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(encoded)
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DebugArtifactStore:
    """Optional per-Evidence-Unit filesystem sink; disabled mode performs no writes."""

    def __init__(
        self,
        *,
        enabled: bool,
        root: Path | None,
        document_id: str,
        unit_id: str,
    ) -> None:
        self.enabled = enabled
        self._entries: list[dict[str, Any]] = []
        self.root: Path | None = None
        if not enabled:
            if root is not None:
                raise ValueError("disabled debug store must not receive a root")
            return
        if root is None:
            raise ValueError("enabled debug store requires an explicit root")
        unit_root = Path(root) / _slug(document_id) / _slug(unit_id)
        if unit_root.exists() and any(unit_root.iterdir()):
            raise FileExistsError(
                f"debug Evidence Unit directory is not empty: {unit_root}"
            )
        unit_root.mkdir(parents=True, exist_ok=True)
        self.root = unit_root

    def _target(self, relative: str) -> Path:
        if self.root is None:
            raise RuntimeError("debug store is disabled")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe debug artifact path: {relative}")
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_bytes(self, relative: str, data: bytes) -> None:
        if not self.enabled:
            return
        target = self._target(relative)
        target.write_bytes(data)
        self._entries.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    def write_json(self, relative: str, value: Any) -> None:
        encoded = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        self.write_bytes(relative, encoded)

    def write_many(self, artifacts: dict[str, bytes]) -> None:
        for relative, data in sorted(artifacts.items()):
            self.write_bytes(relative, data)

    def finalize(self, metadata: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        manifest = {**metadata, "artifacts": list(self._entries)}
        self.write_json("manifest.json", manifest)
        return self.root
