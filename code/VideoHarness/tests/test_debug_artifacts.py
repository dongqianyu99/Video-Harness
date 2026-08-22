import json
from pathlib import Path

import pytest

from video_harness.debug_artifacts import DebugArtifactStore


def test_disabled_store_performs_no_filesystem_writes(tmp_path: Path) -> None:
    store = DebugArtifactStore(
        enabled=False,
        root=None,
        document_id="document/one",
        unit_id="u0000",
    )
    store.write_bytes("frames/a.jpg", b"jpeg")
    store.write_json("call1.json", {"ok": True})
    store.write_many({"sheets/a.png": b"png"})
    assert store.finalize({"status": "complete"}) is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_store_writes_scoped_artifacts_and_manifest(tmp_path: Path) -> None:
    store = DebugArtifactStore(
        enabled=True,
        root=tmp_path,
        document_id="robodojo/episode-0000001",
        unit_id="u0000",
    )
    store.write_bytes("frames/cam_high/frame-00.jpg", b"jpeg")
    store.write_json("call1.json", {"needs_detail": False})
    root = store.finalize({"status": "complete"})

    assert root is not None
    assert (root / "frames/cam_high/frame-00.jpg").read_bytes() == b"jpeg"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert [item["path"] for item in manifest["artifacts"]] == [
        "frames/cam_high/frame-00.jpg",
        "call1.json",
    ]
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])


def test_debug_store_rejects_unsafe_paths_and_nonempty_reruns(tmp_path: Path) -> None:
    store = DebugArtifactStore(
        enabled=True,
        root=tmp_path,
        document_id="document",
        unit_id="unit",
    )
    with pytest.raises(ValueError, match="unsafe"):
        store.write_bytes("../escape", b"x")
    store.write_bytes("artifact.bin", b"x")
    with pytest.raises(FileExistsError, match="not empty"):
        DebugArtifactStore(
            enabled=True,
            root=tmp_path,
            document_id="document",
            unit_id="unit",
        )
