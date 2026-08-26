from __future__ import annotations

from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from video_harness.hdf5_source import (
    HDF5_VIEW_KEYS,
    decode_hdf5_frames,
    hdf5_document_source,
    inspect_hdf5_episode,
    load_hdf5_jpeg,
)
from video_harness.sampling import plan_document_from_source, validate_document
from video_harness.temporal_media import TemporalMediaBuilder


def _stored_jpeg(rgb: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(rgb[..., ::-1]).save(output, format="JPEG", quality=100)
    return output.getvalue()


def _episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.zeros((480, 640, 3), dtype=np.uint8)
    first[..., :] = (230, 170, 20)
    second = np.zeros((480, 640, 3), dtype=np.uint8)
    second[..., :] = (20, 90, 220)
    encoded = [_stored_jpeg(first), _stored_jpeg(second)]
    width = max(map(len, encoded))
    with h5py.File(path, "w") as file:
        file.create_dataset("data_format_version", data="v1.0")
        file.create_dataset("instruction", data="Move the object.")
        file.create_dataset("additional_info/frequency", data=25)
        for key in HDF5_VIEW_KEYS.values():
            group = file.require_group(key.rsplit("/", 1)[0])
            group.create_dataset("shape", data=np.array([480, 640, 3]))
            group.create_dataset("colors", data=np.asarray(encoded, dtype=f"S{width}"))
    return first, second


def _document(path: Path) -> dict:
    episode = inspect_hdf5_episode(path)
    return plan_document_from_source(
        build_id="hdf5-test",
        document_id="robodojo-hdf5/test/episode-0000000",
        source=hdf5_document_source(episode),
        task_instruction=episode.task_instruction,
        sample_hz=1,
    )


def test_hdf5_episode_plans_a_valid_standalone_document(tmp_path: Path) -> None:
    path = (
        tmp_path / "arrange_largest_number" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    _episode(path)
    episode = inspect_hdf5_episode(path)

    assert episode.episode_index == 0
    assert episode.task_name == "arrange_largest_number"
    assert episode.task_instruction == "Move the object."
    assert episode.length == 2
    assert episode.fps == 25
    document = _document(path)
    assert validate_document(document) == document
    assert document["source"]["hdf5_path"] == path.name


def test_hdf5_frames_restore_lerobot_rgb_order(tmp_path: Path) -> None:
    path = tmp_path / "episode_0000000.hdf5"
    first, second = _episode(path)
    frames = decode_hdf5_frames(
        path,
        HDF5_VIEW_KEYS,
        start=0,
        end=1,
        fps=25,
    )

    for view in HDF5_VIEW_KEYS:
        np.testing.assert_allclose(frames[view][0], first, atol=3)
        np.testing.assert_allclose(frames[view][1], second, atol=3)
    jpeg = load_hdf5_jpeg(path, HDF5_VIEW_KEYS["cam_high"], 0)
    decoded = np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"))
    np.testing.assert_allclose(decoded, first, atol=5)


def test_temporal_media_builder_uses_hdf5_without_video_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode_0000000.hdf5"
    _episode(path)
    document = _document(path)

    base = TemporalMediaBuilder(tmp_path).build_base(
        document, document["evidence_units"][0]
    )

    assert base.unit_frames.frame_count == 2
    assert base.unit_frames.image_shape == (480, 640, 3)
    assert len(base.overviews) == 3
    assert len(base.keyframe_sheets) == 3
    assert len(base.boundary_images) == 6
