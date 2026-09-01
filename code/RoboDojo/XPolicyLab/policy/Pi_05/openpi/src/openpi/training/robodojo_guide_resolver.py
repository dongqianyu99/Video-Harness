from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.guide_materializer import materialize_guide
from openpi.training.guide_dataset import GuideRecord


def _default_plan_builder(catalog: Any, *, document_id: str) -> Any:
    reader = importlib.import_module("video_harness.reader")
    return reader.build_guide_plan(catalog, document_id=document_id)


def _default_frame_loader(dataset_root: Path) -> Any:
    media = importlib.import_module("video_harness.media")
    return media.FFmpegFrameLoader(dataset_root)


def _guide_context(record: GuideRecord) -> str:
    return (
        f"guide_index={record.guide_index}, document_id={record.document_id!r}, "
        f"source_episode_index={record.source_episode_index}"
    )


def _validate_source_identity(source: Any, record: GuideRecord) -> None:
    expected = {
        "document_id": record.document_id,
        "source_episode_index": record.source_episode_index,
        "task_index": record.task_index,
        "task_instruction": record.task_instruction,
    }
    for name, expected_value in expected.items():
        actual = getattr(source, name, None)
        if actual != expected_value:
            raise ValueError(f"canonical Guide Document {name} mismatch: expected {expected_value!r}, got {actual!r}")
    if getattr(source, "document", None) is None:
        raise ValueError("canonical Guide Document has no document payload")


def _validate_plan_identity(plan: Any, record: GuideRecord) -> None:
    expected = {
        "document_id": record.document_id,
        "source_episode_index": record.source_episode_index,
        "task_index": record.task_index,
        "task_instruction": record.task_instruction,
    }
    for name, expected_value in expected.items():
        actual = getattr(plan, name, None)
        if actual != expected_value:
            raise ValueError(f"GuidePlan {name} mismatch: expected {expected_value!r}, got {actual!r}")
    if not getattr(plan, "boundaries", None):
        raise ValueError("GuidePlan has no Boundaries")
    if not getattr(plan, "units", None):
        raise ValueError("GuidePlan has no accepted Units")


def _boundary_frame_refs(boundaries: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = []
    for boundary in boundaries:
        episode_frame_index = getattr(boundary, "episode_frame_index", None)
        timestamp_s = getattr(boundary, "timestamp_s", None)
        if (
            isinstance(episode_frame_index, bool)
            or not isinstance(episode_frame_index, (int, np.integer))
            or int(episode_frame_index) < 0
        ):
            raise ValueError("GuidePlan Boundary has invalid episode_frame_index")
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
            raise ValueError("GuidePlan Boundary has invalid timestamp_s")
        refs.append(
            {
                "episode_frame_index": int(episode_frame_index),
                "timestamp_s": float(timestamp_s),
            }
        )
    return tuple(refs)


def _validate_rgb_views(payload: Any, *, context: str) -> tuple[np.ndarray, ...]:
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        raise ValueError(f"{context}: decoder must return exactly three RGB views")
    views: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for view_index, value in enumerate(payload):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{context}: view {view_index} must be a numpy array, got {type(value).__name__}")
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{context}: view {view_index} must have RGB shape [H, W, 3], got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"{context}: view {view_index} must have dtype uint8, got {value.dtype}")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(f"{context}: all three views must share one RGB shape")
        views.append(value)
    return tuple(views)


def _load_validated_boundaries(
    *,
    frame_loader: Any,
    document: Any,
    boundaries: Sequence[Any],
    context: str,
) -> tuple[tuple[np.ndarray, ...], ...]:
    load_many = getattr(frame_loader, "load_views_rgb_many", None)
    if not callable(load_many):
        raise ValueError("frame_loader must provide load_views_rgb_many(document, frame_refs)")
    refs = _boundary_frame_refs(boundaries)
    payloads = tuple(load_many(document, refs))
    if len(payloads) != len(refs):
        raise ValueError(
            "load_views_rgb_many returned an unexpected number of Boundaries: "
            f"expected {len(refs)}, got {len(payloads)}"
        )
    return tuple(
        _validate_rgb_views(payload, context=f"{context}, Boundary {index}") for index, payload in enumerate(payloads)
    )


class VideoHarnessGuideResolver:
    """Resolve one task-level GuideRecord into a three-view GuideInput."""

    def __init__(
        self,
        *,
        document_catalog: Any,
        guide_records: Sequence[GuideRecord],
        dataset_root: Path,
        boundary_tokenizer: Any,
        transition_tokenizer: Any,
        materializer_config: GuideMaterializerConfig,
        plans_by_document: Mapping[str, Any] | None = None,
        frame_loader: Any | None = None,
        plan_builder: Callable[..., Any] | None = None,
    ):
        if not callable(getattr(boundary_tokenizer, "tokenize_text", None)):
            raise ValueError("boundary_tokenizer must provide tokenize_text(text)")
        if not callable(getattr(transition_tokenizer, "tokenize_text", None)):
            raise ValueError("transition_tokenizer must provide tokenize_text(text)")
        records = tuple(guide_records)
        if not records:
            raise ValueError("guide_records must not be empty")
        self._records_by_index: dict[int, GuideRecord] = {}
        for record in records:
            if not isinstance(record, GuideRecord):
                raise ValueError("guide_records must contain GuideRecord values")
            if record.guide_index in self._records_by_index:
                raise ValueError(f"duplicate guide_index={record.guide_index}")
            self._records_by_index[record.guide_index] = record
        self._document_catalog = document_catalog
        self._dataset_root = Path(dataset_root)
        self._boundary_tokenizer = boundary_tokenizer
        self._transition_tokenizer = transition_tokenizer
        self._materializer_config = materializer_config
        if plans_by_document is not None and plan_builder is not None:
            raise ValueError("provide plans_by_document or plan_builder, not both")
        self._plans_by_document = (
            None if plans_by_document is None else dict(plans_by_document)
        )
        if self._plans_by_document is not None and set(self._plans_by_document) != {
            record.document_id for record in records
        }:
            raise ValueError("plans_by_document must exactly match guide_records")
        self._frame_loader = _default_frame_loader(self._dataset_root) if frame_loader is None else frame_loader
        self._plan_builder = (
            _default_plan_builder if plan_builder is None else plan_builder
        )

    def __call__(self, record: GuideRecord) -> GuideInput:
        context = _guide_context(record)
        try:
            expected = self._records_by_index.get(record.guide_index)
            if expected is None or expected != record:
                raise ValueError("record is not the immutable record registered with the resolver")
            source = self._document_catalog.by_document_id(record.document_id)
            _validate_source_identity(source, record)
            plan = (
                self._plans_by_document[record.document_id]
                if self._plans_by_document is not None
                else self._plan_builder(
                    self._document_catalog,
                    document_id=record.document_id,
                )
            )
            _validate_plan_identity(plan, record)

            def decode_boundaries(boundaries: Sequence[Any]) -> tuple[np.ndarray, ...]:
                return tuple(
                    np.stack(views, axis=0)
                    for views in _load_validated_boundaries(
                        frame_loader=self._frame_loader,
                        document=source.document,
                        boundaries=boundaries,
                        context=context,
                    )
                )

            def decode_boundary(boundary: Any) -> np.ndarray:
                return decode_boundaries((boundary,))[0]

            guide = materialize_guide(
                plan,
                boundary_decoder=decode_boundary,
                boundaries_decoder=decode_boundaries,
                boundary_tokenizer=self._boundary_tokenizer,
                transition_tokenizer=self._transition_tokenizer,
                config=self._materializer_config,
            )
            if not isinstance(guide, GuideInput):
                raise ValueError(f"materialize_guide returned {type(guide).__name__}, expected GuideInput")
            return guide
        except Exception as exc:
            raise ValueError(f"VideoHarness Guide resolution failed for {context}: {exc}") from exc
