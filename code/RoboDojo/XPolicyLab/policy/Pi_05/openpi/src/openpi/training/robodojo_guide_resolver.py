from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.guide_materializer import materialize_guide
from openpi.training.guide_dataset import GuideBindingIndex
from openpi.training.guide_dataset import GuideBindingRecord


@dataclass(frozen=True)
class RoboDojoGuideMaterializationConfig:
    """Explicit VideoHarness source and fixed Guide tensor budgets."""

    dataset_root: Path
    profile: str
    max_frames: int
    max_units: int
    max_text_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_root, Path):
            raise ValueError("dataset_root must be an explicit pathlib.Path")
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        for name in ("max_frames", "max_units", "max_text_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def to_materializer_config(self) -> GuideMaterializerConfig:
        return GuideMaterializerConfig(
            max_frames=self.max_frames,
            max_units=self.max_units,
            max_text_tokens=self.max_text_tokens,
        )


@dataclass(frozen=True)
class RoboDojoGuideResolverFactory:
    """Pickle-safe factory for one worker-local VideoHarness resolver."""

    dataset_artifact_path: Path
    documents_artifact_path: Path
    pairs_artifact_path: Path
    dataset_root: Path
    binding_records: tuple[GuideBindingRecord, ...]
    materializer_config: GuideMaterializerConfig
    materializer_configs_by_binding: tuple[
        tuple[int, GuideMaterializerConfig], ...
    ] = ()
    profile: str = "actuator"

    def __call__(self) -> VideoHarnessGuideResolver:
        reader = importlib.import_module("video_harness.reader")
        bundle = reader.load_guide_artifact_bundle(
            dataset_path=self.dataset_artifact_path,
            documents_path=self.documents_artifact_path,
            pairs_path=self.pairs_artifact_path,
        )
        tokenizer_module = importlib.import_module("openpi.models.tokenizer")
        tokenizer = tokenizer_module.PaligemmaTokenizer(
            self.materializer_config.max_text_tokens
        )
        binding_index = GuideBindingIndex.from_bindings(self.binding_records)
        return VideoHarnessGuideResolver(
            artifact_bundle=bundle,
            binding_index=binding_index,
            dataset_root=self.dataset_root,
            tokenizer=tokenizer,
            materializer_config=self.materializer_config,
            materializer_configs_by_binding=dict(
                self.materializer_configs_by_binding
            ),
            profile=self.profile,
        )


def _default_plan_builder(bundle: Any, *, query_episode_index: int, profile: str) -> Any:
    reader = importlib.import_module("video_harness.reader")
    return reader.build_guide_plan(
        bundle,
        query_episode_index=query_episode_index,
        profile=profile,
    )


def _default_frame_loader(dataset_root: Path) -> Any:
    media = importlib.import_module("video_harness.media")
    return media.FFmpegFrameLoader(dataset_root)


def _find_support_source(bundle: Any, document_id: str) -> Any:
    sources = getattr(bundle, "documents", None)
    if sources is None:
        raise ValueError("artifact bundle must provide canonical documents")

    matches = [source for source in sources if source.document_id == document_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one canonical document {document_id!r}, found {len(matches)}"
        )
    return matches[0]


def _validate_rgb_frame(payload: Any, *, context: str) -> np.ndarray:
    if not isinstance(payload, np.ndarray):
        raise ValueError(
            f"{context}: frame loader must return a numpy RGB array, "
            f"got {type(payload).__name__}"
        )
    if payload.ndim != 3 or payload.shape[-1] != 3:
        raise ValueError(
            f"{context}: decoded frame is not RGB [H, W, 3], got {payload.shape}"
        )
    if payload.shape[0] <= 0 or payload.shape[1] <= 0:
        raise ValueError(f"{context}: decoded RGB frame has an empty spatial dimension")
    if payload.dtype != np.uint8:
        raise ValueError(
            f"{context}: decoded RGB frame must have dtype uint8, got {payload.dtype}"
        )
    return np.array(payload, copy=True)


def _load_frame_payload(frame_loader: Any, document: Any, frame_ref: dict[str, Any]) -> Any:
    load_rgb = getattr(frame_loader, "load_rgb", None)
    if callable(load_rgb):
        return load_rgb(document, frame_ref)
    load = getattr(frame_loader, "load", None)
    if callable(load):
        return load(document, frame_ref)
    if callable(frame_loader):
        return frame_loader(document, frame_ref)
    raise ValueError(
        "frame_loader must provide load_rgb(document, frame_ref), "
        "load(document, frame_ref), or be callable"
    )


class VideoHarnessGuideResolver:
    """Resolve one static binding into a materialized GuideInput."""

    def __init__(
        self,
        *,
        artifact_bundle: Any,
        binding_index: GuideBindingIndex,
        dataset_root: Path,
        tokenizer: Any,
        materializer_config: GuideMaterializerConfig,
        materializer_configs_by_binding: Mapping[
            int, GuideMaterializerConfig
        ] | None = None,
        profile: str = "actuator",
        frame_loader: Any | None = None,
        plan_builder: Callable[..., Any] | None = None,
    ):
        if not isinstance(binding_index, GuideBindingIndex):
            raise ValueError("binding_index must be a GuideBindingIndex")
        if not callable(getattr(tokenizer, "tokenize_text", None)):
            raise ValueError("tokenizer must provide tokenize_text(text)")
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError("profile must be a non-empty string")

        self._artifact_bundle = artifact_bundle
        self._binding_index = binding_index
        self._dataset_root = Path(dataset_root)
        self._tokenizer = tokenizer
        self._materializer_config = materializer_config
        self._materializer_configs_by_binding = (
            {}
            if materializer_configs_by_binding is None
            else dict(materializer_configs_by_binding)
        )
        self._profile = profile
        self._frame_loader = (
            _default_frame_loader(self._dataset_root)
            if frame_loader is None
            else frame_loader
        )
        self._plan_builder = _default_plan_builder if plan_builder is None else plan_builder

    def __call__(self, record: GuideBindingRecord) -> GuideInput:
        context = (
            f"binding_index={record.binding_index}, "
            f"query_episode_index={record.query_episode_index}, "
            f"support_document_id={record.support_document_id!r}"
        )

        try:
            expected_record = self._binding_index.by_binding_index(record.binding_index)
            if expected_record != record:
                raise ValueError("record is not the immutable record from binding_index")

            plan = self._plan_builder(
                self._artifact_bundle,
                query_episode_index=record.query_episode_index,
                profile=self._profile,
            )
            self._validate_plan_identity(plan, record)
            source = _find_support_source(
                self._artifact_bundle,
                record.support_document_id,
            )
            self._validate_source_identity(source, record, plan)

            document = source.document

            def checked_frame_dict(frame_ref: Any) -> tuple[dict[str, Any], str]:
                frame_context = f"{context}, unit/frame={frame_ref!r}"
                if frame_ref.document_id != record.support_document_id:
                    raise ValueError(
                        f"{frame_context}: frame references a different document"
                    )
                if frame_ref.episode_index != record.support_episode_index:
                    raise ValueError(
                        f"{frame_context}: frame references a different support episode"
                    )

                episode_frame_index = frame_ref.episode_frame_index
                timestamp_s = frame_ref.timestamp_s
                if (
                    isinstance(episode_frame_index, bool)
                    or not isinstance(episode_frame_index, (int, np.integer))
                    or int(episode_frame_index) < 0
                ):
                    raise ValueError(f"{frame_context}: invalid episode_frame_index")
                if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
                    raise ValueError(f"{frame_context}: invalid timestamp_s")

                frame_dict = {
                    "episode_frame_index": int(episode_frame_index),
                    "timestamp_s": float(timestamp_s),
                }
                return frame_dict, frame_context

            def frame_decoder(frame_ref: Any) -> np.ndarray:
                frame_dict, frame_context = checked_frame_dict(frame_ref)
                payload = _load_frame_payload(
                    self._frame_loader,
                    document,
                    frame_dict,
                )
                return _validate_rgb_frame(payload, context=frame_context)

            frames_decoder = None
            load_rgb_many = getattr(self._frame_loader, "load_rgb_many", None)
            if callable(load_rgb_many):
                def decode_many(frame_refs: Sequence[Any]) -> tuple[np.ndarray, ...]:
                    checked = [checked_frame_dict(frame_ref) for frame_ref in frame_refs]
                    payloads = tuple(
                        load_rgb_many(
                            document,
                            [frame_dict for frame_dict, _ in checked],
                        )
                    )
                    if len(payloads) != len(checked):
                        raise ValueError(
                            "load_rgb_many returned an unexpected number of frames: "
                            f"expected {len(checked)}, got {len(payloads)}"
                        )
                    return tuple(
                        _validate_rgb_frame(payload, context=frame_context)
                        for payload, (_, frame_context) in zip(
                            payloads, checked, strict=True
                        )
                    )

                frames_decoder = decode_many

            guide = materialize_guide(
                plan,
                frame_decoder=frame_decoder,
                frames_decoder=frames_decoder,
                tokenizer=self._tokenizer,
                config=self._materializer_configs_by_binding.get(
                    record.binding_index, self._materializer_config
                ),
            )
            if not isinstance(guide, GuideInput):
                raise ValueError(
                    f"materialize_guide returned {type(guide).__name__}, expected GuideInput"
                )
            return guide
        except Exception as exc:
            raise ValueError(f"VideoHarness Guide resolution failed for {context}: {exc}") from exc

    @staticmethod
    def _validate_plan_identity(plan: Any, record: GuideBindingRecord) -> None:
        expected = {
            "query_episode_index": record.query_episode_index,
            "support_document_id": record.support_document_id,
            "support_episode_index": record.support_episode_index,
            "task_index": record.task_index,
        }
        for field, expected_value in expected.items():
            actual_value = getattr(plan, field, None)
            if actual_value != expected_value:
                raise ValueError(
                    f"GuidePlan {field} mismatch: expected {expected_value!r}, got {actual_value!r}"
                )

        units = getattr(plan, "units", None)
        if not units:
            raise ValueError("GuidePlan has no trainable units")

    @staticmethod
    def _validate_source_identity(
        source: Any,
        record: GuideBindingRecord,
        plan: Any,
    ) -> None:
        expected = {
            "document_id": record.support_document_id,
            "episode_index": record.support_episode_index,
            "task_index": record.task_index,
        }
        for field, expected_value in expected.items():
            actual_value = getattr(source, field, None)
            if actual_value != expected_value:
                raise ValueError(
                    f"canonical support document {field} mismatch: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
        if source.document is None:
            raise ValueError("canonical support document has no document payload")
        if plan.support_document_id != source.document_id:
            raise ValueError("GuidePlan and canonical support document disagree")
