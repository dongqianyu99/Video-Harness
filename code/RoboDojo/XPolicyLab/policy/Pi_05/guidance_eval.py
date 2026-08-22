"""Task-level Guide materialization for Pi_05 evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig, materialize_guide
from openpi.models.tokenizer import PaligemmaTokenizer
from video_harness.eval_guidance import EvalGuidance, EvalGuidanceCatalog, load_eval_guidance_catalog
from video_harness.media import FFmpegFrameLoader


class TaskGuideSession:
    """Materialize one immutable dataset-first Guide and reuse it for a task."""

    def __init__(
        self,
        *,
        documents_path: Path,
        episodes_path: Path,
        dataset_root: Path,
        profile: str = "actuator",
        max_frames: int,
        max_units: int,
        max_text_tokens: int,
        catalog: EvalGuidanceCatalog | None = None,
        frame_loader: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        for name, path, directory in (
            ("documents_path", documents_path, False),
            ("episodes_path", episodes_path, False),
            ("dataset_root", dataset_root, True),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
            if directory and not path.is_dir():
                raise FileNotFoundError(f"{name} directory does not exist: {path}")
            if not directory and not path.is_file():
                raise FileNotFoundError(f"{name} file does not exist: {path}")

        self._catalog = catalog or load_eval_guidance_catalog(
            documents_path,
            episodes_path=episodes_path,
        )
        self._frame_loader = frame_loader or FFmpegFrameLoader(dataset_root)
        self._tokenizer = tokenizer or PaligemmaTokenizer(max_text_tokens)
        self._materializer_config = GuideMaterializerConfig(
            max_frames=max_frames,
            max_units=max_units,
            max_text_tokens=max_text_tokens,
        )
        self._profile = profile
        self._guidance: EvalGuidance | None = None
        self._guide: GuideInput | None = None
        self.materialization_count = 0

    @property
    def guidance(self) -> EvalGuidance | None:
        return self._guidance

    @property
    def guide(self) -> GuideInput | None:
        return self._guide

    def bind_instruction(self, instruction: str) -> GuideInput:
        guidance = self._catalog.resolve(task_instruction=instruction)
        if self._guidance is not None:
            if self._guidance.task_index != guidance.task_index:
                raise ValueError(
                    f"Guide session is already bound to task {self._guidance.task_index}; "
                    f"cannot switch to task {guidance.task_index}"
                )
            assert self._guide is not None
            return self._guide

        plan = self._catalog.build_plan(
            task_index=guidance.task_index,
            profile=self._profile,
        )

        def frame_decoder(frame_ref: Any):
            if frame_ref.document_id != guidance.document_id:
                raise ValueError("Eval Guide frame references a different document")
            if frame_ref.episode_index != guidance.source_episode_index:
                raise ValueError("Eval Guide frame references a different source episode")
            return self._frame_loader.load_rgb(
                guidance.document,
                {
                    "episode_frame_index": frame_ref.episode_frame_index,
                    "timestamp_s": frame_ref.timestamp_s,
                },
            )

        guide = materialize_guide(
            plan,
            frame_decoder=frame_decoder,
            tokenizer=self._tokenizer,
            config=self._materializer_config,
        )
        self._guidance = guidance
        self._guide = guide
        self.materialization_count += 1
        return guide

    def clear(self) -> None:
        self._guidance = None
        self._guide = None
