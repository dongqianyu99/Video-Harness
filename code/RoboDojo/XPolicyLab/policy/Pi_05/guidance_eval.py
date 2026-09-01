"""Task-level Guide materialization for Pi_05 evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from openpi.models.guide_inputs import GuideInput
from openpi.models.guide_materializer import GuideMaterializerConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.guide_dataset import GuideCatalog
from openpi.training.guide_materialization_cache import CachedGuideResolver, open_guide_materialization_cache
from video_harness.reader import GuideDocument, GuideDocumentCatalog, build_guide_plan, load_guide_document_catalog


class TaskGuideSession:
    """Materialize the first accepted Guide for one task and reuse it."""

    def __init__(
        self,
        *,
        documents_root: Path,
        guide_materialization_cache_root: Path,
        max_boundaries: int,
        max_units: int,
        max_boundary_text_tokens: int,
        max_transition_text_tokens: int,
        boundary_num_queries: int = 8,
        transition_num_queries: int = 4,
        catalog: GuideDocumentCatalog | None = None,
        boundary_tokenizer: Any | None = None,
        transition_tokenizer: Any | None = None,
        plan_builder: Callable[..., Any] | None = None,
    ) -> None:
        for name, path in (
            ("documents_root", documents_root),
            ("guide_materialization_cache_root", guide_materialization_cache_root),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
            if not path.is_dir():
                raise FileNotFoundError(f"{name} directory does not exist: {path}")

        self._catalog = catalog or load_guide_document_catalog(documents_root)
        self._boundary_tokenizer = boundary_tokenizer or PaligemmaTokenizer(
            max_boundary_text_tokens
        )
        self._transition_tokenizer = transition_tokenizer or PaligemmaTokenizer(
            max_transition_text_tokens
        )
        self._materializer_config = GuideMaterializerConfig(
            max_boundaries=max_boundaries,
            max_units=max_units,
            max_boundary_text_tokens=max_boundary_text_tokens,
            max_transition_text_tokens=max_transition_text_tokens,
            boundary_num_queries=boundary_num_queries,
            transition_num_queries=transition_num_queries,
        )
        self._plan_builder = build_guide_plan if plan_builder is None else plan_builder
        self._guide_catalog = GuideCatalog.from_document_catalog(self._catalog)
        plans = {
            record.document_id: self._plan_builder(
                self._catalog, document_id=record.document_id
            )
            for record in self._guide_catalog.records
        }
        cache = open_guide_materialization_cache(
            cache_root=guide_materialization_cache_root,
            guide_records=self._guide_catalog.records,
            document_catalog=self._catalog,
            plans_by_document=plans,
            materializer_config=self._materializer_config,
        )
        self._resolver = CachedGuideResolver(
            guide_records=self._guide_catalog.records,
            artifact_records=cache.records,
            materializer_config=self._materializer_config,
        )
        self._guidance: GuideDocument | None = None
        self._guide: GuideInput | None = None
        self.materialization_count = 0

    @property
    def guidance(self) -> GuideDocument | None:
        return self._guidance

    @property
    def guide(self) -> GuideInput | None:
        return self._guide

    @property
    def catalog_digest(self) -> str:
        return self._catalog.catalog_digest

    @property
    def identity(self) -> dict[str, Any]:
        if self._guidance is None:
            return {
                "catalog_digest": self.catalog_digest,
                "document_id": None,
                "source_episode_index": None,
                "task_index": None,
            }
        return {
            "catalog_digest": self.catalog_digest,
            "document_id": self._guidance.document_id,
            "source_episode_index": self._guidance.source_episode_index,
            "task_index": self._guidance.task_index,
        }

    def bind_instruction(self, instruction: str) -> GuideInput:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("evaluation instruction must be a non-empty string")
        candidates = tuple(
            document
            for document in self._catalog.documents
            if document.task_instruction == instruction
        )
        if not candidates:
            raise ValueError(f"No accepted Guidance is available for task {instruction!r}")
        candidate_tasks = sorted({candidate.task_index for candidate in candidates})
        if len(candidate_tasks) != 1:
            raise ValueError(
                f"Evaluation task instruction {instruction!r} is ambiguous across "
                f"task_index values {candidate_tasks}"
            )
        guidance = candidates[0]
        if self._guidance is not None:
            if self._guidance.task_index != guidance.task_index:
                raise ValueError(
                    f"Guide session is already bound to task {self._guidance.task_index}; "
                    f"cannot switch to task {guidance.task_index}"
                )
            assert self._guide is not None
            return self._guide

        guide = self._resolver(
            self._guide_catalog.by_document_id(guidance.document_id)
        )
        self._guidance = guidance
        self._guide = guide
        self.materialization_count += 1
        return guide

    def clear(self) -> None:
        self._guidance = None
        self._guide = None
