from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .annotations import (
    AnnotationError,
    EvidenceBackend,
    EvidenceRequest,
    EvidenceResult,
    InspectionBackend,
    InspectionRequest,
    InspectionResult,
    RepairBackend,
    RepairRequest,
    RepairResult,
)
from .config import HarnessConfig
from .debug_artifacts import DebugArtifactStore
from .evidence import (
    compose_boundary_state_record,
    compose_evidence_record,
)
from .media import FrameDecodeError
from .run_tracking import ApiCallBudgetExceeded
from .sampling import unit_boundary_states
from .temporal_media import (
    BaseMedia,
    DetailRequest,
    TemporalMediaBuilder,
    TemporalMediaError,
    validate_detail_request,
)


@dataclass(frozen=True)
class EvidenceUnitPipelineResult:
    evidence: EvidenceResult
    before_boundary_record: dict[str, Any] | None
    after_boundary_record: dict[str, Any] | None
    inspection: InspectionResult
    detail_status: str
    debug_root: str | None


@dataclass(frozen=True)
class RepairOutcome:
    evidence: EvidenceResult | None
    before_boundary_record: dict[str, Any] | None
    after_boundary_record: dict[str, Any] | None
    attempts: int
    result: RepairResult | None
    error: str | None = None


@dataclass(frozen=True)
class ExistingRepairOutcome:
    canonical_evidence: dict[str, Any] | None
    before_boundary_record: dict[str, Any] | None
    after_boundary_record: dict[str, Any] | None
    attempts: int
    result: RepairResult | None
    error: str | None = None


class EvidenceUnitPipeline:
    def __init__(
        self,
        *,
        inspection_backend: InspectionBackend,
        evidence_backend: EvidenceBackend,
        media_builder: TemporalMediaBuilder,
        config: HarnessConfig,
        repair_backend: RepairBackend | None = None,
    ) -> None:
        self.inspection_backend = inspection_backend
        self.evidence_backend = evidence_backend
        self.repair_backend = repair_backend
        self.media_builder = media_builder
        self.config = config

    def _retry_media(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.config.media_retries + 1):
            try:
                return operation()
            except (FrameDecodeError, TemporalMediaError, OSError):
                if attempt < self.config.media_retries:
                    continue
                raise
        raise AssertionError("unreachable media retry state")

    def _store(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
    ) -> DebugArtifactStore:
        return DebugArtifactStore(
            enabled=self.config.debug,
            root=self.config.debug_root,
            document_id=document["document_id"],
            unit_id=unit["unit_id"],
        )

    def _inspect(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
        base: BaseMedia,
    ) -> tuple[InspectionResult, DetailRequest, str]:
        previous_motion_summary = self._previous_motion_summary(document, unit)
        request = InspectionRequest(
            document_id=document["document_id"],
            unit_id=unit["unit_id"],
            episode_start_frame=base.unit_frames.episode_start_frame,
            episode_end_frame=base.unit_frames.episode_end_frame,
            overviews=base.overviews,
            keyframe_sheets=base.keyframe_sheets,
            gripper_state=base.gripper_state,
            previous_motion_summary=previous_motion_summary,
        )
        last_error: Exception | None = None
        for attempt in range(self.config.inspection_retries + 1):
            try:
                result = self.inspection_backend.inspect(request)
            except ApiCallBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - Call 1 failure is data-local
                last_error = exc
                if attempt < self.config.inspection_retries:
                    continue
                break
            try:
                detail = validate_detail_request(
                    result.inspection,
                    max_frame=base.unit_frames.frame_count - 1,
                )
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.config.inspection_retries:
                    continue
                break
            return result, detail, "requested"
        raise AnnotationError(f"Call 1 failed after retries: {last_error}")

    @staticmethod
    def _previous_motion_summary(
        document: dict[str, Any],
        unit: dict[str, Any],
    ) -> str | None:
        order = unit.get("order")
        units = document.get("evidence_units")
        if (
            not isinstance(order, int)
            or isinstance(order, bool)
            or order <= 0
            or not isinstance(units, list)
            or order >= len(units)
        ):
            return None
        previous = units[order - 1]
        if not isinstance(previous, dict):
            return None
        annotation = previous.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("status") != "complete":
            return None
        record = annotation.get("record")
        if not isinstance(record, dict):
            return None
        summary = record.get("motion_summary")
        return summary.strip() if isinstance(summary, str) and summary.strip() else None

    def _finalize_failure(
        self,
        store: DebugArtifactStore,
        document: dict[str, Any],
        unit: dict[str, Any],
        base: BaseMedia,
        detail_status: str,
    ) -> None:
        store.finalize(
            {
                "document_id": document["document_id"],
                "unit_id": unit["unit_id"],
                "episode_start_frame": base.unit_frames.episode_start_frame,
                "episode_end_frame": base.unit_frames.episode_end_frame,
                "preprocessing_version": self.config.preprocessing_version,
                "detail_status": detail_status,
                "status": "failed",
                "config": self.config.manifest(),
            }
        )

    @staticmethod
    def _boundary_observation(
        boundary: Any,
    ) -> dict[str, str] | None:
        annotation = boundary.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("status") != "complete":
            return None
        record = annotation.get("record")
        if not isinstance(record, dict):
            return None
        observation = record["observation"]
        return dict(observation)

    @staticmethod
    def _repair_context(
        document: dict[str, Any],
        unit: dict[str, Any],
        before_observation: dict[str, str] | None,
        after_observation: dict[str, str] | None,
    ) -> str:
        order = int(unit["order"])
        units = document["evidence_units"]

        def transition_summary(index: int) -> dict[str, Any] | None:
            if not 0 <= index < len(units):
                return None
            annotation = units[index]["annotation"]
            record = annotation.get("record")
            if not isinstance(record, dict):
                return None
            return {
                "unit_id": units[index]["unit_id"],
                "motion_summary": record.get("motion_summary"),
                "action_description": record.get("unit_interpretation", {}).get(
                    "action_description"
                ),
            }

        return json.dumps(
            {
                "before_boundary": before_observation,
                "after_boundary": after_observation,
                "previous_transition": transition_summary(order - 1),
                "next_transition": transition_summary(order + 1),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _validate_repaired_call2(
        call2: dict[str, Any],
        *,
        detail: Any,
        before_context: dict[str, str] | None,
        after_context: dict[str, str] | None,
        allowed_boundary_replacements: frozenset[str],
        required_boundary_replacements: frozenset[str],
    ) -> None:
        if (call2["detail_observation"] is not None) != (detail is not None):
            raise AnnotationError(
                "repair detail_observation must match the supplied detail image"
            )
        if before_context is None and call2["before_boundary_observation"] is None:
            raise AnnotationError("repair must create the missing BEFORE Boundary")
        if after_context is None and call2["after_boundary_observation"] is None:
            raise AnnotationError("repair must create the missing AFTER Boundary")
        if (
            before_context is not None
            and call2["before_boundary_observation"] is not None
            and "before" not in allowed_boundary_replacements
        ):
            raise AnnotationError("repair may not replace the accepted BEFORE Boundary")
        if (
            after_context is not None
            and call2["after_boundary_observation"] is not None
            and "after" not in allowed_boundary_replacements
        ):
            raise AnnotationError("repair may not replace the accepted AFTER Boundary")
        replacements = {
            role
            for role, observation in (
                ("before", call2["before_boundary_observation"]),
                ("after", call2["after_boundary_observation"]),
            )
            if observation is not None
        }
        if not required_boundary_replacements <= replacements:
            raise AnnotationError("repair omitted a required Boundary replacement")

    @staticmethod
    def _validate_normal_call2(
        call2: dict[str, Any],
        *,
        detail: Any,
        before_context: dict[str, str] | None,
        after_context: dict[str, str] | None,
    ) -> None:
        if (call2["detail_observation"] is not None) != (detail is not None):
            raise AnnotationError(
                "Call 2 detail observation must match the supplied detail image"
            )
        if (call2["before_boundary_observation"] is not None) != (
            before_context is None
        ):
            raise AnnotationError(
                "Call 2 must describe BEFORE exactly when no Boundary exists"
            )
        if (call2["after_boundary_observation"] is not None) != (after_context is None):
            raise AnnotationError(
                "Call 2 must describe AFTER exactly when no Boundary exists"
            )

    def _run_repair(
        self,
        *,
        document: dict[str, Any],
        unit: dict[str, Any],
        base: BaseMedia,
        detail: Any,
        motion_summary: str,
        call2: dict[str, Any],
        issue_reason: str,
        attempts: int | None = None,
        store: DebugArtifactStore | None = None,
        allowed_boundary_replacements: frozenset[str] = frozenset(),
        required_boundary_replacements: frozenset[str] = frozenset(),
    ) -> RepairOutcome:
        if self.repair_backend is None:
            return RepairOutcome(None, None, None, 0, None)
        before_boundary, after_boundary = unit_boundary_states(document, unit)
        before_context = self._boundary_observation(before_boundary)
        after_context = self._boundary_observation(after_boundary)
        max_attempts = attempts or self.config.repair_max_attempts
        last_result: RepairResult | None = None
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            request = RepairRequest(
                document_id=document["document_id"],
                unit_id=unit["unit_id"],
                task_instruction=document["task_instruction"],
                issue_reason=issue_reason,
                call1_motion_summary=motion_summary,
                call2=call2,
                boundary_context=self._repair_context(
                    document,
                    unit,
                    before_context,
                    after_context,
                ),
                overviews=base.overviews,
                keyframe_sheets=base.keyframe_sheets,
                boundary_images=base.boundary_images,
                gripper_state=base.gripper_state,
                detail=detail,
                required_boundary_replacements=tuple(
                    sorted(required_boundary_replacements)
                ),
            )
            try:
                result = self.repair_backend.repair(request)
            except ApiCallBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - provider failure is data-local
                last_error = f"{type(exc).__name__}: {exc}"
                if store is not None and store.enabled:
                    store.write_json(
                        f"repair-attempt-{attempt:02d}-error.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    )
                continue
            last_result = result
            if store is not None and store.enabled:
                store.write_json(f"repair-attempt-{attempt:02d}.json", asdict(result))
            if not result.repair["evidence_sufficient"]:
                continue
            resolved_call2 = result.repair["resolved_call2"]
            assert isinstance(resolved_call2, dict)
            try:
                self._validate_repaired_call2(
                    resolved_call2,
                    detail=detail,
                    before_context=before_context,
                    after_context=after_context,
                    allowed_boundary_replacements=allowed_boundary_replacements,
                    required_boundary_replacements=required_boundary_replacements,
                )
            except AnnotationError:
                continue
            evidence_result = EvidenceResult(
                evidence=resolved_call2,
                provider=result.provider,
                requested_model=result.requested_model,
                prompt_version=result.prompt_version,
                trace=result.trace,
            )
            before_record = (
                None
                if resolved_call2["before_boundary_observation"] is None
                else compose_boundary_state_record(
                    resolved_call2["before_boundary_observation"]
                )
            )
            after_record = (
                None
                if resolved_call2["after_boundary_observation"] is None
                else compose_boundary_state_record(
                    resolved_call2["after_boundary_observation"]
                )
            )
            return RepairOutcome(
                evidence_result,
                before_record,
                after_record,
                attempt,
                result,
            )
        return RepairOutcome(
            None,
            None,
            None,
            max_attempts,
            last_result,
            last_error,
        )

    def run(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
    ) -> EvidenceUnitPipelineResult:
        store = self._store(document, unit)
        base = self._retry_media(lambda: self.media_builder.build_base(document, unit))
        inspection, detail_request, detail_status = self._inspect(document, unit, base)
        detail = self._retry_media(
            lambda: self.media_builder.build_detail(base, detail_request)
        )
        if store.enabled:
            store.write_many(
                self._retry_media(lambda: self.media_builder.debug_media(base, detail))
            )
            store.write_json("call1.json", asdict(inspection))

        before_boundary, after_boundary = unit_boundary_states(document, unit)
        before_boundary_observation = self._boundary_observation(before_boundary)
        after_boundary_observation = self._boundary_observation(after_boundary)
        request = EvidenceRequest(
            document_id=document["document_id"],
            unit_id=unit["unit_id"],
            episode_start_frame=base.unit_frames.episode_start_frame,
            episode_end_frame=base.unit_frames.episode_end_frame,
            motion_summary=inspection.inspection["motion_summary"],
            before_boundary_observation=before_boundary_observation,
            after_boundary_observation=after_boundary_observation,
            task_instruction=document["task_instruction"],
            detail=detail,
            boundary_images=base.boundary_images,
            gripper_state=base.gripper_state,
        )
        for attempt in range(1, self.config.call2_retries + 2):
            try:
                last_result = self.evidence_backend.annotate(request)
                self._validate_normal_call2(
                    last_result.evidence,
                    detail=detail,
                    before_context=before_boundary_observation,
                    after_context=after_boundary_observation,
                )
            except ApiCallBudgetExceeded:
                raise
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                if store.enabled:
                    store.write_json(f"call2-attempt-{attempt:02d}-error.json", error)
                if attempt <= self.config.call2_retries:
                    continue
                if store.enabled:
                    store.write_json("call2-error.json", error)
                    self._finalize_failure(
                        store,
                        document,
                        unit,
                        base,
                        detail_status,
                    )
                raise
            break
        if store.enabled:
            store.write_json("call2.json", asdict(last_result))

        canonical = compose_evidence_record(last_result.evidence)
        before_boundary_record = (
            None
            if last_result.evidence["before_boundary_observation"] is None
            else compose_boundary_state_record(
                last_result.evidence["before_boundary_observation"]
            )
        )
        after_boundary_record = (
            None
            if last_result.evidence["after_boundary_observation"] is None
            else compose_boundary_state_record(
                last_result.evidence["after_boundary_observation"]
            )
        )
        evidence = EvidenceResult(
            evidence=canonical,
            provider=last_result.provider,
            requested_model=last_result.requested_model,
            prompt_version=last_result.prompt_version,
            trace=last_result.trace,
        )

        if store.enabled:
            store.write_json(
                "final.json",
                {
                    "evidence": canonical,
                    "before_boundary_record": before_boundary_record,
                    "after_boundary_record": after_boundary_record,
                },
            )
            debug_root = store.finalize(
                {
                    "document_id": document["document_id"],
                    "unit_id": unit["unit_id"],
                    "episode_start_frame": base.unit_frames.episode_start_frame,
                    "episode_end_frame": base.unit_frames.episode_end_frame,
                    "preprocessing_version": self.config.preprocessing_version,
                    "detail_status": detail_status,
                    "status": "complete",
                    "config": self.config.manifest(),
                }
            )
        else:
            debug_root = None

        return EvidenceUnitPipelineResult(
            evidence=evidence,
            before_boundary_record=before_boundary_record,
            after_boundary_record=after_boundary_record,
            inspection=inspection,
            detail_status=detail_status,
            debug_root=None if debug_root is None else str(debug_root),
        )

    def repair_target(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
        *,
        issue_reason: str,
        attempts: int | None = None,
        allowed_boundary_replacements: frozenset[str] = frozenset(),
        required_boundary_replacements: frozenset[str] = frozenset(),
    ) -> ExistingRepairOutcome:
        annotation = unit.get("annotation")
        record = annotation.get("record") if isinstance(annotation, dict) else None
        if not isinstance(record, dict):
            return ExistingRepairOutcome(None, None, None, 0, None)
        base = self._retry_media(lambda: self.media_builder.build_base(document, unit))
        motion_summary = record.get("motion_summary")
        if not isinstance(motion_summary, str) or not motion_summary.strip():
            return ExistingRepairOutcome(None, None, None, 0, None)
        call2 = {
            "motion_summary": record["motion_summary"],
            "before_boundary_observation": None,
            "after_boundary_observation": None,
            "detail_observation": record["detail_observation"],
            "unit_interpretation": record["unit_interpretation"],
        }
        detail = self._retry_media(
            lambda: self.media_builder.build_detail(
                base,
                DetailRequest(
                    (0.0, 0.0, 1.0, 1.0),
                    0,
                    base.unit_frames.frame_count - 1,
                ),
            )
        )
        repair = self._run_repair(
            document=document,
            unit=unit,
            base=base,
            detail=detail,
            motion_summary=motion_summary,
            call2=call2,
            issue_reason=issue_reason,
            attempts=attempts,
            allowed_boundary_replacements=allowed_boundary_replacements,
            required_boundary_replacements=required_boundary_replacements,
        )
        if repair.evidence is None or repair.result is None:
            return ExistingRepairOutcome(
                None,
                None,
                None,
                repair.attempts,
                repair.result,
                repair.error,
            )
        canonical = compose_evidence_record(repair.evidence.evidence)
        return ExistingRepairOutcome(
            canonical,
            repair.before_boundary_record,
            repair.after_boundary_record,
            repair.attempts,
            repair.result,
        )
