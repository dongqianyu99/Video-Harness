from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
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
    boundary_state_is_usable,
    compose_boundary_state_record,
    compose_evidence_record,
    mock_inspection_record,
)
from .sampling import unit_boundary_states
from .temporal_media import (
    BaseMedia,
    DetailRequest,
    TemporalMediaBuilder,
    validate_detail_request,
)


@dataclass(frozen=True)
class EvidenceUnitPipelineResult:
    evidence: EvidenceResult
    initial_evidence: EvidenceResult
    before_boundary_record: dict[str, Any] | None
    after_boundary_record: dict[str, Any] | None
    conflicted_boundary_roles: tuple[str, ...]
    inspection: InspectionResult
    detail_status: str
    repair_attempts: int
    repair: RepairResult | None
    quality_status: str
    debug_root: str | None


@dataclass(frozen=True)
class RepairOutcome:
    evidence: EvidenceResult | None
    before_boundary_record: dict[str, Any] | None
    after_boundary_record: dict[str, Any] | None
    attempts: int
    result: RepairResult | None


@dataclass(frozen=True)
class ExistingRepairOutcome:
    canonical_evidence: dict[str, Any] | None
    before_boundary_record: dict[str, Any] | None
    after_boundary_record: dict[str, Any] | None
    attempts: int
    result: RepairResult | None


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
    ) -> tuple[InspectionResult, DetailRequest | None, str, str | None]:
        previous_motion_summary = self._previous_motion_summary(document, unit)
        request = InspectionRequest(
            document_id=document["document_id"],
            unit_id=unit["unit_id"],
            episode_start_frame=base.unit_frames.episode_start_frame,
            episode_end_frame=base.unit_frames.episode_end_frame,
            overviews=base.overviews,
            keyframe_sheets=base.keyframe_sheets,
            previous_motion_summary=previous_motion_summary,
        )
        last_result: InspectionResult | None = None
        last_error: Exception | None = None
        for attempt in range(self.config.inspection_retries + 1):
            try:
                result = self.inspection_backend.inspect(request)
            except AnnotationError as exc:
                last_error = exc
                if attempt < self.config.inspection_retries:
                    continue
                break
            last_result = result
            try:
                detail = validate_detail_request(
                    result.inspection,
                    max_frame=base.unit_frames.frame_count - 1,
                )
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.config.inspection_retries:
                    continue
                return result, None, "invalid-request-omitted", str(exc)
            return (
                result,
                detail,
                "requested" if detail is not None else "not-requested",
                None,
            )
        if last_result is not None:
            return last_result, None, "inspection-failed-omitted", str(last_error)
        fallback = InspectionResult(
            inspection=mock_inspection_record(),
            provider="harness-fallback",
            requested_model=self.inspection_backend.model,
        )
        return fallback, None, "inspection-failed-omitted", str(last_error)

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
        provenance = annotation.get("provenance")
        call1 = provenance.get("call1") if isinstance(provenance, dict) else None
        if not isinstance(call1, dict) or call1.get("provider") == "harness-fallback":
            return None
        record = annotation.get("record")
        if not isinstance(record, dict) or record.get("quality_status") != "accepted":
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
        *,
        accepted_only: bool = True,
    ) -> dict[str, str] | None:
        annotation = boundary.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("status") != "complete":
            return None
        record = annotation.get("record")
        if not isinstance(record, dict):
            return None
        if accepted_only and not boundary_state_is_usable(record):
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
                "motion_summary": record.get("resolved_motion_summary")
                or record.get("motion_summary"),
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
        if any(value is not None for value in call2["boundary_conflicts"].values()):
            raise AnnotationError(
                "successful repair must resolve all Boundary conflicts"
            )
        if call2["causal_validation"]["status"] != "pass":
            raise AnnotationError("successful repair must return causal status=pass")

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
                "Call 2 must describe BEFORE exactly when no accepted Boundary exists"
            )
        if (call2["after_boundary_observation"] is not None) != (after_context is None):
            raise AnnotationError(
                "Call 2 must describe AFTER exactly when no accepted Boundary exists"
            )
        conflicts = call2["boundary_conflicts"]
        if (before_context is None and conflicts["before"] is not None) or (
            after_context is None and conflicts["after"] is not None
        ):
            raise AnnotationError("Call 2 cannot conflict with a missing Boundary")
        if (
            any(value is not None for value in conflicts.values())
            and call2["causal_validation"]["status"] != "retry"
        ):
            raise AnnotationError("Boundary conflicts must request automatic repair")

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
    ) -> RepairOutcome:
        if self.repair_backend is None:
            return RepairOutcome(None, None, None, 0, None)
        before_boundary, after_boundary = unit_boundary_states(document, unit)
        before_context = self._boundary_observation(
            before_boundary,
            accepted_only=False,
        )
        after_context = self._boundary_observation(
            after_boundary,
            accepted_only=False,
        )
        max_attempts = attempts or self.config.repair_max_attempts
        last_result: RepairResult | None = None
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
                detail=detail,
            )
            try:
                result = self.repair_backend.repair(request)
            except Exception as exc:  # noqa: BLE001 - provider failure is data-local
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
                    resolved_call2["before_boundary_observation"],
                    quality_status="accepted",
                )
            )
            after_record = (
                None
                if resolved_call2["after_boundary_observation"] is None
                else compose_boundary_state_record(
                    resolved_call2["after_boundary_observation"],
                    quality_status="accepted",
                )
            )
            return RepairOutcome(
                evidence_result,
                before_record,
                after_record,
                attempt,
                result,
            )
        return RepairOutcome(None, None, None, max_attempts, last_result)

    def run(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
    ) -> EvidenceUnitPipelineResult:
        store = self._store(document, unit)
        base = self.media_builder.build_base(document, unit)
        inspection, detail_request, detail_status, inspection_error = self._inspect(
            document,
            unit,
            base,
        )
        detail = (
            None
            if detail_request is None
            else self.media_builder.build_detail(base, detail_request)
        )
        if store.enabled:
            store.write_many(self.media_builder.debug_media(base, detail))
            store.write_json("call1.json", asdict(inspection))
            if inspection_error is not None:
                store.write_json(
                    "call1-error.json",
                    {"message": inspection_error, "detail_status": detail_status},
                )

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
        )
        try:
            last_result = self.evidence_backend.annotate(request)
            self._validate_normal_call2(
                last_result.evidence,
                detail=detail,
                before_context=before_boundary_observation,
                after_context=after_boundary_observation,
            )
        except Exception as exc:
            if store.enabled:
                store.write_json(
                    "call2-error.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
            if store.enabled:
                self._finalize_failure(
                    store,
                    document,
                    unit,
                    base,
                    detail_status,
                )
            raise
        if store.enabled:
            store.write_json("call2.json", asdict(last_result))

        initial_evidence = last_result
        repair_outcome = RepairOutcome(None, None, None, 0, None)
        resolved_motion_summary: str | None = None
        if last_result.evidence["causal_validation"]["status"] == "retry":
            conflict_reasons = [
                reason
                for reason in last_result.evidence["boundary_conflicts"].values()
                if reason is not None
            ]
            issue_reason = " ".join(
                [
                    last_result.evidence["causal_validation"]["reason"],
                    *conflict_reasons,
                ]
            )
            repair_outcome = self._run_repair(
                document=document,
                unit=unit,
                base=base,
                detail=detail,
                motion_summary=inspection.inspection["motion_summary"],
                call2=last_result.evidence,
                issue_reason=issue_reason,
                store=store,
                allowed_boundary_replacements=frozenset(
                    role
                    for role, reason in last_result.evidence[
                        "boundary_conflicts"
                    ].items()
                    if reason is not None
                ),
            )
            if repair_outcome.evidence is not None:
                last_result = repair_outcome.evidence
                assert repair_outcome.result is not None
                resolved_motion_summary = repair_outcome.result.repair[
                    "resolved_motion_summary"
                ]

        causal_status = last_result.evidence["causal_validation"]["status"]
        quality_status = "accepted" if causal_status == "pass" else "quarantined"
        boundary_quality_status = (
            "quarantined" if last_result.provider == "mock" else "accepted"
        )
        canonical = compose_evidence_record(
            inspection.inspection["motion_summary"],
            last_result.evidence,
            quality_status=quality_status,
            resolved_motion_summary=resolved_motion_summary,
        )
        before_boundary_record = repair_outcome.before_boundary_record
        if repair_outcome.evidence is None:
            before_boundary_record = (
                None
                if last_result.evidence["before_boundary_observation"] is None
                else compose_boundary_state_record(
                    last_result.evidence["before_boundary_observation"],
                    quality_status=boundary_quality_status,
                )
            )
        after_boundary_record = repair_outcome.after_boundary_record
        if repair_outcome.evidence is None:
            after_boundary_record = (
                None
                if last_result.evidence["after_boundary_observation"] is None
                else compose_boundary_state_record(
                    last_result.evidence["after_boundary_observation"],
                    quality_status=boundary_quality_status,
                )
            )
        conflicted_boundary_roles = tuple(
            role
            for role, reason in last_result.evidence["boundary_conflicts"].items()
            if reason is not None
        )
        evidence = replace(last_result, evidence=canonical)

        if store.enabled:
            store.write_json(
                "final.json",
                {
                    "quality_status": quality_status,
                    "evidence": canonical,
                    "before_boundary_record": before_boundary_record,
                    "after_boundary_record": after_boundary_record,
                    "conflicted_boundary_roles": conflicted_boundary_roles,
                    "repair_attempts": repair_outcome.attempts,
                    "repair": (
                        None
                        if repair_outcome.result is None
                        else asdict(repair_outcome.result)
                    ),
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
                    "quality_status": quality_status,
                    "repair_attempts": repair_outcome.attempts,
                    "status": "complete",
                    "config": self.config.manifest(),
                }
            )
        else:
            debug_root = None

        return EvidenceUnitPipelineResult(
            evidence=evidence,
            initial_evidence=initial_evidence,
            before_boundary_record=before_boundary_record,
            after_boundary_record=after_boundary_record,
            conflicted_boundary_roles=conflicted_boundary_roles,
            inspection=inspection,
            detail_status=detail_status,
            repair_attempts=repair_outcome.attempts,
            repair=repair_outcome.result,
            quality_status=quality_status,
            debug_root=None if debug_root is None else str(debug_root),
        )

    def repair_existing(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
        *,
        issue_reason: str,
        attempts: int | None = None,
    ) -> ExistingRepairOutcome:
        annotation = unit.get("annotation")
        record = annotation.get("record") if isinstance(annotation, dict) else None
        if not isinstance(record, dict):
            return ExistingRepairOutcome(None, None, None, 0, None)
        base = self.media_builder.build_base(document, unit)
        motion_summary = record.get("resolved_motion_summary") or record.get(
            "motion_summary"
        )
        if not isinstance(motion_summary, str) or not motion_summary.strip():
            return ExistingRepairOutcome(None, None, None, 0, None)
        call2 = {
            "before_boundary_observation": None,
            "after_boundary_observation": None,
            "boundary_conflicts": {"before": None, "after": None},
            "detail_observation": None,
            "unit_interpretation": record["unit_interpretation"],
            "causal_validation": record["causal_validation"],
        }
        repair = self._run_repair(
            document=document,
            unit=unit,
            base=base,
            detail=None,
            motion_summary=motion_summary,
            call2=call2,
            issue_reason=issue_reason,
            attempts=attempts,
            allowed_boundary_replacements=frozenset({"before", "after"}),
        )
        if repair.evidence is None or repair.result is None:
            return ExistingRepairOutcome(
                None,
                None,
                None,
                repair.attempts,
                repair.result,
            )
        canonical = compose_evidence_record(
            record["motion_summary"],
            repair.evidence.evidence,
            quality_status="accepted",
            resolved_motion_summary=repair.result.repair["resolved_motion_summary"],
        )
        return ExistingRepairOutcome(
            canonical,
            repair.before_boundary_record,
            repair.after_boundary_record,
            repair.attempts,
            repair.result,
        )
