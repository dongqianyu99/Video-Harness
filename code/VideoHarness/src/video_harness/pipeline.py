from __future__ import annotations

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
)
from .config import HarnessConfig
from .debug_artifacts import DebugArtifactStore
from .evidence import compose_evidence_record, mock_inspection_record
from .temporal_media import (
    BaseMedia,
    DetailRequest,
    TemporalMediaBuilder,
    validate_detail_request,
)


@dataclass(frozen=True)
class PipelineResult:
    evidence: EvidenceResult
    inspection: InspectionResult
    detail_status: str
    call2_attempts: int
    selected_call2_attempt: int
    review_status: str
    debug_root: str | None


class UnitPipeline:
    def __init__(
        self,
        *,
        inspection_backend: InspectionBackend,
        evidence_backend: EvidenceBackend,
        media_builder: TemporalMediaBuilder,
        config: HarnessConfig,
    ) -> None:
        self.inspection_backend = inspection_backend
        self.evidence_backend = evidence_backend
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
        request = InspectionRequest(
            document_id=document["document_id"],
            unit_id=unit["unit_id"],
            episode_start_frame=base.unit_frames.episode_start_frame,
            episode_end_frame=base.unit_frames.episode_end_frame,
            overviews=base.overviews,
            stages=base.stages,
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

    def _finalize_failure(
        self,
        store: DebugArtifactStore,
        document: dict[str, Any],
        unit: dict[str, Any],
        base: BaseMedia,
        detail_status: str,
        attempts: int,
    ) -> None:
        store.finalize(
            {
                "document_id": document["document_id"],
                "unit_id": unit["unit_id"],
                "episode_start_frame": base.unit_frames.episode_start_frame,
                "episode_end_frame": base.unit_frames.episode_end_frame,
                "preprocessing_version": self.config.preprocessing_version,
                "detail_status": detail_status,
                "call2_attempts": attempts,
                "status": "failed",
                "config": self.config.manifest(),
            }
        )

    def run(
        self,
        document: dict[str, Any],
        unit: dict[str, Any],
    ) -> PipelineResult:
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

        last_result: EvidenceResult | None = None
        last_result_attempt: int | None = None
        last_error: AnnotationError | None = None
        previous_attempt: dict[str, Any] | None = None
        attempts_used = 0
        for attempt in range(1, self.config.call2_max_attempts + 1):
            attempts_used = attempt
            request = EvidenceRequest(
                document_id=document["document_id"],
                unit_id=unit["unit_id"],
                episode_start_frame=base.unit_frames.episode_start_frame,
                episode_end_frame=base.unit_frames.episode_end_frame,
                motion_summary=inspection.inspection["motion_summary"],
                task_instruction=document["task_instruction"],
                detail=detail,
                endpoints=base.endpoints,
                previous_attempt=previous_attempt,
            )
            try:
                result = self.evidence_backend.annotate(request)
            except AnnotationError as exc:
                last_error = exc
                if store.enabled:
                    store.write_json(
                        f"call2-attempt-{attempt:02d}-error.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    )
                continue

            detail_is_present = result.evidence["detail_observation"] is not None
            if detail_is_present != (detail is not None):
                last_error = AnnotationError(
                    "Call 2 detail_observation must be present exactly when a detail sheet is supplied"
                )
                if store.enabled:
                    store.write_json(
                        f"call2-attempt-{attempt:02d}-error.json",
                        {"type": "AnnotationError", "message": str(last_error)},
                    )
                continue

            last_result = result
            last_result_attempt = attempt
            previous_attempt = result.evidence
            if store.enabled:
                store.write_json(
                    f"call2-attempt-{attempt:02d}.json",
                    asdict(result),
                )
            if result.evidence["causal_validation"]["status"] == "pass":
                break

        if last_result is None:
            if store.enabled:
                self._finalize_failure(
                    store,
                    document,
                    unit,
                    base,
                    detail_status,
                    attempts_used,
                )
            if last_error is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Call 2 produced no result")
            raise last_error

        assert last_result_attempt is not None
        causal_status = last_result.evidence["causal_validation"]["status"]
        review_status = "accepted" if causal_status == "pass" else "needs_review"
        canonical = compose_evidence_record(
            inspection.inspection["motion_summary"],
            last_result.evidence,
            review_status=review_status,
        )
        evidence = replace(last_result, evidence=canonical)

        if store.enabled:
            store.write_json(
                "final.json",
                {
                    "selected_call2_attempt": last_result_attempt,
                    "call2_attempts": attempts_used,
                    "review_status": review_status,
                    "evidence": canonical,
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
                    "call2_attempts": attempts_used,
                    "selected_call2_attempt": last_result_attempt,
                    "review_status": review_status,
                    "status": "complete",
                    "config": self.config.manifest(),
                }
            )
        else:
            debug_root = None

        return PipelineResult(
            evidence=evidence,
            inspection=inspection,
            detail_status=detail_status,
            call2_attempts=attempts_used,
            selected_call2_attempt=last_result_attempt,
            review_status=review_status,
            debug_root=None if debug_root is None else str(debug_root),
        )
