from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = "video-harness.run-state"


class ApiCallBudgetExceeded(RuntimeError):
    """Raised before a provider call that would exceed the shared run budget."""


class RunTracker:
    """Append-only run log plus a shared, process-safe API call counter."""

    def __init__(
        self,
        root: Path,
        *,
        max_api_calls: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if max_api_calls is not None and (
            isinstance(max_api_calls, bool)
            or not isinstance(max_api_calls, int)
            or max_api_calls < 0
        ):
            raise ValueError("max_api_calls must be a non-negative integer")
        self.root = Path(root)
        self.max_api_calls = max_api_calls
        self.context = dict(context or {})
        self.lock_path = self.root / "run-state.lock"
        self.state_path = self.root / "run-state.json"
        self.events_path = self.root / "events.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.state_path.is_file():
                self._write_state_locked({"api_calls_reserved": 0})

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_state_locked(self) -> dict[str, int]:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != STATE_SCHEMA_VERSION
            or set(value) != {"schema_version", "api_calls_reserved"}
            or isinstance(value["api_calls_reserved"], bool)
            or not isinstance(value["api_calls_reserved"], int)
            or value["api_calls_reserved"] < 0
        ):
            raise ValueError(f"Invalid run state: {self.state_path}")
        return {"api_calls_reserved": value["api_calls_reserved"]}

    def _write_state_locked(self, state: dict[str, int]) -> None:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "api_calls_reserved": state["api_calls_reserved"],
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".run-state.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _event(self, event: str, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "pid": os.getpid(),
            **self.context,
            **fields,
        }

    def _append_event_locked(self, value: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()

    def log(self, event: str, **fields: Any) -> None:
        with self._locked():
            self._append_event_locked(self._event(event, fields))

    def reserve_api_call(self, **fields: Any) -> int:
        with self._locked():
            state = self._read_state_locked()
            current = state["api_calls_reserved"]
            if self.max_api_calls is not None and current >= self.max_api_calls:
                self._append_event_locked(
                    self._event(
                        "api_budget_exhausted",
                        {
                            **fields,
                            "api_calls_reserved": current,
                            "max_api_calls": self.max_api_calls,
                        },
                    )
                )
                raise ApiCallBudgetExceeded(
                    f"shared API call budget exhausted: {current}/{self.max_api_calls}"
                )
            call_index = current + 1
            self._write_state_locked({"api_calls_reserved": call_index})
            return call_index

    def snapshot(self) -> dict[str, int | None]:
        with self._locked():
            state = self._read_state_locked()
        return {
            "api_calls_reserved": state["api_calls_reserved"],
            "max_api_calls": self.max_api_calls,
        }


class TrackingBackend:
    """Record one logical provider call around an existing backend method."""

    def __init__(self, backend: Any, tracker: RunTracker) -> None:
        self.backend = backend
        self.tracker = tracker
        self.provider = backend.provider
        self.model = backend.model

    @staticmethod
    def _request_identity(request: Any) -> tuple[str | None, str | None]:
        document_id = getattr(request, "document_id", None)
        unit_id = getattr(request, "unit_id", None)
        if document_id is None and hasattr(request, "canonical_sequence"):
            try:
                value = json.loads(request.canonical_sequence)
                document_id = value.get("document_id")
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        return document_id, unit_id

    def _call(
        self,
        stage: str,
        request: Any,
        method: Callable[[Any], Any],
    ) -> Any:
        document_id, unit_id = self._request_identity(request)
        fields = {
            "stage": stage,
            "provider": self.provider,
            "model": self.model,
            "document_id": document_id,
            "unit_id": unit_id,
        }
        api_call_index = (
            None if self.provider == "mock" else self.tracker.reserve_api_call(**fields)
        )
        self.tracker.log(
            "provider_call_started",
            **fields,
            api_call_index=api_call_index,
        )
        started = time.monotonic()
        try:
            result = method(request)
        except Exception as exc:
            self.tracker.log(
                "provider_call_failed",
                **fields,
                api_call_index=api_call_index,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        trace = getattr(result, "trace", None)
        self.tracker.log(
            "provider_call_completed",
            **fields,
            api_call_index=api_call_index,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            request_id=getattr(trace, "request_id", None),
            response_model=getattr(trace, "response_model", None),
            usage=getattr(trace, "usage", None),
        )
        return result

    def inspect(self, request: Any) -> Any:
        return self._call("call1", request, self.backend.inspect)

    def annotate(self, request: Any) -> Any:
        return self._call("call2", request, self.backend.annotate)

    def repair(self, request: Any) -> Any:
        return self._call("repair", request, self.backend.repair)

    def audit_sequence(self, request: Any) -> Any:
        return self._call("sequence_audit", request, self.backend.audit_sequence)
