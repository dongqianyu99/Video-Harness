from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from video_harness.run_tracking import (
    ApiCallBudgetExceeded,
    RunTracker,
    TrackingBackend,
)


def test_shared_api_budget_is_atomic_and_disabled_by_default(tmp_path) -> None:
    unlimited = RunTracker(tmp_path / "unlimited")
    assert unlimited.reserve_api_call(stage="call1") == 1
    assert unlimited.reserve_api_call(stage="call2") == 2
    assert unlimited.snapshot() == {"api_calls_reserved": 2, "max_api_calls": None}

    root = tmp_path / "limited"
    first = RunTracker(root, max_api_calls=50)
    second = RunTracker(root, max_api_calls=50)
    with ThreadPoolExecutor(max_workers=8) as executor:
        indices = list(
            executor.map(
                lambda _: first.reserve_api_call(stage="call1"),
                range(25),
            )
        ) + list(
            executor.map(
                lambda _: second.reserve_api_call(stage="call2"),
                range(25),
            )
        )
    assert sorted(indices) == list(range(1, 51))
    with pytest.raises(ApiCallBudgetExceeded, match="50/50"):
        first.reserve_api_call(stage="repair")
    assert first.snapshot() == {"api_calls_reserved": 50, "max_api_calls": 50}


def test_tracking_backend_records_provider_trace_and_usage(tmp_path) -> None:
    class Backend:
        provider = "openai"
        model = "test-model"

        @staticmethod
        def inspect(request):
            del request
            return SimpleNamespace(
                trace=SimpleNamespace(
                    request_id="req-1",
                    response_model="resolved-model",
                    usage={"input_tokens": 10, "output_tokens": 2},
                )
            )

    tracker = RunTracker(tmp_path, max_api_calls=1, context={"shard_index": 0})
    backend = TrackingBackend(Backend(), tracker)
    backend.inspect(SimpleNamespace(document_id="doc-1", unit_id="u0000"))

    events = [
        json.loads(line)
        for line in tracker.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "provider_call_started",
        "provider_call_completed",
    ]
    completed = events[-1]
    assert completed["api_call_index"] == 1
    assert completed["document_id"] == "doc-1"
    assert completed["request_id"] == "req-1"
    assert completed["usage"] == {"input_tokens": 10, "output_tokens": 2}
