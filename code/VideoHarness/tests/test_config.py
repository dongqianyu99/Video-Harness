from pathlib import Path

import pytest

from video_harness.config import HarnessConfig


def test_normal_mode_has_no_debug_destination() -> None:
    config = HarnessConfig()
    assert config.debug is False
    assert config.debug_root is None
    assert config.manifest()["debug_root"] is None
    assert config.provider_timeout_s == 300.0
    assert config.provider_max_retries == 2
    assert config.call2_retries == 2
    assert config.ffmpeg_timeout_s == 120.0


def test_debug_mode_requires_explicit_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required"):
        HarnessConfig(debug=True)
    with pytest.raises(ValueError, match="omitted"):
        HarnessConfig(debug=False, debug_root=tmp_path)
    assert HarnessConfig(debug=True, debug_root=tmp_path).debug_root == tmp_path


@pytest.mark.parametrize("field", ["inspection_retries", "call2_retries"])
def test_provider_output_retries_are_nonnegative(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        HarnessConfig(**{field: -1})
    with pytest.raises(ValueError, match=field):
        HarnessConfig(**{field: True})


@pytest.mark.parametrize(
    "field",
    ["repair_max_attempts", "sequence_audit_max_attempts"],
)
def test_automatic_processing_attempts_are_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        HarnessConfig(**{field: 0})


def test_sequence_repair_rounds_are_bounded_nonnegative() -> None:
    with pytest.raises(ValueError, match="sequence_repair_rounds"):
        HarnessConfig(sequence_repair_rounds=-1)


@pytest.mark.parametrize("field", ["provider_timeout_s", "ffmpeg_timeout_s"])
def test_timeouts_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        HarnessConfig(**{field: 0})


def test_provider_retries_may_be_disabled_but_not_negative() -> None:
    assert HarnessConfig(provider_max_retries=0).provider_max_retries == 0
    with pytest.raises(ValueError, match="provider_max_retries"):
        HarnessConfig(provider_max_retries=-1)


def test_timing_contract_is_explicit() -> None:
    with pytest.raises(ValueError, match="25 Hz"):
        HarnessConfig(fps=50)
    with pytest.raises(ValueError, match="26-frame"):
        HarnessConfig(unit_frame_count=25)
