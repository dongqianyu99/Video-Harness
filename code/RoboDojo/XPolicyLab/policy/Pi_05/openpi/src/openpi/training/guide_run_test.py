from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpi.training.guide_run import GuidedRunLayout
from openpi.training.guide_run import configure_guided_run_logging
from openpi.training.guide_run import finish_guided_wandb
from openpi.training.guide_run import init_guided_wandb
from openpi.training.guide_run import prepare_guided_run_layout
from openpi.training.guide_run import validate_guided_run_resume
from openpi.training.guide_run import write_guided_run_manifest


@dataclass(frozen=True)
class _RunConfig:
    experiment_name: str
    value: int = 3
    resume: bool = False
    overwrite: bool = False


def _layout(tmp_path: Path) -> GuidedRunLayout:
    return GuidedRunLayout.from_paths(
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "run" / "checkpoints",
    )


def test_run_layout_creates_unified_artifact_tree_and_manifest(tmp_path: Path):
    layout = _layout(tmp_path)
    prepare_guided_run_layout(layout, resume=False, overwrite=False)
    layout.checkpoints.mkdir()
    write_guided_run_manifest(
        layout,
        run_config=_RunConfig("test-run"),
        status="running",
    )
    write_guided_run_manifest(
        layout,
        run_config=_RunConfig("test-run"),
        status="completed",
    )

    assert layout.logs.is_dir()
    assert layout.wandb.is_dir()
    assert layout.checkpoints.is_dir()
    assert layout.eval.is_dir()
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["completed_at"] is not None
    assert manifest["layout"]["eval"] == str(layout.eval)
    assert manifest["config"]["experiment_name"] == "test-run"


def test_run_layout_refuses_mixed_artifacts_and_validates_resume(tmp_path: Path):
    layout = _layout(tmp_path)
    prepare_guided_run_layout(layout, resume=False, overwrite=False)

    with pytest.raises(FileExistsError, match="already contains artifacts"):
        prepare_guided_run_layout(layout, resume=False, overwrite=False)
    with pytest.raises(FileNotFoundError, match="manifest"):
        prepare_guided_run_layout(layout, resume=True, overwrite=False)


def test_resume_rejects_config_and_runtime_data_drift(tmp_path: Path):
    layout = _layout(tmp_path)
    prepare_guided_run_layout(layout, resume=False, overwrite=False)
    original = _RunConfig("test-run")
    data = {
        "catalog_build_id": "build",
        "catalog_digest": "catalog",
        "task_sample_digest": "samples",
        "guide_representation_digest": "representation",
        "guide_materialization_digest": "materialization",
        "guide_materialization_cache_digest": "cache",
        "guide_max_units": 2,
        "guide_max_boundaries": 3,
        "guide_boundary_num_queries": 8,
        "guide_transition_num_queries": 4,
        "guides_per_batch": 2,
        "queries_per_guide": 4,
        "remainder_strategy": "drop",
        "gradient_accumulation_steps": 1,
    }
    write_guided_run_manifest(
        layout,
        run_config=original,
        status="running",
        runtime={"data": data},
    )

    resumed = replace(original, resume=True)
    validate_guided_run_resume(layout, run_config=resumed)
    validate_guided_run_resume(layout, run_config=resumed, runtime_data=data)

    with pytest.raises(ValueError, match="config does not match"):
        validate_guided_run_resume(
            layout,
            run_config=replace(resumed, value=99),
        )
    with pytest.raises(ValueError, match="catalog_digest"):
        validate_guided_run_resume(
            layout,
            run_config=resumed,
            runtime_data={**data, "catalog_digest": "changed"},
        )
    with pytest.raises(ValueError, match="guide_max_units"):
        validate_guided_run_resume(
            layout,
            run_config=resumed,
            runtime_data={**data, "guide_max_units": 1},
        )
    with pytest.raises(ValueError, match="guide_materialization_cache_digest"):
        validate_guided_run_resume(
            layout,
            run_config=resumed,
            runtime_data={**data, "guide_materialization_cache_digest": "changed"},
        )


def test_file_logging_preserves_terminal_handler_and_writes_train_log(tmp_path: Path):
    layout = _layout(tmp_path)
    prepare_guided_run_layout(layout, resume=False, overwrite=False)
    root = logging.getLogger()
    existing_handlers = tuple(root.handlers)
    configure_guided_run_logging(layout, resume=False)

    logging.getLogger("guide-run-test").warning("persist-this-line")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "persist-this-line" in layout.train_log.read_text(encoding="utf-8")
    assert all(handler in logging.getLogger().handlers for handler in existing_handlers)


class _FakeWandbRun:
    def __init__(self):
        self.id = "wandb-test-id"
        self.defined = []
        self.finished = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class _FakeWandb:
    def __init__(self):
        self.calls = []
        self.run = _FakeWandbRun()

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


def test_wandb_uses_run_directory_and_stable_resume_id(tmp_path: Path):
    layout = _layout(tmp_path)
    prepare_guided_run_layout(layout, resume=False, overwrite=False)
    wandb = _FakeWandb()
    resolved = SimpleNamespace(
        wandb_enabled=True,
        project_name="project",
        exp_name="experiment",
    )
    run_config = _RunConfig("test-run")

    run = init_guided_wandb(
        layout=layout,
        run_config=run_config,
        resolved_config=resolved,
        resuming=False,
        wandb_module=wandb,
    )

    assert wandb.calls[0]["dir"] == str(layout.wandb)
    assert layout.wandb_id.read_text().strip() == "wandb-test-id"
    assert run.defined

    resumed_wandb = _FakeWandb()
    resumed = init_guided_wandb(
        layout=layout,
        run_config=run_config,
        resolved_config=resolved,
        resuming=True,
        wandb_module=resumed_wandb,
    )
    assert resumed_wandb.calls[0]["id"] == "wandb-test-id"
    assert resumed_wandb.calls[0]["resume"] == "must"
    finish_guided_wandb(resumed, exit_code=0)
    assert resumed.finished == [0]
