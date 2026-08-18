from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05 import model as _model


class _Policy:
    def __init__(self):
        self.guides = []
        self.infer_calls = []

    def set_guide(self, guide):
        self.guides.append(guide)

    def infer(self, observation, **kwargs):
        self.infer_calls.append((observation, kwargs))
        return {"actions": np.ones((3, 2), dtype=np.float32)}


class _Session:
    def __init__(self):
        self.guide = None
        self.instructions = []

    def bind_instruction(self, instruction):
        self.instructions.append(instruction)
        if self.guide is None:
            self.guide = object()
        return self.guide


def _observation(instruction="toast bread", env_idx=0):
    return {
        "env_idx": env_idx,
        "instruction": instruction,
        "state": np.asarray([0.1, 0.2], dtype=np.float32),
        "images": {
            "cam_high": np.zeros((2, 2, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        },
    }


def _guided_model(monkeypatch):
    policy = _Policy()
    session = _Session()
    monkeypatch.setattr(_model.Model, "get_model", lambda self, model_cfg: policy)
    monkeypatch.setattr(_model.Model, "_create_guide_session", lambda self, model_cfg: session)
    instance = _model.Model(
        {
            "task_name": "make_toast",
            "env_cfg_type": None,
            "guidance_enabled": True,
        }
    )
    return instance, policy, session


def test_guided_model_materializes_once_and_reset_preserves_task_guide(monkeypatch):
    instance, policy, session = _guided_model(monkeypatch)

    instance.update_obs(_observation())
    first_guide = policy.guides[-1]
    instance.get_action()
    instance.reset()
    instance.update_obs(_observation())

    assert policy.guides[-1] is first_guide
    assert session.instructions == ["toast bread", "toast bread"]
    assert instance.observation_window is not None
    assert policy.infer_calls


def test_guided_batch_requires_one_shared_instruction(monkeypatch):
    instance, _, _ = _guided_model(monkeypatch)

    with pytest.raises(ValueError, match="shared task instruction"):
        instance.update_obs_batch([
            _observation("toast bread", 0),
            _observation("put cup away", 1),
        ])


def test_prepare_case_validates_task_and_trial_end_clears_only_observation(monkeypatch):
    instance, policy, session = _guided_model(monkeypatch)
    response = instance.prepare_case({"task_name": "make_toast", "action_case_id": "case-1"})
    instance.update_obs(_observation())
    guide = policy.guides[-1]
    instance.on_trial_end({"success": False})

    assert response["task_name"] == "make_toast"
    assert instance.observation_window is None
    assert session.guide is guide

    with pytest.raises(ValueError, match="does not match"):
        instance.prepare_case({"task_name": "other"})


def test_stock_mode_does_not_create_guidance_session(monkeypatch):
    policy = _Policy()
    monkeypatch.setattr(_model.Model, "get_model", lambda self, model_cfg: policy)
    instance = _model.Model(
        {
            "task_name": "make_toast",
            "env_cfg_type": None,
            "guidance_enabled": False,
        }
    )

    instance.update_obs(_observation())

    assert policy.guides == []


def test_get_model_selects_guided_checkpoint_factory(monkeypatch, tmp_path):
    instance = _model.Model.__new__(_model.Model)
    instance.guidance_enabled = True
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    native_config = object()
    sentinel = object()
    calls = []

    monkeypatch.setattr(_model, "_resolve_pi05_model_root", lambda _cfg: checkpoint)
    monkeypatch.setattr(_model._config, "get_config", lambda _name: native_config)
    monkeypatch.setattr(_model._normalize, "load", lambda _path: {"stats": True})

    def create_policy(config, model_root, *, norm_stats):
        calls.append((config, model_root, norm_stats))
        return sentinel

    original_import = _model.importlib.import_module
    monkeypatch.setattr(
        _model.importlib,
        "import_module",
        lambda name: SimpleNamespace(create_trained_guided_policy=create_policy)
        if name == "openpi.policies.guided_policy"
        else original_import(name),
    )

    result = instance.get_model(
        {
            "train_config_name": "native-config",
            "repo_id": "arx_x5_sim",
        }
    )

    assert result is sentinel
    assert calls == [(native_config, str(checkpoint), {"stats": True})]
