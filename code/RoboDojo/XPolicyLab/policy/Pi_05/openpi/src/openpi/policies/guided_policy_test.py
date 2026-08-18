from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.guide_inputs import GuideInput
from openpi.policies import guided_policy as _guided_policy


def _guide(groups: int = 1) -> GuideInput:
    return GuideInput(
        images=jnp.zeros((groups, 2, 2, 2, 3), dtype=jnp.float32),
        image_mask=jnp.ones((groups, 2), dtype=jnp.bool_),
        text_tokens=jnp.zeros((groups, 1, 4), dtype=jnp.int32),
        text_mask=jnp.ones((groups, 1, 4), dtype=jnp.bool_),
        unit_mask=jnp.ones((groups, 1), dtype=jnp.bool_),
        before_slot=jnp.zeros((groups, 1), dtype=jnp.int32),
        after_slot=jnp.ones((groups, 1), dtype=jnp.int32),
    )


class _FakeModel:
    def __init__(self):
        self.calls = []

    def sample_actions(self, *_args, **_kwargs):
        raise AssertionError("stock sample_actions must not be called")

    def sample_guided_actions(self, rng, observation, *, guide, **kwargs):
        self.calls.append((rng, observation, guide, kwargs))
        return jnp.ones((1, 1, 3, 2), dtype=jnp.float32)


def _observation() -> dict:
    return {
        "image": {},
        "image_mask": {},
        "state": np.asarray([0.25, -0.5], dtype=np.float32),
        "tokenized_prompt": np.asarray([1, 2, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.asarray([True, True, False], dtype=np.bool_),
    }


def test_infer_uses_grouped_guided_sampling_and_squeezes_output(monkeypatch):
    monkeypatch.setattr(_guided_policy.nnx_utils, "module_jit", lambda function: function)
    model = _FakeModel()
    policy = _guided_policy.GuidedPolicy(model)
    guide = _guide()
    policy.set_guide(guide)

    result = policy.infer(_observation())

    assert result["state"].shape == (2,)
    assert result["actions"].shape == (3, 2)
    assert len(model.calls) == 1
    _, observation, received_guide, _ = model.calls[0]
    assert observation.state.shape == (1, 1, 2)
    assert received_guide is guide


def test_infer_requires_guide_and_unbatched_query(monkeypatch):
    monkeypatch.setattr(_guided_policy.nnx_utils, "module_jit", lambda function: function)
    policy = _guided_policy.GuidedPolicy(_FakeModel())
    with pytest.raises(RuntimeError, match="set_guide"):
        policy.infer(_observation())

    policy.set_guide(_guide())
    batched = _observation()
    batched["state"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unbatched"):
        policy.infer(batched)


def test_set_guide_requires_g_one(monkeypatch):
    monkeypatch.setattr(_guided_policy.nnx_utils, "module_jit", lambda function: function)
    policy = _guided_policy.GuidedPolicy(_FakeModel())

    with pytest.raises(ValueError, match="G=1"):
        policy.set_guide(_guide(groups=2))


def test_factory_rejects_pytorch_and_missing_params(tmp_path):
    config = SimpleNamespace()
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="JAX"):
        _guided_policy.create_trained_guided_policy(config, tmp_path)

    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="params"):
        _guided_policy.create_trained_guided_policy(config, tmp_path)
