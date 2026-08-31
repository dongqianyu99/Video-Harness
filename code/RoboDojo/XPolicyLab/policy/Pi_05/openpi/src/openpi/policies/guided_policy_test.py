from __future__ import annotations

from types import SimpleNamespace

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.guide_encoder import GuideMemory
from openpi.models.guide_inputs import GuideInput
from openpi.policies import guided_policy as _guided_policy


def _guide(groups: int = 1) -> GuideInput:
    source_kind = jnp.asarray([0] * 8 + [1] * 4 + [0] * 8, dtype=jnp.int32)
    source_index = jnp.asarray([0] * 8 + [0] * 4 + [1] * 8, dtype=jnp.int32)
    source_offset = jnp.asarray([*range(8), *range(4), *range(8)], dtype=jnp.int32)
    return GuideInput(
        boundary_images=jnp.zeros((groups, 2, 3, 2, 2, 3), dtype=jnp.float32),
        boundary_image_mask=jnp.ones((groups, 2, 3), dtype=jnp.bool_),
        boundary_text_tokens=jnp.zeros((groups, 2, 3, 4), dtype=jnp.int32),
        boundary_text_mask=jnp.ones((groups, 2, 3, 4), dtype=jnp.bool_),
        transition_text_tokens=jnp.zeros((groups, 1, 4), dtype=jnp.int32),
        transition_text_mask=jnp.ones((groups, 1, 4), dtype=jnp.bool_),
        boundary_mask=jnp.ones((groups, 2), dtype=jnp.bool_),
        unit_mask=jnp.ones((groups, 1), dtype=jnp.bool_),
        memory_source_kind=jnp.broadcast_to(source_kind, (groups, 20)),
        memory_source_index=jnp.broadcast_to(source_index, (groups, 20)),
        memory_source_offset=jnp.broadcast_to(source_offset, (groups, 20)),
        memory_mask=jnp.ones((groups, 20), dtype=jnp.bool_),
    )


class _FakeModel:
    def __init__(self):
        self.encode_calls = []
        self.calls = []

    def sample_actions(self, *_args, **_kwargs):
        raise AssertionError("stock sample_actions must not be called")

    def encode_guide(self, guide):
        self.encode_calls.append(guide)
        return SimpleNamespace(tokens=guide.memory_source_index)

    def sample_guided_actions_with_memory(self, rng, observation, *, guide_memory, **kwargs):
        self.calls.append((rng, observation, guide_memory, kwargs))
        return jnp.ones((1, 1, 3, 2), dtype=jnp.float32)


class _JittableFakeModel(nnx.Module):
    def sample_actions(self, *_args, **_kwargs):
        return jnp.zeros((1, 3, 2), dtype=jnp.float32)

    def encode_guide(self, guide):
        return GuideMemory(
            tokens=guide.memory_source_index[..., None].astype(jnp.float32),
            token_mask=guide.memory_mask,
        )

    def sample_guided_actions_with_memory(self, _rng, observation, *, guide_memory, **_kwargs):
        groups, queries = observation.state.shape[:2]
        value = jnp.sum(
            jnp.where(
                guide_memory.token_mask[..., None],
                guide_memory.tokens,
                0,
            ),
            axis=(1, 2),
        )
        return jnp.broadcast_to(value[:, None, None, None], (groups, queries, 3, 2))


def _observation() -> dict:
    return {
        "image": {},
        "image_mask": {},
        "state": np.asarray([0.25, -0.5], dtype=np.float32),
        "tokenized_prompt": np.asarray([1, 2, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.asarray([True, True, False], dtype=np.bool_),
    }


def test_infer_uses_cached_memory_and_squeezes_output(monkeypatch):
    monkeypatch.setattr(_guided_policy.nnx_utils, "module_jit", lambda function: function)
    model = _FakeModel()
    policy = _guided_policy.GuidedPolicy(model)
    guide = _guide()
    policy.set_guide(guide)

    result = policy.infer(_observation())
    policy.infer(_observation())

    assert result["state"].shape == (2,)
    assert result["actions"].shape == (3, 2)
    assert model.encode_calls == [guide]
    assert len(model.calls) == 2
    _, observation, received_memory, _ = model.calls[0]
    assert observation.state.shape == (1, 1, 2)
    assert received_memory is policy.guide_memory


def test_set_guide_caches_by_object_identity_and_clear_invalidates(monkeypatch):
    monkeypatch.setattr(_guided_policy.nnx_utils, "module_jit", lambda function: function)
    model = _FakeModel()
    policy = _guided_policy.GuidedPolicy(model)
    guide = _guide()

    policy.set_guide(guide)
    first_memory = policy.guide_memory
    policy.set_guide(guide)

    assert model.encode_calls == [guide]
    assert policy.guide is guide
    assert policy.guide_memory is first_memory

    replacement = _guide()
    policy.set_guide(replacement)
    assert model.encode_calls == [guide, replacement]
    assert policy.guide is replacement
    assert policy.guide_memory is not first_memory

    policy.clear_guide()
    assert policy.guide is None
    assert policy.guide_memory is None
    with pytest.raises(RuntimeError, match="set_guide"):
        policy.infer(_observation())

    policy.set_guide(replacement)
    assert model.encode_calls == [guide, replacement, replacement]


def test_cached_memory_crosses_real_module_jit_boundary():
    policy = _guided_policy.GuidedPolicy(_JittableFakeModel())
    guide = _guide()

    policy.set_guide(guide)
    first_memory = policy.guide_memory
    first = policy.infer(_observation())["actions"]
    policy.set_guide(guide)
    second = policy.infer(_observation())["actions"]

    assert policy.guide_memory is first_memory
    np.testing.assert_array_equal(first, second)


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


def test_set_guide_rejects_invalid_memory_map_before_jit(monkeypatch):
    monkeypatch.setattr(_guided_policy.nnx_utils, "module_jit", lambda function: function)
    policy = _guided_policy.GuidedPolicy(_FakeModel())
    guide = _guide().replace(memory_source_index=_guide().memory_source_index.at[0, 0].set(99))

    with pytest.raises(ValueError, match="cover every valid"):
        policy.set_guide(guide)


def test_factory_rejects_pytorch_and_missing_params(tmp_path):
    config = SimpleNamespace()
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="JAX"):
        _guided_policy.create_trained_guided_policy(config, tmp_path)

    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="params"):
        _guided_policy.create_trained_guided_policy(config, tmp_path)
