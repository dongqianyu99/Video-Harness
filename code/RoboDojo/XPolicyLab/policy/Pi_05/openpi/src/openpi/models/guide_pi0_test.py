from __future__ import annotations

from types import SimpleNamespace

import flax.nnx as nnx
import jax

from openpi.models import guide_pi0
from openpi.models import guide_pi0_config
from openpi.models import model as _model
from openpi.models.pi0 import Pi0


def test_guide_pi0_is_a_pi0_subclass() -> None:
    assert issubclass(guide_pi0.GuidePi0, Pi0)


def test_guide_pi0_calls_parent_once_and_adds_only_guide_encoder(monkeypatch) -> None:
    parent_calls: list[tuple[object, object]] = []
    encoder_calls: list[dict[str, object]] = []
    gemma_calls: list[object] = []

    def fake_pi0_init(self, config, rngs):
        parent_calls.append((config, rngs))
        _model.BaseModel.__init__(
            self,
            config.action_dim,
            config.action_horizon,
            config.max_token_len,
        )

    class _FakeGuideFeatureEncoder(nnx.Module):
        def __init__(
            self,
            input_dim,
            output_dim,
            *,
            boundary_num_queries,
            transition_num_queries,
            width,
            num_heads,
            ffn_hidden_dim,
            rngs,
        ):
            encoder_calls.append(
                {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "boundary_num_queries": boundary_num_queries,
                    "transition_num_queries": transition_num_queries,
                    "width": width,
                    "num_heads": num_heads,
                    "ffn_hidden_dim": ffn_hidden_dim,
                    "rngs": rngs,
                }
            )

    def fake_get_config(variant):
        gemma_calls.append(variant)
        return SimpleNamespace(width=2048)

    def forbidden_backbone_constructor(*args, **kwargs):
        raise AssertionError("GuidePi0 must not construct another backbone")

    monkeypatch.setattr(Pi0, "__init__", fake_pi0_init)
    monkeypatch.setattr(guide_pi0, "GuideFeatureEncoder", _FakeGuideFeatureEncoder)
    monkeypatch.setattr(
        guide_pi0,
        "_gemma",
        SimpleNamespace(
            get_config=fake_get_config,
            Module=forbidden_backbone_constructor,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        guide_pi0,
        "_siglip",
        SimpleNamespace(Module=forbidden_backbone_constructor),
        raising=False,
    )

    config = guide_pi0_config.GuidePi0Config(
        guide_boundary_num_queries=6,
        guide_transition_num_queries=3,
        guide_resampler_width=512,
        guide_resampler_num_heads=8,
        guide_resampler_ffn_hidden_dim=1024,
    )
    rngs = nnx.Rngs(jax.random.key(0))

    model = guide_pi0.GuidePi0(config, rngs=rngs)

    assert len(parent_calls) == 1
    assert parent_calls[0][0] is config
    assert parent_calls[0][1] is rngs
    assert gemma_calls == [config.paligemma_variant]
    assert len(encoder_calls) == 1
    assert model.guide_encoder is not None

    encoder_kwargs = encoder_calls[0]
    assert encoder_kwargs["input_dim"] == 2048
    assert encoder_kwargs["output_dim"] == 2048
    assert encoder_kwargs["boundary_num_queries"] == config.guide_boundary_num_queries
    assert encoder_kwargs["transition_num_queries"] == config.guide_transition_num_queries
    assert encoder_kwargs["width"] == config.guide_resampler_width
    assert encoder_kwargs["num_heads"] == config.guide_resampler_num_heads
    assert encoder_kwargs["ffn_hidden_dim"] == config.guide_resampler_ffn_hidden_dim
    assert encoder_kwargs["rngs"] is rngs
