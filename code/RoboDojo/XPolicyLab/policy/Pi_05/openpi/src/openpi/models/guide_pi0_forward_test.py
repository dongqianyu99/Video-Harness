from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from openpi.models import guide_encoder
from openpi.models import guide_pi0
from openpi.models.guide_inputs import GuideInput

_GROUPS = 2
_FRAMES = 3
_PATCHES = 2
_UNITS = 2
_TEXT_TOKENS = 4
_BOUNDARY_TEXT_TOKENS = 3
_WIDTH = 5
_QUERIES = 3
_IMAGE_HEIGHT = 2
_IMAGE_WIDTH = 3


def _make_guide() -> GuideInput:
    return GuideInput(
        boundary_images=jnp.arange(
            _GROUPS * _FRAMES * 3 * _IMAGE_HEIGHT * _IMAGE_WIDTH * 3,
            dtype=jnp.float32,
        ).reshape(_GROUPS, _FRAMES, 3, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3),
        boundary_image_mask=jnp.ones((_GROUPS, _FRAMES, 3), dtype=jnp.bool_),
        boundary_text_tokens=jnp.arange(
            _GROUPS * _FRAMES * 3 * _BOUNDARY_TEXT_TOKENS,
            dtype=jnp.int32,
        ).reshape(_GROUPS, _FRAMES, 3, _BOUNDARY_TEXT_TOKENS),
        boundary_text_mask=jnp.ones(
            (_GROUPS, _FRAMES, 3, _BOUNDARY_TEXT_TOKENS), dtype=jnp.bool_
        ),
        transition_text_tokens=jnp.arange(
            _GROUPS * _UNITS * _TEXT_TOKENS,
            dtype=jnp.int32,
        ).reshape(_GROUPS, _UNITS, _TEXT_TOKENS),
        transition_text_mask=jnp.ones(
            (_GROUPS, _UNITS, _TEXT_TOKENS), dtype=jnp.bool_
        ),
        boundary_mask=jnp.ones((_GROUPS, _FRAMES), dtype=jnp.bool_),
        unit_mask=jnp.array(
            [[True, True], [True, False]],
            dtype=jnp.bool_,
        ),
        memory_source_kind=jnp.zeros((_GROUPS, 8), dtype=jnp.int32),
        memory_source_index=jnp.zeros((_GROUPS, 8), dtype=jnp.int32),
        memory_source_offset=jnp.zeros((_GROUPS, 8), dtype=jnp.int32),
        memory_mask=jnp.ones((_GROUPS, 8), dtype=jnp.bool_),
    )


class _RecordingImageEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def __call__(self, images, *, train: bool):
        self.calls.append((images.shape, train))
        batch = images.shape[0]
        tokens = jnp.arange(
            batch * _PATCHES * _WIDTH,
            dtype=jnp.float32,
        ).reshape(batch, _PATCHES, _WIDTH)
        return tokens, None


class _RecordingTextEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], str]] = []

    def __call__(self, token_ids, *, method: str):
        self.calls.append((token_ids.shape, method))
        batch, tokens = token_ids.shape
        return jnp.arange(
            batch * tokens * _WIDTH,
            dtype=jnp.float32,
        ).reshape(batch, tokens, _WIDTH)


class _RecordingGuideFeatureEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(
        self,
        boundary_image_tokens,
        boundary_image_mask,
        boundary_text_embeddings,
        boundary_text_mask,
        transition_text_embeddings,
        transition_text_mask,
        boundary_mask,
        unit_mask,
        memory_source_kind,
        memory_source_index,
        memory_source_offset,
        memory_mask,
    ) -> guide_encoder.GuideMemory:
        self.calls.append(
            (
                boundary_image_tokens,
                boundary_image_mask,
                boundary_text_embeddings,
                boundary_text_mask,
                transition_text_embeddings,
                transition_text_mask,
                boundary_mask,
                unit_mask,
                memory_source_kind,
                memory_source_index,
                memory_source_offset,
                memory_mask,
            )
        )
        tokens = jnp.ones(
            (_GROUPS, _UNITS * _QUERIES, _WIDTH),
            dtype=jnp.float32,
        )
        token_mask = jnp.repeat(unit_mask, _QUERIES, axis=1)
        return guide_encoder.GuideMemory(tokens=tokens, token_mask=token_mask)


class _RecordingControlPrefix:
    def __init__(self, tokens, token_mask) -> None:
        self.tokens = tokens
        self.token_mask = token_mask
        self.calls: list[object] = []

    def __call__(self, observation):
        self.calls.append(observation)
        return self.tokens, self.token_mask, jnp.zeros((self.tokens.shape[1],), dtype=jnp.bool_)


def test_encode_guide_uses_shared_backbones_once_and_preserves_group_shapes() -> None:
    image_encoder = _RecordingImageEncoder()
    text_embedder = _RecordingTextEmbedder()
    guide_encoder_stub = _RecordingGuideFeatureEncoder()
    model = SimpleNamespace(
        PaliGemma=SimpleNamespace(img=image_encoder, llm=text_embedder),
        guide_encoder=guide_encoder_stub,
    )
    guide = _make_guide()

    memory = guide_pi0.GuidePi0.encode_guide(model, guide)

    assert image_encoder.calls == [
        ((_GROUPS * _FRAMES * 3, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3), False)
    ]
    assert text_embedder.calls == [
        ((_GROUPS * _FRAMES * 3, _BOUNDARY_TEXT_TOKENS), "embed"),
        ((_GROUPS * _UNITS, _TEXT_TOKENS), "embed"),
    ]
    assert len(guide_encoder_stub.calls) == 1

    (
        boundary_image_tokens,
        boundary_image_mask,
        boundary_text_embeddings,
        boundary_text_mask,
        transition_text_embeddings,
        transition_text_mask,
        boundary_mask,
        unit_mask,
        memory_source_kind,
        memory_source_index,
        memory_source_offset,
        memory_mask,
    ) = guide_encoder_stub.calls[0]

    expected_boundary_image_tokens = jnp.arange(
        _GROUPS * _FRAMES * 3 * _PATCHES * _WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _FRAMES, 3, _PATCHES, _WIDTH)
    expected_boundary_text_embeddings = jnp.arange(
        _GROUPS * _FRAMES * 3 * _BOUNDARY_TEXT_TOKENS * _WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _FRAMES, 3, _BOUNDARY_TEXT_TOKENS, _WIDTH)
    expected_transition_text_embeddings = jnp.arange(
        _GROUPS * _UNITS * _TEXT_TOKENS * _WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _UNITS, _TEXT_TOKENS, _WIDTH)

    np.testing.assert_array_equal(boundary_image_tokens, expected_boundary_image_tokens)
    np.testing.assert_array_equal(
        boundary_text_embeddings, expected_boundary_text_embeddings
    )
    np.testing.assert_array_equal(
        transition_text_embeddings, expected_transition_text_embeddings
    )
    np.testing.assert_array_equal(boundary_image_mask, guide.boundary_image_mask)
    np.testing.assert_array_equal(boundary_text_mask, guide.boundary_text_mask)
    np.testing.assert_array_equal(transition_text_mask, guide.transition_text_mask)
    np.testing.assert_array_equal(boundary_mask, guide.boundary_mask)
    np.testing.assert_array_equal(unit_mask, guide.unit_mask)
    np.testing.assert_array_equal(memory_source_kind, guide.memory_source_kind)
    np.testing.assert_array_equal(memory_source_index, guide.memory_source_index)
    np.testing.assert_array_equal(memory_source_offset, guide.memory_source_offset)
    np.testing.assert_array_equal(memory_mask, guide.memory_mask)

    assert memory.tokens.shape == (_GROUPS, _UNITS * _QUERIES, _WIDTH)
    assert memory.token_mask.shape == (_GROUPS, _UNITS * _QUERIES)
    np.testing.assert_array_equal(
        memory.token_mask,
        jnp.repeat(guide.unit_mask, _QUERIES, axis=1),
    )


def test_encode_guide_does_not_add_a_third_backbone_expert() -> None:
    image_encoder = _RecordingImageEncoder()
    text_embedder = _RecordingTextEmbedder()
    model = SimpleNamespace(
        PaliGemma=SimpleNamespace(img=image_encoder, llm=text_embedder),
        guide_encoder=_RecordingGuideFeatureEncoder(),
    )

    guide_pi0.GuidePi0.encode_guide(model, _make_guide())

    assert set(vars(model.PaliGemma)) == {"img", "llm"}


def test_embed_guide_control_prefix_broadcasts_guide_and_calls_native_prefix_once() -> None:
    guide_memory = guide_encoder.GuideMemory(
        tokens=jnp.arange(_GROUPS * 2 * _WIDTH, dtype=jnp.float32).reshape(_GROUPS, 2, _WIDTH),
        token_mask=jnp.array(
            [[True, False], [True, True]],
            dtype=jnp.bool_,
        ),
    )
    control_tokens = jnp.arange(
        _GROUPS * 3 * _QUERIES * _WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS * _QUERIES, 3, _WIDTH)
    control_mask = jnp.array(
        [[True, True, False]] * (_GROUPS * _QUERIES),
        dtype=jnp.bool_,
    )
    control_prefix = _RecordingControlPrefix(control_tokens, control_mask)
    flat_observation = object()
    model = SimpleNamespace(embed_prefix=control_prefix)

    combined_tokens, guide_mask, returned_control_mask = (
        guide_pi0.GuidePi0._embed_guide_control_prefix(  # noqa: SLF001
            model,
            guide_memory,
            flat_observation,
            queries_per_guide=_QUERIES,
        )
    )

    expected_guide_tokens = jnp.repeat(guide_memory.tokens, _QUERIES, axis=0)
    expected_guide_mask = jnp.repeat(guide_memory.token_mask, _QUERIES, axis=0)
    expected_tokens = jnp.concatenate([expected_guide_tokens, control_tokens], axis=1)

    assert control_prefix.calls == [flat_observation]
    np.testing.assert_array_equal(combined_tokens, expected_tokens)
    np.testing.assert_array_equal(guide_mask, expected_guide_mask)
    np.testing.assert_array_equal(returned_control_mask, control_mask)
