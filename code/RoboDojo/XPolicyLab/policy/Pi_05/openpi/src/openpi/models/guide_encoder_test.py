import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import guide_encoder
from openpi.models import guide_tokens
from openpi.shared import nnx_utils

_D = 6
_KB = 2
_KT = 1


def _make_encoder() -> guide_encoder.GuideFeatureEncoder:
    return guide_encoder.GuideFeatureEncoder(
        input_dim=_D,
        output_dim=_D,
        boundary_num_queries=_KB,
        transition_num_queries=_KT,
        width=8,
        num_heads=2,
        ffn_hidden_dim=16,
        rngs=nnx.Rngs(0),
    )


def _inputs():
    boundary_images = jax.random.normal(jax.random.key(1), (1, 3, 3, 2, _D))
    boundary_image_mask = jnp.ones((1, 3, 3), dtype=jnp.bool_)
    boundary_text = jax.random.normal(jax.random.key(2), (1, 3, 3, 2, _D))
    boundary_text_mask = jnp.ones((1, 3, 3, 2), dtype=jnp.bool_)
    transition_text = jax.random.normal(jax.random.key(3), (1, 2, 3, _D))
    transition_text_mask = jnp.ones((1, 2, 3), dtype=jnp.bool_)
    boundary_mask = jnp.ones((1, 3), dtype=jnp.bool_)
    unit_mask = jnp.ones((1, 2), dtype=jnp.bool_)
    # B0, T0, B1, T1, B2.
    source_kind = jnp.array([[0, 0, 1, 0, 0, 1, 0, 0]], dtype=jnp.int32)
    source_index = jnp.array([[0, 0, 0, 1, 1, 1, 2, 2]], dtype=jnp.int32)
    source_offset = jnp.array([[0, 1, 0, 0, 1, 0, 0, 1]], dtype=jnp.int32)
    memory_mask = jnp.ones((1, 8), dtype=jnp.bool_)
    return (
        boundary_images,
        boundary_image_mask,
        boundary_text,
        boundary_text_mask,
        transition_text,
        transition_text_mask,
        boundary_mask,
        unit_mask,
        source_kind,
        source_index,
        source_offset,
        memory_mask,
    )


def test_hierarchical_encoder_shape_and_jit() -> None:
    encoder = _make_encoder()
    inputs = _inputs()

    eager = encoder(*inputs)
    compiled = nnx_utils.module_jit(encoder.__call__)(*inputs)

    assert eager.tokens.shape == (1, 8, _D)
    assert eager.token_mask.shape == (1, 8)
    np.testing.assert_allclose(compiled.tokens, eager.tokens, rtol=1e-5, atol=1e-6)


def test_transition_resampler_consumes_only_transition_text() -> None:
    encoder = _make_encoder()
    inputs = _inputs()
    transition_batch = guide_tokens.assemble_transition_tokens(
        inputs[4],
        inputs[5],
        inputs[7],
    )

    original = encoder.transition_resampler(
        transition_batch.tokens,
        transition_batch.token_mask,
        transition_batch.role_ids,
        inputs[7],
    )
    changed_boundary_inputs = inputs[0] + 10_000.0
    del changed_boundary_inputs  # Boundary features are deliberately absent from this call.
    repeated = encoder.transition_resampler(
        transition_batch.tokens,
        transition_batch.token_mask,
        transition_batch.role_ids,
        inputs[7],
    )

    np.testing.assert_array_equal(repeated, original)
    assert original.shape == (1, 2, _KT, _D)


def test_padding_boundary_and_unit_produce_zero_source_memory() -> None:
    encoder = _make_encoder()
    inputs = list(_inputs())
    inputs[6] = jnp.array([[True, True, False]])
    inputs[7] = jnp.array([[True, False]])
    inputs[8] = jnp.array([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=jnp.int32)
    inputs[9] = jnp.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    inputs[10] = jnp.array([[0, 1, 0, 0, 1, 0, 0, 0]], dtype=jnp.int32)
    inputs[11] = jnp.array([[True, True, True, True, True, False, False, False]])

    memory = encoder(*inputs)

    np.testing.assert_array_equal(memory.tokens[0, 5:], 0.0)
    np.testing.assert_array_equal(memory.token_mask, inputs[11])


def test_parameter_tree_contains_two_resamplers_and_memory_types() -> None:
    paths = set(nnx.state(_make_encoder(), nnx.Param).flat_state())

    assert any(path[0] == "boundary_resampler" for path in paths)
    assert any(path[0] == "transition_resampler" for path in paths)
    assert ("memory_type_embeddings",) in paths
    assert not any(path[0] == "unit_resampler" for path in paths)


def test_shared_backbones_encode_three_views_and_both_text_banks() -> None:
    image_calls = []
    text_calls = []

    def image_encoder(images, *, train):
        image_calls.append((images.shape, train))
        return jnp.ones((images.shape[0], 4, _D)), None

    def text_embedder(tokens, *, method):
        text_calls.append((tokens.shape, method))
        return jnp.ones((*tokens.shape, _D))

    features = guide_encoder.encode_shared_guide_features(
        jnp.ones((2, 3, 3, 8, 8, 3)),
        jnp.ones((2, 3, 3, 5), dtype=jnp.int32),
        jnp.ones((2, 4, 7), dtype=jnp.int32),
        image_encoder=image_encoder,
        text_embedder=text_embedder,
    )

    assert image_calls == [((18, 8, 8, 3), False)]
    assert text_calls == [((18, 5), "embed"), ((8, 7), "embed")]
    assert features.boundary_image_tokens.shape == (2, 3, 3, 4, _D)
    assert features.boundary_text_embeddings.shape == (2, 3, 3, 5, _D)
    assert features.transition_text_embeddings.shape == (2, 4, 7, _D)
