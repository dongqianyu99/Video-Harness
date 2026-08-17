from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import guide_encoder
from openpi.models import guide_tokens
from openpi.shared import nnx_utils

_GROUPS = 2
_FRAMES = 3
_PATCHES = 2
_UNITS = 3
_QUERIES = 3
_TEXT_TOKENS = 2
_INPUT_DIM = 4
_OUTPUT_DIM = 5
_RESAMPLER_WIDTH = 4
_NUM_HEADS = 2
_FFN_HIDDEN_DIM = 8
_WIDTH = 2
_IMAGE_HEIGHT = 2
_IMAGE_WIDTH = 3
_IMAGE_CHANNELS = 3
_FEATURE_WIDTH = 4
_VOCAB_SIZE = 32


def _make_inputs() -> tuple[jax.Array, jax.Array]:
    unit_memory = jnp.arange(
        _GROUPS * _UNITS * _QUERIES * _WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _UNITS, _QUERIES, _WIDTH)
    unit_mask = jnp.ones((_GROUPS, _UNITS), dtype=jnp.bool_)
    return unit_memory, unit_mask


def _flatten(inputs: tuple[jax.Array, jax.Array]) -> guide_encoder.GuideMemory:
    return guide_encoder.flatten_unit_memory(*inputs)


def _make_encoder(*, seed: int = 0) -> guide_encoder.GuideFeatureEncoder:
    return guide_encoder.GuideFeatureEncoder(
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        num_queries=_QUERIES,
        width=_RESAMPLER_WIDTH,
        num_heads=_NUM_HEADS,
        ffn_hidden_dim=_FFN_HIDDEN_DIM,
        rngs=nnx.Rngs(seed),
    )


def _make_feature_inputs(*, masked: bool = False):
    frame_tokens = jnp.arange(
        _GROUPS * _FRAMES * _PATCHES * _INPUT_DIM,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _FRAMES, _PATCHES, _INPUT_DIM)
    text_embeddings = (
        jnp.arange(
            _GROUPS * _UNITS * _TEXT_TOKENS * _INPUT_DIM,
            dtype=jnp.float32,
        ).reshape(_GROUPS, _UNITS, _TEXT_TOKENS, _INPUT_DIM)
        + 1000.0
    )

    frame_mask = jnp.ones((_GROUPS, _FRAMES), dtype=jnp.bool_)
    text_mask = jnp.ones((_GROUPS, _UNITS, _TEXT_TOKENS), dtype=jnp.bool_)
    unit_mask = jnp.ones((_GROUPS, _UNITS), dtype=jnp.bool_)
    before_slot = jnp.array(
        [[0, 1, 2], [1, 2, 0]],
        dtype=jnp.int32,
    )
    after_slot = jnp.array(
        [[1, 2, 0], [2, 0, 1]],
        dtype=jnp.int32,
    )

    if masked:
        frame_mask = jnp.array(
            [[True, False, True], [False, True, True]],
            dtype=jnp.bool_,
        )
        text_mask = jnp.array(
            [
                [[True, False], [False, True], [True, True]],
                [[False, True], [True, False], [True, True]],
            ],
            dtype=jnp.bool_,
        )
        unit_mask = jnp.array(
            [[True, True, False], [True, False, True]],
            dtype=jnp.bool_,
        )
        before_slot = jnp.array(
            [[0, 1, -1], [2, -1, 0]],
            dtype=jnp.int32,
        )
        after_slot = jnp.array(
            [[1, 2, -1], [0, -1, 1]],
            dtype=jnp.int32,
        )

    return (
        frame_tokens,
        frame_mask,
        text_embeddings,
        text_mask,
        unit_mask,
        before_slot,
        after_slot,
    )


def _make_backbone_inputs() -> tuple[jax.Array, jax.Array]:
    images = jnp.arange(
        _GROUPS * _FRAMES * _IMAGE_HEIGHT * _IMAGE_WIDTH * _IMAGE_CHANNELS,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _FRAMES, _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS)
    text_token_ids = jnp.arange(
        _GROUPS * _UNITS * _TEXT_TOKENS,
        dtype=jnp.int32,
    ).reshape(_GROUPS, _UNITS, _TEXT_TOKENS)
    return images, text_token_ids


class _RecordingImageEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def __call__(self, images: jax.Array, *, train: bool):
        self.calls.append((images.shape, train))
        batch = images.shape[0]
        tokens = jnp.arange(
            batch * _PATCHES * _FEATURE_WIDTH,
            dtype=jnp.float32,
        ).reshape(batch, _PATCHES, _FEATURE_WIDTH)
        return tokens, None


class _RecordingTextEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], str]] = []

    def __call__(self, token_ids: jax.Array, *, method: str):
        self.calls.append((token_ids.shape, method))
        batch, tokens = token_ids.shape
        return jnp.arange(
            batch * tokens * _FEATURE_WIDTH,
            dtype=jnp.float32,
        ).reshape(batch, tokens, _FEATURE_WIDTH)


def _pure_image_encoder(images: jax.Array, *, train: bool):
    if train:
        raise ValueError("test image encoder must be called with train=False")
    pooled = jnp.mean(images, axis=(1, 2, 3))
    tokens = jnp.broadcast_to(
        pooled[:, None, None],
        (images.shape[0], _PATCHES, _FEATURE_WIDTH),
    )
    return tokens, None


def _pure_text_embedder(token_ids: jax.Array, *, method: str):
    if method != "embed":
        raise ValueError("test text embedder must be called with method='embed'")
    values = token_ids.astype(jnp.float32)[..., None]
    return jnp.broadcast_to(
        values,
        (token_ids.shape[0], token_ids.shape[1], _FEATURE_WIDTH),
    )


def test_guide_memory_is_a_jax_pytree() -> None:
    memory = guide_encoder.GuideMemory(
        tokens=jnp.zeros((_GROUPS, _UNITS * _QUERIES, _WIDTH), dtype=jnp.float32),
        token_mask=jnp.ones((_GROUPS, _UNITS * _QUERIES), dtype=jnp.bool_),
    )

    leaves = jax.tree_util.tree_leaves(memory)
    assert len(leaves) == 2

    mapped = jax.tree_util.tree_map(lambda leaf: leaf, memory)
    assert jax.tree_util.tree_structure(mapped) == jax.tree_util.tree_structure(memory)

    jitted = jax.jit(lambda value: value)(memory)
    np.testing.assert_array_equal(jitted.tokens, memory.tokens)
    np.testing.assert_array_equal(jitted.token_mask, memory.token_mask)


def test_flatten_uses_unit_major_query_minor_order() -> None:
    unit_memory, unit_mask = _make_inputs()

    output = _flatten((unit_memory, unit_mask))
    expected = unit_memory.reshape(_GROUPS, _UNITS * _QUERIES, _WIDTH)

    assert output.tokens.shape == (_GROUPS, _UNITS * _QUERIES, _WIDTH)
    np.testing.assert_array_equal(output.tokens, expected)


def test_token_mask_repeats_each_unit_mask_for_all_queries() -> None:
    unit_memory, _ = _make_inputs()
    unit_mask = jnp.array(
        [[True, False, True], [False, True, True]],
        dtype=jnp.bool_,
    )

    output = _flatten((unit_memory, unit_mask))
    expected_mask = jnp.repeat(unit_mask, _QUERIES, axis=1)

    np.testing.assert_array_equal(output.token_mask, expected_mask)


def test_masked_units_are_zeroed_even_when_input_memory_is_nonzero() -> None:
    unit_memory, _ = _make_inputs()
    unit_memory = unit_memory.at[0, 1].set(999.0)
    unit_memory = unit_memory.at[1, 2].set(777.0)
    unit_mask = jnp.array(
        [[True, False, True], [True, True, False]],
        dtype=jnp.bool_,
    )

    output = _flatten((unit_memory, unit_mask))

    np.testing.assert_array_equal(output.tokens[0, _QUERIES : 2 * _QUERIES], 0.0)
    np.testing.assert_array_equal(output.tokens[1, 2 * _QUERIES :], 0.0)
    assert np.all(output.token_mask[0, _QUERIES : 2 * _QUERIES] == 0)
    assert np.all(output.token_mask[1, 2 * _QUERIES :] == 0)


def test_groups_remain_independent_during_flatten() -> None:
    group_zero = jnp.arange(
        _UNITS * _QUERIES * _WIDTH,
        dtype=jnp.float32,
    ).reshape(1, _UNITS, _QUERIES, _WIDTH)
    unit_memory = jnp.concatenate([group_zero, group_zero + 1000.0], axis=0)
    unit_mask = jnp.ones((_GROUPS, _UNITS), dtype=jnp.bool_)

    output = _flatten((unit_memory, unit_mask))

    np.testing.assert_array_equal(output.tokens[0], group_zero.reshape(_UNITS * _QUERIES, _WIDTH))
    np.testing.assert_array_equal(
        output.tokens[1],
        (group_zero + 1000.0).reshape(_UNITS * _QUERIES, _WIDTH),
    )


def test_jit_flatten_matches_eager() -> None:
    inputs = _make_inputs()
    eager = _flatten(inputs)
    jitted = jax.jit(_flatten)(inputs)

    np.testing.assert_array_equal(jitted.tokens, eager.tokens)
    np.testing.assert_array_equal(jitted.token_mask, eager.token_mask)


def test_shape_and_dtype_errors_fail_clearly_in_eager_mode() -> None:
    unit_memory, unit_mask = _make_inputs()

    with pytest.raises(ValueError, match="unit_memory"):
        _flatten((unit_memory[0], unit_mask))

    with pytest.raises(ValueError, match="unit_mask"):
        _flatten((unit_memory, unit_mask[..., None]))

    with pytest.raises(ValueError, match="floating"):
        _flatten((unit_memory.astype(jnp.int32), unit_mask))

    with pytest.raises(ValueError, match="bool"):
        _flatten((unit_memory, unit_mask.astype(jnp.int32)))


def test_feature_encoder_matches_the_explicit_pipeline() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs(masked=True)

    actual = encoder(*inputs)
    assembled = guide_tokens.assemble_unit_tokens(*inputs)
    unit_memory = encoder.unit_resampler(
        assembled.tokens,
        assembled.token_mask,
        assembled.role_ids,
        inputs[4],
    )
    expected = guide_encoder.flatten_unit_memory(unit_memory, inputs[4])

    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_array_equal(actual.token_mask, expected.token_mask)


def test_feature_encoder_output_shape_and_mask() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs(masked=True)

    output = encoder(*inputs)
    expected_mask = jnp.repeat(inputs[4], _QUERIES, axis=1)

    assert output.tokens.shape == (_GROUPS, _UNITS * _QUERIES, _OUTPUT_DIM)
    assert output.token_mask.shape == (_GROUPS, _UNITS * _QUERIES)
    np.testing.assert_array_equal(output.token_mask, expected_mask)


def test_feature_encoder_jit_matches_eager() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs(masked=True)

    eager = encoder(*inputs)
    jitted = nnx_utils.module_jit(encoder.__call__)(*inputs)

    np.testing.assert_allclose(jitted.tokens, eager.tokens, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(jitted.token_mask, eager.token_mask)


def test_masked_frame_and_text_content_do_not_change_output() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs(masked=True)
    frame_tokens, frame_mask, text_embeddings, text_mask, unit_mask, before_slot, after_slot = inputs

    changed_frame_tokens = frame_tokens + jnp.where(
        frame_mask[..., None, None],
        0.0,
        10_000.0,
    )
    changed_text_embeddings = text_embeddings + jnp.where(
        text_mask[..., None],
        0.0,
        10_000.0,
    )
    changed_inputs = (
        changed_frame_tokens,
        frame_mask,
        changed_text_embeddings,
        text_mask,
        unit_mask,
        before_slot,
        after_slot,
    )

    original = encoder(*inputs)
    changed = encoder(*changed_inputs)

    np.testing.assert_array_equal(changed.tokens, original.tokens)
    np.testing.assert_array_equal(changed.token_mask, original.token_mask)


def test_groups_do_not_affect_each_other() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs()
    frame_tokens, frame_mask, text_embeddings, text_mask, unit_mask, before_slot, after_slot = inputs

    changed_inputs = (
        frame_tokens.at[1].add(10_000.0),
        frame_mask,
        text_embeddings.at[1].add(10_000.0),
        text_mask,
        unit_mask,
        before_slot,
        after_slot,
    )

    original = encoder(*inputs)
    changed = encoder(*changed_inputs)

    np.testing.assert_array_equal(changed.tokens[0], original.tokens[0])
    np.testing.assert_array_equal(changed.token_mask[0], original.token_mask[0])


def test_gradients_reach_features_and_resampler_parameters() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs()
    _, frame_mask, _, text_mask, unit_mask, before_slot, after_slot = inputs

    def loss_fn(
        module: guide_encoder.GuideFeatureEncoder,
        frame_tokens: jax.Array,
        text_embeddings: jax.Array,
    ) -> jax.Array:
        output = module(
            frame_tokens,
            frame_mask,
            text_embeddings,
            text_mask,
            unit_mask,
            before_slot,
            after_slot,
        )
        weights = jnp.linspace(
            0.5,
            1.5,
            output.tokens.size,
            dtype=output.tokens.dtype,
        ).reshape(output.tokens.shape)
        return jnp.mean(jnp.square(output.tokens) * weights)

    parameter_grads, frame_grads, text_grads = nnx.grad(
        loss_fn,
        argnums=(0, 1, 2),
    )(encoder, inputs[0], inputs[2])
    flat_grads = {path: variable.value for path, variable in parameter_grads.flat_state().items()}

    for path in (
        ("unit_resampler", "learned_queries"),
        ("unit_resampler", "input_projection", "kernel"),
        ("unit_resampler", "output_projection", "kernel"),
    ):
        assert path in flat_grads
        assert np.any(np.abs(flat_grads[path]) > 1e-9), f"Expected nonzero gradient for {path}"

    assert np.any(np.abs(np.asarray(frame_grads)[np.asarray(frame_mask)]) > 1e-12)
    assert np.any(np.abs(np.asarray(text_grads)[np.asarray(text_mask)]) > 1e-12)


def test_parameter_tree_contains_only_the_unit_resampler() -> None:
    encoder = _make_encoder()
    parameter_state = nnx.state(encoder, nnx.Param).flat_state()

    assert parameter_state
    for path in parameter_state:
        path_parts = tuple(str(part) for part in path)
        assert path_parts[0] == "unit_resampler"
        assert not any(
            forbidden in "/".join(path_parts).lower()
            for forbidden in ("img", "llm", "paligemma", "siglip", "backbone")
        )

    assert not hasattr(encoder, "img")
    assert not hasattr(encoder, "llm")
    assert not hasattr(encoder, "paligemma")


def test_invalid_inputs_fail_in_assembly_or_resampler() -> None:
    encoder = _make_encoder()
    inputs = _make_feature_inputs()

    with pytest.raises(ValueError, match="text_embeddings"):
        encoder(
            inputs[0],
            inputs[1],
            inputs[2][..., :-1],
            *inputs[3:],
        )

    assembled = guide_tokens.assemble_unit_tokens(*inputs)
    with pytest.raises(ValueError, match="input_dim"):
        encoder.unit_resampler(
            assembled.tokens[..., :-1],
            assembled.token_mask,
            assembled.role_ids,
            inputs[4],
        )

    with pytest.raises(ValueError, match="unit_mask"):
        encoder.unit_resampler(
            assembled.tokens,
            assembled.token_mask,
            assembled.role_ids,
            inputs[4][..., None],
        )


def test_shared_backbone_features_are_a_jax_pytree() -> None:
    features = guide_encoder.GuideBackboneFeatures(
        frame_tokens=jnp.zeros(
            (_GROUPS, _FRAMES, _PATCHES, _FEATURE_WIDTH),
            dtype=jnp.float32,
        ),
        text_embeddings=jnp.zeros(
            (_GROUPS, _UNITS, _TEXT_TOKENS, _FEATURE_WIDTH),
            dtype=jnp.float32,
        ),
    )

    leaves = jax.tree_util.tree_leaves(features)
    assert len(leaves) == 2
    assert all(not isinstance(leaf, nnx.Param) for leaf in leaves)

    mapped = jax.tree_util.tree_map(lambda leaf: leaf, features)
    assert jax.tree_util.tree_structure(mapped) == jax.tree_util.tree_structure(features)

    jitted = jax.jit(lambda value: value)(features)
    np.testing.assert_array_equal(jitted.frame_tokens, features.frame_tokens)
    np.testing.assert_array_equal(jitted.text_embeddings, features.text_embeddings)


def test_shared_adapter_flattens_calls_and_restores_group_major_shapes() -> None:
    images, text_token_ids = _make_backbone_inputs()
    image_encoder = _RecordingImageEncoder()
    text_embedder = _RecordingTextEmbedder()

    features = guide_encoder.encode_shared_guide_features(
        images,
        text_token_ids,
        image_encoder=image_encoder,
        text_embedder=text_embedder,
    )

    assert image_encoder.calls == [
        ((_GROUPS * _FRAMES, _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS), False)
    ]
    assert text_embedder.calls == [((_GROUPS * _UNITS, _TEXT_TOKENS), "embed")]

    expected_frame_tokens = jnp.arange(
        _GROUPS * _FRAMES * _PATCHES * _FEATURE_WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _FRAMES, _PATCHES, _FEATURE_WIDTH)
    expected_text_embeddings = jnp.arange(
        _GROUPS * _UNITS * _TEXT_TOKENS * _FEATURE_WIDTH,
        dtype=jnp.float32,
    ).reshape(_GROUPS, _UNITS, _TEXT_TOKENS, _FEATURE_WIDTH)

    assert features.frame_tokens.shape == expected_frame_tokens.shape
    assert features.text_embeddings.shape == expected_text_embeddings.shape
    np.testing.assert_array_equal(features.frame_tokens, expected_frame_tokens)
    np.testing.assert_array_equal(features.text_embeddings, expected_text_embeddings)


def test_shared_adapter_with_pure_fakes_is_jittable() -> None:
    images, text_token_ids = _make_backbone_inputs()

    def run(
        image_values: jax.Array,
        token_values: jax.Array,
    ) -> guide_encoder.GuideBackboneFeatures:
        return guide_encoder.encode_shared_guide_features(
            image_values,
            token_values,
            image_encoder=_pure_image_encoder,
            text_embedder=_pure_text_embedder,
        )

    eager = run(images, text_token_ids)
    jitted = jax.jit(run)(images, text_token_ids)

    np.testing.assert_array_equal(jitted.frame_tokens, eager.frame_tokens)
    np.testing.assert_array_equal(jitted.text_embeddings, eager.text_embeddings)


def test_shared_adapter_preserves_gradients_to_images_and_text_table() -> None:
    images, text_token_ids = _make_backbone_inputs()
    table = jnp.arange(
        _VOCAB_SIZE * _FEATURE_WIDTH,
        dtype=jnp.float32,
    ).reshape(_VOCAB_SIZE, _FEATURE_WIDTH)

    def loss_fn(
        image_values: jax.Array,
        embedding_table: jax.Array,
    ) -> jax.Array:
        def table_text_embedder(token_ids: jax.Array, *, method: str):
            if method != "embed":
                raise ValueError("expected embed method")
            return embedding_table[token_ids]

        features = guide_encoder.encode_shared_guide_features(
            image_values,
            text_token_ids,
            image_encoder=_pure_image_encoder,
            text_embedder=table_text_embedder,
        )
        return jnp.sum(features.frame_tokens**2) + jnp.sum(features.text_embeddings**2)

    image_grads, table_grads = jax.grad(loss_fn, argnums=(0, 1))(images, table)

    assert np.any(np.abs(np.asarray(image_grads)) > 1e-12)
    assert np.any(np.abs(np.asarray(table_grads)) > 1e-12)


def test_shared_adapter_input_shape_dtype_and_channel_errors_fail_clearly() -> None:
    images, text_token_ids = _make_backbone_inputs()

    bad_images = (
        (images[0], text_token_ids, "images"),
        (images.astype(jnp.uint8), text_token_ids, "images"),
        (images[..., :-1], text_token_ids, "images"),
        (images, text_token_ids[0], "text_token_ids"),
        (images, text_token_ids.astype(jnp.float32), "text_token_ids"),
        (images, text_token_ids[:1], "text_token_ids"),
    )

    for bad_images_value, bad_text_ids, message in bad_images:
        with pytest.raises(ValueError, match=message):
            guide_encoder.encode_shared_guide_features(
                bad_images_value,
                bad_text_ids,
                image_encoder=_pure_image_encoder,
                text_embedder=_pure_text_embedder,
            )


@pytest.mark.parametrize(
    "failure_kind",
    ["image_rank", "image_leading", "text_rank", "text_leading", "text_length", "width"],
)
def test_shared_adapter_encoder_output_errors_fail_clearly(failure_kind: str) -> None:
    images, text_token_ids = _make_backbone_inputs()

    def image_encoder(flat_images: jax.Array, *, train: bool):
        del train
        batch = flat_images.shape[0]
        if failure_kind == "image_rank":
            return jnp.zeros((batch, _PATCHES * _FEATURE_WIDTH), dtype=jnp.float32), None
        if failure_kind == "image_leading":
            return jnp.zeros((batch - 1, _PATCHES, _FEATURE_WIDTH), dtype=jnp.float32), None
        return jnp.zeros((batch, _PATCHES, _FEATURE_WIDTH), dtype=jnp.float32), None

    def text_embedder(flat_text_ids: jax.Array, *, method: str):
        del method
        batch, tokens = flat_text_ids.shape
        if failure_kind == "text_rank":
            return jnp.zeros((batch, tokens * _FEATURE_WIDTH), dtype=jnp.float32)
        if failure_kind == "text_leading":
            return jnp.zeros((batch - 1, tokens, _FEATURE_WIDTH), dtype=jnp.float32)
        if failure_kind == "text_length":
            return jnp.zeros((batch, tokens + 1, _FEATURE_WIDTH), dtype=jnp.float32)
        if failure_kind == "width":
            return jnp.zeros((batch, tokens, _FEATURE_WIDTH + 1), dtype=jnp.float32)
        return jnp.zeros((batch, tokens, _FEATURE_WIDTH), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"(image|text|width)"):
        guide_encoder.encode_shared_guide_features(
            images,
            text_token_ids,
            image_encoder=image_encoder,
            text_embedder=text_embedder,
        )
