from __future__ import annotations

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.training import guide_weight_loaders

_GUIDE_PREFIX = ("guide_encoder",)


def _reference_params() -> dict[str, dict[str, object]]:
    return {
        "PaliGemma": {
            "kernel": jax.ShapeDtypeStruct((2, 3), jnp.float32),
        },
        "action_in_proj": {
            "kernel": jax.ShapeDtypeStruct((3, 4), jnp.float16),
        },
        "time_mlp": {
            "bias": jax.ShapeDtypeStruct((4,), jnp.float32),
        },
        "guide_encoder": {
            "learned_queries": jax.ShapeDtypeStruct((2, 4), jnp.float32),
            "output_projection": {
                "kernel": jax.ShapeDtypeStruct((4, 5), jnp.float32),
            },
        },
    }


def _loaded_native_params() -> dict[str, dict[str, object]]:
    return {
        "PaliGemma": {
            "kernel": np.arange(6, dtype=np.float16).reshape(2, 3),
        },
        "action_in_proj": {
            "kernel": np.arange(12, dtype=np.float32).reshape(3, 4),
        },
        "time_mlp": {
            "bias": np.arange(4, dtype=np.float16),
        },
    }


def _flat(tree):
    return flax.traverse_util.flatten_dict(tree)


def _assert_structure_shape_and_dtype_match(reference, result) -> None:
    flat_reference = _flat(reference)
    flat_result = _flat(result)
    assert set(flat_result) == set(flat_reference)

    for path, reference_leaf in flat_reference.items():
        result_leaf = flat_result[path]
        assert result_leaf.shape == reference_leaf.shape, path
        assert result_leaf.dtype == reference_leaf.dtype, path


def test_strict_merge_replaces_native_and_preserves_guide_leaves() -> None:
    reference = _reference_params()
    loaded = _loaded_native_params()

    result = guide_weight_loaders.strict_merge_pi05_base_params(loaded, reference)

    _assert_structure_shape_and_dtype_match(reference, result)

    flat_reference = _flat(reference)
    flat_loaded = _flat(loaded)
    flat_result = _flat(result)

    for path, loaded_leaf in flat_loaded.items():
        np.testing.assert_array_equal(flat_result[path], loaded_leaf)

    for path, reference_leaf in flat_reference.items():
        if path[:1] == _GUIDE_PREFIX:
            assert flat_result[path] is reference_leaf


def test_strict_merge_casts_native_dtype_to_reference_dtype() -> None:
    result = guide_weight_loaders.strict_merge_pi05_base_params(
        _loaded_native_params(),
        _reference_params(),
    )

    assert result["PaliGemma"]["kernel"].dtype == np.dtype(np.float32)
    assert result["action_in_proj"]["kernel"].dtype == np.dtype(np.float16)
    assert result["time_mlp"]["bias"].dtype == np.dtype(np.float32)


def test_merged_tree_can_be_reduced_to_native_partial_params() -> None:
    result = guide_weight_loaders.strict_merge_pi05_base_params(
        _loaded_native_params(),
        _reference_params(),
    )

    flat_result = _flat(result)
    native_partial = {
        path: leaf
        for path, leaf in flat_result.items()
        if not isinstance(leaf, jax.ShapeDtypeStruct)
    }

    assert set(native_partial) == set(_flat(_loaded_native_params()))
    assert all(isinstance(leaf, np.ndarray) for leaf in native_partial.values())


def test_strict_merge_requires_a_guide_leaf_in_reference() -> None:
    reference = _reference_params()
    reference.pop("guide_encoder")

    with pytest.raises(ValueError, match="guide_encoder"):
        guide_weight_loaders.strict_merge_pi05_base_params(
            _loaded_native_params(),
            reference,
        )


def test_strict_merge_rejects_missing_native_key() -> None:
    loaded = _loaded_native_params()
    loaded.pop("time_mlp")

    with pytest.raises(ValueError, match="time_mlp/bias"):
        guide_weight_loaders.strict_merge_pi05_base_params(loaded, _reference_params())


def test_strict_merge_rejects_unknown_extra_key() -> None:
    loaded = _loaded_native_params()
    loaded["unexpected"] = {"kernel": np.zeros((1,), dtype=np.float32)}

    with pytest.raises(ValueError, match="unexpected/kernel"):
        guide_weight_loaders.strict_merge_pi05_base_params(loaded, _reference_params())


def test_strict_merge_rejects_guide_key_in_base_checkpoint() -> None:
    loaded = _loaded_native_params()
    loaded["guide_encoder"] = {
        "learned_queries": np.zeros((2, 4), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="guide_encoder/learned_queries"):
        guide_weight_loaders.strict_merge_pi05_base_params(loaded, _reference_params())


def test_strict_merge_rejects_native_shape_mismatch() -> None:
    loaded = _loaded_native_params()
    loaded["PaliGemma"]["kernel"] = np.zeros((99, 3), dtype=np.float16)

    with pytest.raises(ValueError, match="PaliGemma/kernel"):
        guide_weight_loaders.strict_merge_pi05_base_params(loaded, _reference_params())


def test_similar_but_different_prefix_is_native_not_guide() -> None:
    reference = _reference_params()
    loaded = _loaded_native_params()
    reference["guide_encoder_extra"] = {
        "kernel": jax.ShapeDtypeStruct((2, 2), jnp.float32),
    }
    loaded["guide_encoder_extra"] = {
        "kernel": np.ones((2, 2), dtype=np.float16),
    }

    result = guide_weight_loaders.strict_merge_pi05_base_params(loaded, reference)

    assert isinstance(result["guide_encoder_extra"]["kernel"], np.ndarray)
    assert result["guide_encoder_extra"]["kernel"].dtype == np.dtype(np.float32)


def test_strict_merge_preserves_pytree_structure() -> None:
    result = guide_weight_loaders.strict_merge_pi05_base_params(
        _loaded_native_params(),
        _reference_params(),
    )

    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(_reference_params())


def test_loader_downloads_and_restores_then_delegates_to_strict_merge(monkeypatch) -> None:
    reference = _reference_params()
    loaded = _loaded_native_params()
    calls = {}
    restored_path = object()
    expected_result = {"sentinel": np.array(1, dtype=np.int32)}

    def fake_maybe_download(path):
        calls["download_path"] = path
        return restored_path

    def fake_restore_params(path, *, restore_type):
        calls["restore_path"] = path
        calls["restore_type"] = restore_type
        return loaded

    def fake_strict_merge(loaded_params, reference_params, *, guide_prefix):
        calls["loaded_params"] = loaded_params
        calls["reference_params"] = reference_params
        calls["guide_prefix"] = guide_prefix
        return expected_result

    monkeypatch.setattr(guide_weight_loaders.download, "maybe_download", fake_maybe_download)
    monkeypatch.setattr(_model, "restore_params", fake_restore_params)
    monkeypatch.setattr(
        guide_weight_loaders,
        "strict_merge_pi05_base_params",
        fake_strict_merge,
    )

    loader = guide_weight_loaders.GuidePi0BaseWeightLoader("synthetic/pi05_base")
    result = loader.load(reference)

    assert result is expected_result
    assert calls["download_path"] == "synthetic/pi05_base"
    assert calls["restore_path"] is restored_path
    assert calls["restore_type"] is np.ndarray
    assert calls["loaded_params"] is loaded
    assert calls["reference_params"] is reference
    assert calls["guide_prefix"] == _GUIDE_PREFIX
