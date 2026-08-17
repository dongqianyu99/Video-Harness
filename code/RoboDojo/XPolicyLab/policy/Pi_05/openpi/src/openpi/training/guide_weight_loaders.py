from __future__ import annotations

import dataclasses

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download


def _is_guide_path(
    path: tuple[str, ...],
    guide_prefix: tuple[str, ...],
) -> bool:
    """Return whether a leaf path belongs to the guide subtree."""

    return len(path) >= len(guide_prefix) and path[: len(guide_prefix)] == guide_prefix


def _format_path(path: tuple[str, ...]) -> str:
    """Format a tuple leaf path for readable error messages."""

    return "/".join(path) if path else "<root>"


def _flatten_params(
    params: at.Params,
) -> dict[tuple[str, ...], object]:
    """Flatten a nested parameter tree while preserving tuple paths."""

    return flax.traverse_util.flatten_dict(params)


def _classify_reference_keys(
    flat_reference: dict[tuple[str, ...], object],
    *,
    guide_prefix: tuple[str, ...],
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Split reference leaves into guide and native key sets."""

    guide_keys = {path for path in flat_reference if _is_guide_path(path, guide_prefix)}
    native_keys = set(flat_reference) - guide_keys
    return guide_keys, native_keys


def _format_paths(paths: set[tuple[str, ...]]) -> str:
    """Format multiple parameter paths for an error message."""

    return ", ".join(_format_path(path) for path in sorted(paths))


def _validate_key_sets(
    flat_loaded: dict[tuple[str, ...], object],
    *,
    guide_keys: set[tuple[str, ...]],
    native_keys: set[tuple[str, ...]],
    guide_prefix: tuple[str, ...],
) -> None:
    """Require the loaded checkpoint keys to match native keys exactly."""

    if not guide_keys:
        raise ValueError(
            f"Reference parameter tree contains no guide parameters under prefix {_format_path(guide_prefix)}"
        )

    loaded_keys = set(flat_loaded)
    missing_keys = native_keys - loaded_keys
    extra_keys = loaded_keys - native_keys

    if missing_keys or extra_keys:
        problems = []

        if missing_keys:
            problems.append(f"missing keys: {_format_paths(missing_keys)}")

        if extra_keys:
            problems.append(f"extra keys: {_format_paths(extra_keys)}")

        raise ValueError(
            "Base checkpoint keys do not exactly match native reference keys; "
            + "; ".join(problems)
        )


def _merge_flat_leaves(
    flat_loaded: dict[tuple[str, ...], object],
    flat_reference: dict[tuple[str, ...], object],
    *,
    guide_keys: set[tuple[str, ...]],
    native_keys: set[tuple[str, ...]],
) -> dict[tuple[str, ...], object]:
    """Merge native checkpoint leaves with reference guide leaves."""

    result = {}

    for path in sorted(native_keys):
        loaded_leaf = flat_loaded[path]
        reference_leaf = flat_reference[path]

        if loaded_leaf.shape != reference_leaf.shape:
            raise ValueError(
                f"Shape mismatch for {_format_path(path)}: "
                f"loaded shape {loaded_leaf.shape}, "
                f"reference shape {reference_leaf.shape}"
            )

        try:
            result[path] = np.asarray(
                loaded_leaf,
                dtype=reference_leaf.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot cast {_format_path(path)} to reference dtype {reference_leaf.dtype}") from exc

    for path in sorted(guide_keys):
        result[path] = flat_reference[path]

    return result


def strict_merge_pi05_base_params(
    loaded_params: at.Params,
    reference_params: at.Params,
    *,
    guide_prefix: tuple[str, ...] = ("guide_encoder",),
) -> at.Params:
    """Strictly merge a native pi05 base checkpoint into a GuidePi0 tree."""

    flat_loaded = _flatten_params(loaded_params)
    flat_reference = _flatten_params(reference_params)

    guide_keys, native_keys = _classify_reference_keys(
        flat_reference,
        guide_prefix=guide_prefix,
    )

    _validate_key_sets(
        flat_loaded,
        guide_keys=guide_keys,
        native_keys=native_keys,
        guide_prefix=guide_prefix,
    )

    flat_result = _merge_flat_leaves(
        flat_loaded,
        flat_reference,
        guide_keys=guide_keys,
        native_keys=native_keys,
    )

    return flax.traverse_util.unflatten_dict(flat_result)


@dataclasses.dataclass(frozen=True)
class GuidePi0BaseWeightLoader:
    """Load an unmodified pi05_base checkpoint for GuidePi0 initialization."""

    params_path: str
    guide_prefix: tuple[str, ...] = ("guide_encoder",)

    def load(self, params: at.Params) -> at.Params:
        checkpoint_path = download.maybe_download(self.params_path)

        loaded_params = _model.restore_params(
            checkpoint_path,
            restore_type=np.ndarray,
        )

        return strict_merge_pi05_base_params(
            loaded_params,
            params,
            guide_prefix=self.guide_prefix,
        )
