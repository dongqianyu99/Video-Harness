import jax.numpy as jnp
import numpy as np

from openpi.models import pi0
from openpi.models.guide_attention import make_gca_ar_mask
from openpi.models.guide_attention import make_gca_attn_mask


def test_gca_ar_mask_has_guide_control_action_block_boundaries() -> None:
    actual = make_gca_ar_mask(guide_tokens=3, control_tokens=2, action_tokens=3)
    expected = jnp.array([False, False, False, True, False, True, False, False])

    np.testing.assert_array_equal(actual, expected)


def test_gca_attention_mask_has_expected_visibility_and_padding() -> None:
    guide_mask = jnp.array([[True, True, False]])
    control_mask = jnp.array([[True, False]])
    action_mask = jnp.array([[True, True, False]])

    actual = make_gca_attn_mask(guide_mask, control_mask, action_mask)
    expected = jnp.array(
        [
            [True, True, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, False, True, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, False, True, False, True, True, False],
            [True, True, False, True, False, True, True, False],
            [False, False, False, False, False, False, False, False],
        ]
    )[None, ...]

    np.testing.assert_array_equal(actual, expected)

    padding_indices = (2, 4, 7)
    for index in padding_indices:
        assert not np.any(actual[:, index, :])
        assert not np.any(actual[:, :, index])


def test_all_guide_masked_reduces_to_stock_control_action_mask() -> None:
    guide_mask = jnp.zeros((1, 3), dtype=jnp.bool_)
    control_mask = jnp.array([[True, False]])
    action_mask = jnp.array([[True, True, False]])

    actual = make_gca_attn_mask(guide_mask, control_mask, action_mask)
    actual_control_action = actual[:, 3:, 3:]

    stock_input_mask = jnp.concatenate([control_mask, action_mask], axis=1)
    stock_ar_mask = make_gca_ar_mask(guide_tokens=0, control_tokens=2, action_tokens=3)
    expected_control_action = pi0.make_attn_mask(stock_input_mask, stock_ar_mask)

    np.testing.assert_array_equal(actual_control_action, expected_control_action)
