from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.guide_pi0 import GuidePi0


@dataclasses.dataclass(frozen=True)
class GuidePi0Config(pi0_config.Pi0Config):
    """Configuration for the Guide-conditioned Pi05 model structure."""

    pi05: bool = True

    guide_num_queries: int = 8
    guide_resampler_width: int = 1024
    guide_resampler_num_heads: int = 8
    guide_resampler_ffn_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.pi05:
            raise ValueError("GuidePi0Config only supports pi05=True")

        if self.guide_num_queries <= 0:
            raise ValueError("guide_num_queries must be positive")

        if self.guide_resampler_width <= 0:
            raise ValueError("guide_resampler_width must be positive")

        if self.guide_resampler_num_heads <= 0:
            raise ValueError("guide_resampler_num_heads must be positive")

        if self.guide_resampler_width % self.guide_resampler_num_heads != 0:
            raise ValueError("guide_resampler_width must be divisible by guide_resampler_num_heads")

        if self.guide_resampler_ffn_hidden_dim is not None and self.guide_resampler_ffn_hidden_dim <= 0:
            raise ValueError("guide_resampler_ffn_hidden_dim must be positive")

    @override
    def create(self, rng: at.KeyArrayLike) -> GuidePi0:
        from openpi.models.guide_pi0 import GuidePi0  # noqa: PLC0415

        return GuidePi0(self, rngs=nnx.Rngs(rng))
