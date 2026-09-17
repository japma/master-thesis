"""The interfaces the two-stage evaluation pipeline consumes."""

from typing import Protocol

from torch import Tensor


class LatentCodec(Protocol):
    def encode(self, x: Tensor) -> Tensor:
        """`(B, C, H, W)` images in [0, 1] to `(B, D)` posterior-mean latents."""
        ...

    def decode(self, z: Tensor) -> Tensor:
        """`(B, D)` latents to `(B, C, H, W)` images in [0, 1]."""
        ...


class ConditionalSampler(Protocol):
    def sample(self, y: Tensor, n_per_label: int) -> Tensor:
        """`(L, 3)` labels to `(L, n_per_label, D)` latents; `[i]` belongs to `y[i]`."""
        ...
