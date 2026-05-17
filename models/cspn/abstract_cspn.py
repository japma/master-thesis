"""Abstract base class for conditional sum-product network models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractCSPN(nn.Module, ABC):
    """Common interface for conditional latent priors.

    Implementations may be custom CSPN variants or wrappers around external
    probabilistic circuit libraries such as SPFlow.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return log p(z | context)."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """Return sampled latent vector."""
        raise NotImplementedError
