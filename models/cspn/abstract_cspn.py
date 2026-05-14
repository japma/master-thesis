"""Abstract base class for conditional sum-product network models."""

from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractCSPN(nn.Module, ABC):
    """Common interface for conditional latent priors.

    Implementations may be custom CSPN variants or wrappers around external
    probabilistic circuit libraries such as SPFlow.
    """

    @abstractmethod
    def forward(self, z, labels):
        """Return log p(z | labels)."""
        raise NotImplementedError
