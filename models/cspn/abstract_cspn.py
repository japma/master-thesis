"""Abstract base class for conditional sum-product network models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractCSPN(nn.Module, ABC):
    """Common interface for conditional latent priors.

    Implementations may be custom CSPN variant or wrappers around external
    probabilistic circuit libraries such as SPFlow.
    """

    # @abstractmethod
    # @property
    # def num_classes(self) -> int:
    #    """Number of classes for which the model can condition."""
    #    raise NotImplementedError

    @abstractmethod
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return log p(z | labels).

        Args:
            z: Latent vectors (N, latent_dim)
            labels: Class labels (N,) - one-hot encoding is handled internally

        Returns:
            Log probabilities (N,)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """Return sampled latent vector.

        Args:
            labels: Class labels (N,) - one-hot encoding is handled internally

        Returns:
            Sampled latent vectors (N, latent_dim)
        """
        raise NotImplementedError
