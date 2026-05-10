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

    @abstractmethod
    def predict_latent(self, labels):
        """Return a label-conditioned latent prototype."""
        raise NotImplementedError

    @abstractmethod
    def transform_latent(self, z, source_labels, target_labels, strength=1.0):
        """Transform latent vectors from source labels to target labels."""
        raise NotImplementedError
