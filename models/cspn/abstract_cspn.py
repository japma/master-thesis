"""Abstract base class for conditional sum-product network models."""

from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractCSPN(nn.Module, ABC):
    """Common interface for conditional latent priors.

    Implementations may be custom CSPN variants or wrappers around external
    probabilistic circuit libraries such as SPFlow.
    """

    @property
    def backend_name(self):
        """Return backend identifier used by this implementation."""
        return "custom"

    @abstractmethod
    def forward(self, z, labels):
        """Return log p(z | labels)."""
        raise NotImplementedError

    def log_joint(self, z, labels):
        """Optional: return log p(z, labels) when available."""
        raise NotImplementedError("Joint scoring is not implemented for this CSPN.")

    @abstractmethod
    def predict_latent(self, labels):
        """Return a label-conditioned latent prototype."""
        raise NotImplementedError

    @abstractmethod
    def transform_latent(self, z, source_labels, target_labels, strength=1.0):
        """Transform latent vectors from source labels to target labels."""
        raise NotImplementedError

    def sample(self, labels, num_samples=1):
        """Sampling is optional for the minimal interface."""
        raise NotImplementedError("Sampling is not implemented for this CSPN.")