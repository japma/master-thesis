"""Abstract autoencoder base module."""

from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractAutoencoder(nn.Module, ABC):
    """Abstract base class for autoencoder models.

    An autoencoder learns to encode input data into a latent representation
    and decode it back to reconstruct the original input.
    """

    @abstractmethod
    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: Input tensor

        Returns:
            Latent representation tensor
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent):
        """Decode latent representation to reconstruct input.

        Args:
            latent: Latent representation tensor

        Returns:
            Reconstructed tensor
        """
        raise NotImplementedError

    def forward(self, x):
        """Forward pass: encode and reconstruct.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

    def get_latent(self, x):
        """Get latent representation for input.

        Args:
            x: Input tensor

        Returns:
            Latent representation tensor
        """
        return self.encode(x)

    def reconstruct(self, x):
        """Reconstruct input from itself.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor
        """
        return self.forward(x)
