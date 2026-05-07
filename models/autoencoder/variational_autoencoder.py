"""Variational autoencoder model."""

import logging

import torch
import torch.nn as nn

from .abstract_autoencoder import AbstractAutoencoder

logger = logging.getLogger(__name__)


def reparameterize(mu, logvar):
    """Sample latent using reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VariationalAutoencoder(AbstractAutoencoder):
    """VAE with Gaussian latent space and convolutional encoder/decoder."""

    def __init__(self, input_shape, latent_size, base_channels):
        super().__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size

        channels, height, width = self.input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            example = torch.zeros(1, channels, height, width)
            encoded_example = self.encoder(example)

        # Store as a single tuple rather than three separate attributes
        self.encoded_shape = tuple(encoded_example.shape[1:])  # (C, H, W)
        self.encoded_flat_dim = encoded_example.numel()

        logger.info(
            "Encoded shape: %s, flat dim: %d", self.encoded_shape, self.encoded_flat_dim
        )

        self.mu_head = nn.Linear(self.encoded_flat_dim, latent_size)
        self.logvar_head = nn.Linear(self.encoded_flat_dim, latent_size)

        self.decoder_input = nn.Linear(latent_size, self.encoded_flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.encoded_shape[0],
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(base_channels, channels, kernel_size=3, stride=1, padding=1),
            # No activation here — apply in the loss or normalise inputs to match.
            # Use sigmoid if inputs are in [0, 1] with BCE loss;
            # use tanh (or nothing) if inputs are normalised to [-1, 1] with MSE.
        )

    def encode_distribution(self, x):
        """Return mean and log-variance of q(z|x)."""
        features = self.encoder(x)
        flat_features = features.view(features.size(0), -1)
        mu = self.mu_head(flat_features)
        logvar = self.logvar_head(flat_features)
        return mu, logvar

    def encode(self, x):
        """Encode to a deterministic latent representation (returns µ)."""
        mu, _ = self.encode_distribution(x)
        return mu

    def decode(self, latent):
        """Decode a latent vector to an image."""
        decoded = self.decoder_input(latent)
        decoded = decoded.view(latent.size(0), *self.encoded_shape)
        return self.decoder(decoded)

    def forward(self, x):
        """Full forward pass. Returns (reconstruction, µ, log σ²)."""
        mu, logvar = self.encode_distribution(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
