"""Variational autoencoder model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_autoencoder import AbstractAutoencoder


class VariationalAutoencoder(AbstractAutoencoder):
    """VAE with Gaussian latent space and convolutional encoder/decoder."""

    def __init__(self, input_size, latent_size=32, image_shape=None, base_channels=32):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.image_shape = self._resolve_image_shape(input_size, image_shape)
        channels, height, width = self.image_shape

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

        self.encoded_channels = encoded_example.shape[1]
        self.encoded_height = encoded_example.shape[2]
        self.encoded_width = encoded_example.shape[3]
        self.encoded_flat_dim = (
            self.encoded_channels * self.encoded_height * self.encoded_width
        )

        self.mu_head = nn.Linear(self.encoded_flat_dim, latent_size)
        self.logvar_head = nn.Linear(self.encoded_flat_dim, latent_size)

        self.decoder_input = nn.Linear(latent_size, self.encoded_flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.encoded_channels,
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
        )

    @staticmethod
    def _resolve_image_shape(input_size, image_shape):
        if image_shape is not None:
            if len(image_shape) != 3:
                raise ValueError(
                    "image_shape must be a (channels, height, width) tuple"
                )
            channels, height, width = image_shape
            if channels * height * width != input_size:
                raise ValueError(
                    "image_shape product must match input_size "
                    f"({channels}*{height}*{width} != {input_size})"
                )
            return image_shape

        known_shapes = {
            784: (1, 28, 28),
            3072: (3, 32, 32),
            12288: (3, 64, 64),
        }
        if input_size in known_shapes:
            return known_shapes[input_size]

        raise ValueError(
            "Unable to infer image shape from input_size. "
            "Pass image_shape=(channels, height, width)."
        )

    def encode_distribution(self, x):
        """Return mean and log-variance of q(z|x)."""
        features = self.encoder(x)
        flat_features = features.view(features.size(0), -1)
        mu = self.mu_head(flat_features)
        logvar = self.logvar_head(flat_features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample latent using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode to a sampled latent representation."""
        mu, logvar = self.encode_distribution(x)
        return self.reparameterize(mu, logvar)

    def decode(self, latent):
        decoded = self.decoder_input(latent)
        decoded = decoded.view(
            latent.size(0),
            self.encoded_channels,
            self.encoded_height,
            self.encoded_width,
        )
        decoded = self.decoder(decoded)

        return torch.sigmoid(decoded)

    @staticmethod
    def kl_divergence(mu, logvar, reduction="mean"):
        """Compute KL(q(z|x) || p(z))."""
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if reduction == "none":
            return kl_per_sample
        if reduction == "sum":
            return kl_per_sample.sum()
        if reduction == "mean":
            return kl_per_sample.mean()

        raise ValueError("reduction must be one of: 'none', 'sum', 'mean'")
