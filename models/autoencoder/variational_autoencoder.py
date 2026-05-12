"""Variational autoencoder model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_autoencoder import AbstractAutoencoder


def reparameterize(mu, logvar):
    """Sample latent using reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and a skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class VariationalAutoencoder(AbstractAutoencoder):
    """
    VAE with Gaussian latent space, convolutional encoder/decoder,
    residual blocks, and BatchNorm throughout.

    Args:
        input_shape:   (C, H, W) of the input images.
        latent_size:   Dimensionality of the latent space z.
        base_channels: Channel width of the first conv layer; doubles each stride-2 block.
        num_blocks:    Number of stride-2 downsampling blocks (encoder depth).
        res_blocks:    Number of residual blocks inserted between each stride-2 block.
    """

    def __init__(
        self,
        input_shape,
        latent_size,
        base_channels=32,
        num_blocks=2,
        res_blocks=1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size

        channels, height, width = input_shape

        enc_layers = []
        ch_in = channels
        for i in range(num_blocks):
            ch_out = base_channels * (2**i)
            enc_layers += [
                nn.Conv2d(
                    ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            for _ in range(res_blocks):
                enc_layers.append(ResBlock(ch_out))
            ch_in = ch_out

        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            example = torch.zeros(1, channels, height, width)
            encoded_example = self.encoder(example)

        self.encoded_shape = tuple(encoded_example.shape[1:])
        self.encoded_flat_dim = encoded_example.numel()

        self.mu_head = nn.Linear(self.encoded_flat_dim, latent_size)
        self.logvar_head = nn.Linear(self.encoded_flat_dim, latent_size)

        self.decoder_input = nn.Linear(latent_size, self.encoded_flat_dim)

        dec_layers = []
        for i in range(num_blocks - 1, -1, -1):
            ch_in = base_channels * (2**i)
            ch_out = base_channels * (2 ** (i - 1)) if i > 0 else base_channels

            for _ in range(res_blocks):
                dec_layers.append(ResBlock(ch_in))

            dec_layers += [
                nn.ConvTranspose2d(
                    ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]

        dec_layers.append(
            nn.Conv2d(base_channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.Sequential(*dec_layers)

    def encode_distribution(self, x):
        """Return mean and log-variance of q(z|x)."""
        features = self.encoder(x)
        flat = features.view(features.size(0), -1)
        mu = self.mu_head(flat)
        logvar = self.logvar_head(flat).clamp(-10, 10)  # prevent KL overflow
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
