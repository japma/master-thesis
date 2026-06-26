from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import AutoencoderConfig

from .abstract_autoencoder import AbstractAutoencoder, AutoencoderForwardOutput


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VariationalAutoencoder(AbstractAutoencoder):
    def __init__(
        self,
        config: AutoencoderConfig,
    ):
        super().__init__()
        self.config = config
        print(type(self.config))

        # --- Encoder ---
        encoder_blocks = []
        current_channels = 3  # for rgb
        for _ in range(config.num_blocks):
            h_dim = current_channels * 2
            encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels, h_dim, kernel_size=3, padding=1, stride=2
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            current_channels = h_dim

        self.encoder = nn.Sequential(*encoder_blocks)

        self.encoded_size = config.image_size // (2**config.num_blocks)
        flat_dim = current_channels * self.encoded_size * self.encoded_size

        self.mu_head = nn.Linear(flat_dim, config.latent_dim)
        self.log_var_head = nn.Linear(flat_dim, config.latent_dim)

        self.bottleneck_channels = current_channels

        # --- Decoder ---
        self.fc_decode = nn.Linear(config.latent_dim, flat_dim)

        decoder_blocks = []
        for _ in range(config.num_blocks - 1):
            h_dim = current_channels // 2
            decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_channels,
                        h_dim,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                )
            )
            current_channels = h_dim

        decoder_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    current_channels,
                    3,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    output_padding=1,
                )
            )
        )

        self.decoder = nn.Sequential(*decoder_blocks)

    def _encode_distribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        return mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self._encode_distribution(x)
        return mu

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.bottleneck_channels, self.encoded_size, self.encoded_size)
        return self.decoder(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)

    def forward(self, x: torch.Tensor) -> AutoencoderForwardOutput:
        mu, log_var = self._encode_distribution(x)
        z = reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def get_config(self) -> dict:
        cfg = self.config.model_dump()
        return cfg

    def get_latent_dim(self) -> int:
        return self.config.latent_dim
