import torch
import torch.nn as nn

from warnings import deprecated
from .abstract_autoencoder import AbstractAutoencoder


@deprecated("Not used anymore, use VAE instead")
class SimpleAutoencoder(AbstractAutoencoder):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.input_size = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def get_config(self):
        return {
            "model_type": "simple",
            "input_size": self.input_size,
            "latent_size": self.latent_dim,
        }
