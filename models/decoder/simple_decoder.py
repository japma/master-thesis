import torch.nn as nn

from .abstract_decoder import AbstractDecoder


class SimpleDecoder(AbstractDecoder):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.latent_size = latent_size
        self.output_size = output_size

        self.network = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid(),  # Output normalized to [0, 1]
        )

    def decode(self, latent):
        reconstructed = self.network(latent)
        return reconstructed
