"""Autoencoder model for unsupervised learning."""

import torch.nn as nn
from models.encoder import SimpleEncoder
from models.decoder import SimpleDecoder


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size=32,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = SimpleEncoder(input_size, latent_size)
        self.decoder = SimpleDecoder(latent_size, input_size)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)
