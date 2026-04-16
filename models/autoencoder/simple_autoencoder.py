"""Autoencoder model for unsupervised learning."""

from .abstract_autoencoder import AbstractAutoencoder
from ..encoder import SimpleEncoder
from ..decoder import SimpleDecoder


class SimpleAutoencoder(AbstractAutoencoder):
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

    def decode(self, latent):
        return self.decoder(latent)
