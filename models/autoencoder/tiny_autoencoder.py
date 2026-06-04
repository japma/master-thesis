from abc import ABC

import torch
from diffusers import AutoencoderTiny

from models.autoencoder import AbstractAutoencoder


class TinyAutoencoderWrapper(AbstractAutoencoder, ABC):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(x)
            return latents.latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.decode(z)
