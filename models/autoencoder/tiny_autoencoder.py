from abc import ABC

import torch
from diffusers import AutoencoderTiny

from models.autoencoder import AbstractAutoencoder


class TinyAutoencoderWrapper(AbstractAutoencoder, ABC):
    def __init__(self, name: str = "madebyollin/taesd"):
        super().__init__()
        self.vae = AutoencoderTiny.from_pretrained(name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(x)
            return latents.latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.vae.decode(z)
            return recon.sample
