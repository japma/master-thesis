from abc import ABC

import torch
from diffusers import AutoencoderTiny

from models.autoencoder import AbstractAutoencoder


class CelebAAutoencoderWrapper(AbstractAutoencoder, ABC):
    def __init__(self, name: str = "madebyollin/taesd"):
        super().__init__()
        self.name = name
        self.vae = AutoencoderTiny.from_pretrained(name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def get_config(self) -> dict:
        return {
            "model_type": "tiny",
            "name": self.name,
        }

    def get_latent_dim(self) -> int:
        # TODO place holder until it is resolved if this is needed
        return 32
