import torch
from diffusers import AutoencoderTiny

from models.autoencoder import AbstractAutoencoder
from utils.config import PretrainedAutoencoderConfig


class TinyAutoencoderWrapper(AbstractAutoencoder):
    def __init__(self, config: PretrainedAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.vae = AutoencoderTiny.from_pretrained(self.config.name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(x)
            return latents.latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.vae.decode(z)
            return recon.sample

    def get_config(self) -> dict:
        return self.config.model_dump()

    def get_latent_dim(self) -> int:
        return self.config.latent_dim
