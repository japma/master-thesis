import torch
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from models.autoencoder import AbstractAutoencoder


class PretrainedVAE(AbstractAutoencoder):
    def __init__(self, name: str, height: int, width: int) -> None:
        super().__init__()
        ckpt = AutoencoderKL.from_pretrained(name)
        if ckpt is None:
            raise ValueError("No checkpoint file found")
        self.vae: AutoencoderKL = ckpt
        self.vae.eval()

        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(1, 3, height, width)
            latent: AutoencoderKLOutput = self.vae.encode(dummy)
        self.latent_dim: torch.Size = latent.latent_dist.sample().shape[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out: AutoencoderKLOutput = self.vae.encode(x)
        rg = out.latent_dist.sample().flatten(start_dim=1)
        return rg

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        unflattened_z = z.unflatten(1, self.latent_dim)
        rg = self.vae.decode(unflattened_z).sample
        return rg

    def get_latent_dim(self) -> torch.Size:
        return self.latent_dim
