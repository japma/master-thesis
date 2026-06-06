"""Factory for creating autoencoder models."""

from enum import Enum

import diffusers
import torch
from .abstract_autoencoder import AbstractAutoencoder
from .simple_autoencoder import SimpleAutoencoder
from .variational_autoencoder import VariationalAutoencoder
from .tiny_autoencoder import TinyAutoencoderWrapper


class AutoencoderType(Enum):
    SIMPLE = "simple"
    VARIATIONAL = "variational"
    TINY = "tiny"

    def __str__(self) -> str:
        return self.value


# TODO move into utils directory
def load_pretrained_autoencoder(
    model: str, ae_type: AutoencoderType
) -> AbstractAutoencoder:
    if ae_type == AutoencoderType.SIMPLE or ae_type == AutoencoderType.VARIATIONAL:
        ckpt = torch.load(model, map_location=torch.device("cpu"), weights_only=True)
        cfg = ckpt["model_cfg"]
        ae = create_autoencoder(cfg)
        ae.load_state_dict(ckpt["model_state"])
        return ae
    elif ae_type == AutoencoderType.TINY:
        return TinyAutoencoderWrapper(model)
    else:
        raise ValueError(f"Unsupported autoencoder type: {type}")


def create_autoencoder(cfg: dict) -> AbstractAutoencoder:
    if cfg["model_type"] == "tiny":
        raise ValueError(
            "Tiny autoencoder does not require configuration. Please use TinyAutoencoderWrapper directly."
        )
    elif cfg["model_type"] == "simple":
        return SimpleAutoencoder(
            input_size=cfg["input_size"],
            latent_size=cfg["latent_size"],
        )
    elif cfg["model_type"] == "variational":
        return VariationalAutoencoder(
            input_shape=cfg["input_shape"],
            latent_size=cfg["latent_size"],
            base_channels=cfg["base_channels"],
            num_blocks=cfg["num_blocks"],
            res_blocks=cfg["res_blocks"],
        )
    else:
        raise ValueError(f"Unsupported autoencoder type: {cfg['model_type']}")
