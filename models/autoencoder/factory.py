"""Factory for creating autoencoder models."""

from enum import Enum
import torch
from .abstract_autoencoder import AbstractAutoencoder
from .simple_autoencoder import SimpleAutoencoder
from .variational_autoencoder import VariationalAutoencoder
from .tiny_autoencoder import TinyAutoencoderWrapper


class AutoencoderType(Enum):
    """Available autoencoder model types."""

    SIMPLE = "simple"
    VARIATIONAL = "variational"
    TINY = "tiny"

    def __str__(self) -> str:
        return self.value


def create_autoencoder(
    model_type: AutoencoderType,
    input_shape: tuple[int, int, int],
    latent_size: int,
    **kwargs,
) -> AbstractAutoencoder:

    if model_type == AutoencoderType.SIMPLE:
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        return SimpleAutoencoder(
            input_size=input_size,
            latent_size=latent_size,
        )

    elif model_type == AutoencoderType.VARIATIONAL:
        return VariationalAutoencoder(
            input_shape=input_shape,
            latent_size=latent_size,
            base_channels=kwargs.get("base_channels", 32),
            num_blocks=kwargs.get("num_blocks", 2),
            res_blocks=kwargs.get("res_blocks", 1),
        )

    elif model_type == AutoencoderType.TINY:
        return TinyAutoencoderWrapper()

    else:
        raise ValueError(f"Unknown autoencoder type: {model_type}")
