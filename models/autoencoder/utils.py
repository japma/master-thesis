"""Factory for creating autoencoder models."""

from enum import Enum
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


def load_autencoder(
    model_type: AutoencoderType,
    path: str | None = None,
    name: str = "madebyollin/taesd",
) -> AbstractAutoencoder:
    if model_type == AutoencoderType.SIMPLE:
        return SimpleAutoencoder()
    elif model_type == AutoencoderType.VARIATIONAL:
        return VariationalAutoencoder()
    elif model_type == AutoencoderType.TINY:
        return TinyAutoencoderWrapper(name=name)
    else:
        raise ValueError(f"Unsupported autoencoder type: {model_type}")
