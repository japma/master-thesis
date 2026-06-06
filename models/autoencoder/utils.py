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


def load_pretrained_autoencoder(model: str, type: AutoencoderType) -> AbstractAutoencoder:
    if type == AutoencoderType.SIMPLE:
        pass
    elif type == AutoencoderType.VARIATIONAL:
        pass
    elif type == AutoencoderType.TINY:
        return TinyAutoencoderWrapper(model)
    else:
        raise ValueError(f"Unsupported autoencoder type: {type}")


def create_autoencoder() -> AbstractAutoencoder:
    pass
