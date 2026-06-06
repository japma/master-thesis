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


def load_pretrained_model(model_path: str, device: torch.device) -> AbstractAutoencoder:
    pass


def create_autoencoder() -> AbstractAutoencoder:
    pass
