"""Factory for creating autoencoder models."""

from enum import Enum

from models.autoencoder import TinyAutoencoderWrapper


class AutoencoderType(Enum):
    SIMPLE = "simple"
    VARIATIONAL = "variational"
    PRETRAINED = "pretrained"

    def __str__(self) -> str:
        return self.value


def load_pretrained_autoencoder(name: str) -> TinyAutoencoderWrapper:
    if name == "madebyollin/taesd":
        return TinyAutoencoderWrapper(name="madebyollin/taesd")
    else:
        raise ValueError
