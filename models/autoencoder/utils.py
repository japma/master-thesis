"""Factory for creating autoencoder models."""

from enum import Enum


class AutoencoderType(Enum):
    SIMPLE = "simple"
    VARIATIONAL = "variational"
    PRETRAINED = "pretrained"

    def __str__(self) -> str:
        return self.value
