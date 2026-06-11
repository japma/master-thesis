"""Abstract autoencoder base module."""

from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
import torch.nn as nn


AutoencoderForwardOutput: TypeAlias = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


class AbstractAutoencoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> AutoencoderForwardOutput:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @abstractmethod
    def get_latent_dim(self) -> int:
        pass
