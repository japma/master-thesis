"""Abstract autoencoder base module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AutoencoderForwardOutput:
    reconstructed: torch.Tensor
    latent: torch.Tensor


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
        return AutoencoderForwardOutput(
            reconstructed=reconstructed,
            latent=latent,
        )

    def get_config(self) -> dict:
        return {}

    @abstractmethod
    def get_latent_dim(self) -> torch.Size:
        pass
