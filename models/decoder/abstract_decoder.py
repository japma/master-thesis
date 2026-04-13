"""Abstract decoder base module."""

from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractDecoder(nn.Module, ABC):
    def forward(self, latent):
        return self.decode(latent)

    @abstractmethod
    def decode(self, latent):
        pass
