from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractNeuralBaseline(nn.Module, ABC):
    @abstractmethod
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass
