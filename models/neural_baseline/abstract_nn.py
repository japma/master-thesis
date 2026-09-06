from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractNeuralBaseline(nn.Module, ABC):
    @abstractmethod
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        """Signature matches AbstractCSPN.sample so the generation probes can take a
        baseline wherever they take a circuit."""
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass
