from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractCSPN(nn.Module, ABC):
    @abstractmethod
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass
