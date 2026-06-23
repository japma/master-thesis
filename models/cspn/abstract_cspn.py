from abc import ABC, abstractmethod
from enum import StrEnum

import torch
import torch.nn as nn


class CSPNType(StrEnum):
    PSINET = "psinet"
    SPFLOW = "spflow"
    CUSTOM = "custom"
    PSINET_DEPRECATED = "PsiNetCSPN"
    CUSTOM_DEPRECATED = "custom_cspn"


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
