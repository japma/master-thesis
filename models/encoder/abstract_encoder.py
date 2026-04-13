"""Abstract encoder base module."""

from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractEncoder(nn.Module, ABC):
    def forward(self, x):
        return self.encode(x)

    @abstractmethod
    def encode(self, x):
        pass
