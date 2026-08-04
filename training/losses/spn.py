import torch
from torch import nn

from training.losses.base import LossOutput


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor) -> LossOutput:
        nll = -outputs.mean()
        return LossOutput(total=nll)
