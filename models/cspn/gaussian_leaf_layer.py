import torch
from torch import nn


class GaussianLeafLayer(nn.Module):
    def __init__(self, num_scopes: int, num_leaves: int):
        super().__init__()
        self.num_scopes = num_scopes
        self.num_leaves = num_leaves

    def forward(
        self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        var = torch.exp(logvar)
        log_density = -0.5 * torch.log(2 * torch.pi * var) - 0.5 * (
            (x.unsqueeze(-1) - mu) ** 2 / var
        )
        return log_density
