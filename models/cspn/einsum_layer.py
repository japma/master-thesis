import torch
from torch import nn


class EinsumLayer(nn.Module):
    def __init__(
        self,
        num_input_nodes: int,
        num_output_nodes: int,
    ):
        super().__init__()
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, cond_weights: torch.Tensor
    ) -> torch.Tensor:
        output = torch.einsum("ni,nj,noij->no", left, right, cond_weights)
        return output
