import torch
from torch import nn
import torch.nn.functional as F


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
        """
        :param left:         (N, num_input_nodes)  log-probs from left scopes
        :param right:        (N, num_input_nodes)  log-probs from right scopes
        :param cond_weights: (N, num_output_nodes, num_input_nodes, num_input_nodes) logits
        :return:             (N, num_output_nodes)  log-probs
        """
        N, O, I, _ = cond_weights.shape

        log_weights = F.log_softmax(cond_weights.view(N, O, I * I), dim=-1).view(
            N, O, I, I
        )  # (N, O, I, I)

        log_prod = left.unsqueeze(2) + right.unsqueeze(1)  # (N, I, I)

        output = torch.logsumexp(
            log_weights + log_prod.unsqueeze(1), dim=(-2, -1)
        )  # (N, O)
        return output
