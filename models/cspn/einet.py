from models.cspn.nn_for_einet import EinetConditioningNetwork
from models.cspn.einsum_layer import EinsumLayer
from models.cspn.gaussian_leaf_layer import GaussianLeafLayer
import torch
from torch import nn


class Einet(nn.Module):
    def __init__(self, num_vars, context_dim, num_leaves, num_nodes):
        super().__init__()
        self.leaf_layer = GaussianLeafLayer(num_scopes=num_vars, num_leaves=num_leaves)
        self.einsum_layer = EinsumLayer(
            num_input_nodes=num_leaves, num_output_nodes=num_nodes
        )
        self.cond_net = EinetConditioningNetwork(
            context_dim=context_dim,
            num_scopes=num_vars,
            num_leaves=num_leaves,
            num_nodes=num_leaves,  # input nodes to einsum = num_leaves
            num_output_nodes=num_nodes,
        )
        # root mixing weights (unconditional, or also conditioned if you want)
        self.root_weights = nn.Parameter(torch.ones(num_nodes) / num_nodes)

    def forward(self, x, context):
        # (N, num_vars, num_leaves)
        mu, logvar, weights = self.cond_net(context)
        log_leaves = self.leaf_layer(x, mu, logvar)

        # split scopes into left / right halves -> (N, num_leaves)
        half = x.shape[1] // 2
        left = log_leaves[:, :half, :].sum(dim=1)  # (N, num_leaves)
        right = log_leaves[:, half:, :].sum(dim=1)  # (N, num_leaves)

        # einsum layer -> (N, num_nodes)
        # weights: (N, num_nodes, num_leaves, num_leaves)
        h = self.einsum_layer(left, right, weights)  # "ni,nj,noij->no"

        # root: mix over nodes -> scalar log-prob per sample
        log_p = torch.logsumexp(h + self.root_weights.log(), dim=-1)  # (N,)
        return log_p
