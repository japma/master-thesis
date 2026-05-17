from typing import List, Tuple
import torch
from torch import nn


class EinetConditioningNetwork(nn.Module):
    def __init__(
        self,
        context_dim: int,
        num_scopes: int,
        num_leaves: int,
        layer_nodes: List[int],
    ):
        """
        :param context_dim: Dimensionality of the conditioning context.
        :param num_scopes:  Number of leaf scopes (= num_vars).
        :param num_leaves:  Number of leaf components per scope.
        :param layer_nodes: List of output node counts per EinsumLayer, bottom to top.
                            e.g. [8, 4, 2, 2, 2] for depth=5, num_nodes=8.
        """
        super().__init__()
        self.num_scopes = num_scopes
        self.num_leaves = num_leaves
        self.layer_nodes = layer_nodes
        self.depth = len(layer_nodes)

        # Leaf parameters: mu + logvar for each scope and leaf component
        leaf_params = num_scopes * num_leaves * 2

        # Weight parameters per layer:
        # At layer l there are 2^(depth-1-l) region pairs (num_pairs_l).
        # Each pair needs out_nodes * in_nodes * in_nodes weights.
        # in_nodes for layer 0 = num_leaves, else layer_nodes[l-1].
        self.layer_shapes: List[Tuple[int, int, int]] = []  # (num_pairs, out, in)
        weight_params = 0
        for l in range(self.depth):
            num_pairs = 2 ** (self.depth - 1 - l)
            out_nodes = layer_nodes[l]
            in_nodes = num_leaves if l == 0 else layer_nodes[l - 1]
            self.layer_shapes.append((num_pairs, out_nodes, in_nodes))
            weight_params += num_pairs * out_nodes * in_nodes * in_nodes

        total_params = leaf_params + weight_params
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, total_params),
        )

    def forward(self, context: torch.Tensor):
        """
        :param context: (N, context_dim)
        :return:
            mu:          (N, num_scopes, num_leaves)
            logvar:      (N, num_scopes, num_leaves)
            all_weights: list of (N, num_pairs, out_nodes, in_nodes, in_nodes),
                         one tensor per EinsumLayer
        """
        context = context.float()
        out = self.mlp(context)
        N = out.shape[0]
        S, K = self.num_scopes, self.num_leaves

        # Slice off leaf parameters
        mu = out[:, : S * K].view(N, S, K)
        logvar = out[:, S * K : 2 * S * K].view(N, S, K)

        # Slice off per-layer weights
        cursor = 2 * S * K
        all_weights = []
        for num_pairs, out_nodes, in_nodes in self.layer_shapes:
            size = num_pairs * out_nodes * in_nodes * in_nodes
            w = out[:, cursor : cursor + size].view(
                N, num_pairs, out_nodes, in_nodes, in_nodes
            )
            all_weights.append(w)
            cursor += size

        return mu, logvar, all_weights
