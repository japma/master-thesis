
import torch
from torch import nn


class EinetConditioningNetwork(nn.Module):
    def __init__(
        self,
        context_dim: int,
        num_scopes: int,
        num_leaves: int,
        layer_nodes: list[int],
        nn_hidden_dim: int = 256,
        nn_num_hidden_layers: int = 2,
    ):
        """
        :param context_dim: Dimensionality of the conditioning context.
        :param num_scopes:  Number of leaf scopes (= num_vars).
        :param num_leaves:  Number of leaf components per scope.
        :param layer_nodes: List of output node counts per EinsumLayer, bottom to top.
                            e.g. [8, 4, 2, 2, 2] for depth=5, num_nodes=8.
        :param nn_hidden_dim: Hidden layer dimension (default 256).
        :param nn_num_hidden_layers: Number of hidden layers (default 2).
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
        self.layer_shapes: list[tuple[int, int, int]] = []  # (num_pairs, out, in)
        weight_params = 0
        for l in range(self.depth):
            num_pairs = 2 ** (self.depth - 1 - l)
            out_nodes = layer_nodes[l]
            in_nodes = num_leaves if l == 0 else layer_nodes[l - 1]
            self.layer_shapes.append((num_pairs, out_nodes, in_nodes))
            weight_params += num_pairs * out_nodes * in_nodes * in_nodes

        total_params = leaf_params + weight_params

        mlp_layers = nn.ModuleList()
        mlp_layers.append(nn.Linear(context_dim, nn_hidden_dim))
        for _ in range(nn_num_hidden_layers):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(nn_hidden_dim, nn_hidden_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(nn_hidden_dim, total_params))
        self.mlp = nn.Sequential(*mlp_layers)

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
        # Clamp logvar to prevent numerical overflow/underflow in exp()
        # Matches the clamping bounds used in the autoencoder
        logvar = torch.clamp(logvar, min=-10, max=10)

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
