import math
from models.cspn import AbstractCSPN
from models.cspn.CustomEinet.nn_for_einet import EinetConditioningNetwork
from models.cspn.CustomEinet.einsum_layer import EinsumLayer
from models.cspn.CustomEinet.gaussian_leaf_layer import GaussianLeafLayer
import torch
import torch.nn.functional as F
from torch import nn


# TODO this does not look right
class Einet(AbstractCSPN):
    def __init__(
        self,
        num_vars: int,
        context_dim: int,
        num_leaves: int,
        num_nodes: int,
        nn_hidden_dim: int = 256,
        nn_num_hidden_layers: int = 2,
    ):
        """
        :param num_vars:    Number of input variables. Must be a power of 2.
        :param context_dim: Number of classes for one-hot encoding (this is the context dimension).
        :param num_leaves:  Number of leaf components per scope (bottom of tree).
        :param num_nodes:   Number of nodes at the first einsum layer (halves each layer up).
        :param nn_hidden_dim: Hidden layer dimension for the conditioning network (default 256).
        :param nn_num_hidden_layers: Number of hidden layers in the conditioning network (default 2).
        """
        super().__init__()
        assert num_vars > 0 and (num_vars & (num_vars - 1)) == 0, (
            "num_vars must be a power of 2"
        )

        self._num_classes = context_dim
        self.num_vars = num_vars
        self.depth = int(math.log2(num_vars))

        self.layer_nodes = [max(2, num_nodes // (2**l)) for l in range(self.depth)]

        self.leaf_layer = GaussianLeafLayer(num_scopes=num_vars, num_leaves=num_leaves)

        self.einsum_layers = nn.ModuleList(
            [
                EinsumLayer(
                    num_input_nodes=num_leaves if l == 0 else self.layer_nodes[l - 1],
                    num_output_nodes=self.layer_nodes[l],
                )
                for l in range(self.depth)
            ]
        )

        self.root_weights = nn.Parameter(torch.zeros(self.layer_nodes[-1]))

        self.cond_net = EinetConditioningNetwork(
            context_dim=context_dim,
            num_scopes=num_vars,
            num_leaves=num_leaves,
            layer_nodes=self.layer_nodes,
            nn_hidden_dim=nn_hidden_dim,
            nn_num_hidden_layers=nn_num_hidden_layers,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes for one-hot encoding."""
        return self._num_classes

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param z:       (N, num_vars)
        :param labels:  (N,) class labels
        :return:        (N,) log p(z | labels)
        """
        context = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        mu, logvar, all_weights = self.cond_net(context)
        # mu, logvar:  (N, num_vars, num_leaves)
        # all_weights: list of (N, num_pairs_l, out_nodes_l, in_nodes_l, in_nodes_l)

        # Leaf log-densities: (N, num_vars, num_leaves)
        log_leaves = self.leaf_layer(z, mu, logvar)

        # Bottom-up: h starts as (N, num_vars, num_leaves), regions halve each layer
        h = log_leaves

        for l, einsum_layer in enumerate(self.einsum_layers):
            # h: (N, num_regions, num_input_nodes)
            left = h[:, 0::2, :]  # (N, num_pairs, in_nodes)
            right = h[:, 1::2, :]  # (N, num_pairs, in_nodes)

            N, num_pairs, _ = left.shape
            weights_l = all_weights[l]  # (N, num_pairs, out_nodes, in_nodes, in_nodes)

            # Flatten pairs into batch dim, run einsum layer, restore
            left_flat = left.reshape(N * num_pairs, -1)
            right_flat = right.reshape(N * num_pairs, -1)
            w_flat = weights_l.reshape(
                N * num_pairs,
                einsum_layer.num_output_nodes,
                einsum_layer.num_input_nodes,
                einsum_layer.num_input_nodes,
            )
            h_flat = einsum_layer(left_flat, right_flat, w_flat)  # (N*P, out_nodes)
            h = h_flat.reshape(N, num_pairs, einsum_layer.num_output_nodes)

        # h: (N, 1, root_nodes) -> (N, root_nodes)
        h = h.squeeze(1)
        log_p = torch.logsumexp(h + self.root_weights.log_softmax(dim=-1), dim=-1)
        return log_p

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Ancestral sampling top-down through the binary tree.

        :param labels:  (N,) class labels
        :return:        (N, num_vars)
        """
        # Convert labels to one-hot encoding internally
        context = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        N = context.shape[0]
        mu, logvar, all_weights = self.cond_net(context)

        # Step 1: sample root node -> (N, 1)
        root_probs = self.root_weights.softmax(dim=-1)
        selected = torch.multinomial(root_probs.expand(N, -1), num_samples=1)  # (N, 1)

        # Step 2: top-down through layers, doubling regions at each step
        for l in reversed(range(self.depth)):
            einsum_layer = self.einsum_layers[l]
            weights_l = all_weights[l]  # (N, num_pairs, out_nodes, in_nodes, in_nodes)
            num_pairs = weights_l.shape[1]
            in_nodes = einsum_layer.num_input_nodes

            # Gather weight matrix for the selected output node per region
            # selected: (N, num_pairs)
            sel_exp = selected.view(N, num_pairs, 1, 1, 1).expand(
                N, num_pairs, 1, in_nodes, in_nodes
            )
            w_sel = weights_l.gather(dim=2, index=sel_exp).squeeze(2)
            # w_sel: (N, num_pairs, in_nodes, in_nodes)

            # Sample (i, j) jointly
            w_flat = w_sel.view(N * num_pairs, in_nodes * in_nodes).softmax(dim=-1)
            ij = torch.multinomial(w_flat, num_samples=1).squeeze(1)  # (N*num_pairs,)
            i_sel = (ij // in_nodes).view(N, num_pairs)
            j_sel = (ij % in_nodes).view(N, num_pairs)

            # Interleave left (i) and right (j) to double the region count
            selected = torch.stack([i_sel, j_sel], dim=2).view(N, num_pairs * 2)

        # selected: (N, num_vars) — leaf component index per variable

        # Step 3: sample from selected leaf Gaussians
        sel_leaf = selected.unsqueeze(2)  # (N, num_vars, 1)
        mu_sel = mu.gather(dim=2, index=sel_leaf).squeeze(2)
        std_sel = (0.5 * logvar.gather(dim=2, index=sel_leaf)).exp().squeeze(2)

        return torch.normal(mu_sel, std_sel)  # (N, num_vars)

    def get_config(self):
        return {}
