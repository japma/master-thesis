from models.cspn.nn_for_einet import EinetConditioningNetwork
from models.cspn.einsum_layer import EinsumLayer
from models.cspn.gaussian_leaf_layer import GaussianLeafLayer
import torch
from torch import nn


class Einet(nn.Module):
    def __init__(self, num_vars, context_dim, num_leaves, num_nodes):
        super().__init__()
        self.leaf_layer = GaussianLeafLayer(num_scopes=num_vars, num_leaves=num_leaves)
        # TODO increase number of einsum layers
        self.einsum_layers = nn.ModuleList(
            [EinsumLayer(num_input_nodes=num_leaves, num_output_nodes=num_nodes)]
        )
        self.cond_net = EinetConditioningNetwork(
            context_dim=context_dim,
            num_scopes=num_vars,
            num_leaves=num_leaves,
            num_nodes=num_leaves,  # input nodes to einsum = num_leaves
            num_output_nodes=num_nodes,
        )
        # root mixing weights as logits (zeros = uniform init)
        self.root_weights = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, x, context):
        # (N, num_vars, num_leaves)
        mu, logvar, weights = self.cond_net(context)
        log_leaves = self.leaf_layer(x, mu, logvar)

        # split scopes into left / right halves -> (N, num_leaves)
        half = x.shape[1] // 2
        left = log_leaves[:, :half, :].sum(dim=1)  # (N, num_leaves)
        right = log_leaves[:, half:, :].sum(dim=1)  # (N, num_leaves)

        h = self.einsum_layers[0](left, right, weights)  # (N, num_nodes)

        # root: logits -> log_softmax, then logsumexp over nodes -> (N,)
        log_p = torch.logsumexp(h + self.root_weights.log_softmax(dim=-1), dim=-1)
        return log_p

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """
        :param context: Tensor of shape (N, context_dim)
        :return: Tensor of shape (N, num_vars)
        """
        N = context.shape[0]
        mu, logvar, weights = self.cond_net(context)

        root_probs = self.root_weights.softmax(dim=-1)  # (num_nodes,)
        k = torch.multinomial(root_probs.expand(N, -1), num_samples=1).squeeze(1)

        k_expanded = k.view(N, 1, 1, 1).expand(N, 1, weights.shape[2], weights.shape[3])
        w_k = weights.gather(dim=1, index=k_expanded).squeeze(1)
        w_k_flat = w_k.view(N, -1).softmax(dim=-1)
        ij = torch.multinomial(w_k_flat, num_samples=1).squeeze(1)
        # TODO implement correct halving of scopes for more than 1 einsum layer
        i = ij // self.einsum_layers.num_input_nodes
        j = ij % self.einsum_layer.num_input_nodes

        half = self.leaf_layer.num_scopes // 2

        i_exp = i.view(N, 1, 1).expand(N, half, 1)
        j_exp = j.view(N, 1, 1).expand(N, half, 1)

        mu_left = mu[:, :half, :].gather(dim=2, index=i_exp).squeeze(2)
        mu_right = mu[:, half:, :].gather(dim=2, index=j_exp).squeeze(2)
        std_left = (
            (0.5 * logvar[:, :half, :].gather(dim=2, index=i_exp)).exp().squeeze(2)
        )
        std_right = (
            (0.5 * logvar[:, half:, :].gather(dim=2, index=j_exp)).exp().squeeze(2)
        )

        x_left = torch.normal(mu_left, std_left)
        x_right = torch.normal(mu_right, std_right)

        return torch.cat([x_left, x_right], dim=1)
