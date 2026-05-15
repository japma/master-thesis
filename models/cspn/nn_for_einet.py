from torch import nn


class EinetConditioningNetwork(nn.Module):
    def __init__(
        self, context_dim, num_scopes, num_leaves, num_nodes, num_output_nodes
    ):
        super().__init__()
        # total params needed per sample:
        # leaves: num_scopes * num_leaves * 2  (mu + logvar)
        # einsum weights: num_output_nodes * num_nodes * num_nodes
        leaf_params = num_scopes * num_leaves * 2
        weight_params = num_output_nodes * num_nodes * num_nodes
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, leaf_params + weight_params),
        )
        self.num_scopes = num_scopes
        self.num_leaves = num_leaves
        self.num_nodes = num_nodes
        self.num_output_nodes = num_output_nodes

    def forward(self, context):  # context: (N, context_dim)
        context = context.float()
        out = self.mlp(context)
        S, K, O = self.num_scopes, self.num_leaves, self.num_output_nodes
        mu = out[:, : S * K].view(-1, S, K)
        logvar = out[:, S * K : 2 * S * K].view(-1, S, K)
        weights = out[:, 2 * S * K :].view(-1, O, self.num_nodes, self.num_nodes)
        return mu, logvar, weights
