import torch
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.leaves.normal import Normal
from spflow.zoo.einet import Einet

from models.cspn.abstract_cspn import AbstractCSPN, CSPNType
from models.cspn.spflow.nn_for_spflow import NeuralNetworkForSPFlow


class SPFlowCSPN(AbstractCSPN):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        num_sums: int = 10,
        num_leaves: int = 10,
        depth: int = 3,
        num_repetitions: int = 5,
        nn_layers: int = 2,
        nn_hidden_dim: int = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_sums = num_sums
        self.num_leaves = num_leaves
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.nn_layers = nn_layers
        self.nn_hidden_dim = nn_hidden_dim

        self.conditioning_net = NeuralNetworkForSPFlow(
            num_classes=num_classes,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            num_layers=nn_layers,
            hidden_dim=nn_hidden_dim,
        )

        leaf_modules: list[LeafModule] = [
            Normal(
                scope=Scope([i], evidence=[latent_dim]),
                out_channels=num_leaves,
                num_repetitions=num_repetitions,
                parameter_fn=self.conditioning_net,
            )
            for i in range(latent_dim)
        ]

        self.einet = Einet(
            leaf_modules=leaf_modules,
            num_classes=num_classes,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_repetitions,
            layer_type="einsum",
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_col = labels.unsqueeze(1)
        data = torch.cat([z, labels_col], dim=1)
        return self.einet.log_likelihood(data)

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        batch = labels.shape[0]
        nan_latents = torch.full(
            (batch, self.latent_dim), float("nan"), device=labels.device
        )
        labels_col = labels.unsqueeze(1).float()
        data = torch.cat([nan_latents, labels_col], dim=1)
        result = self.einet.sample(data=data)
        assert isinstance(result, torch.Tensor)
        return result[:, : self.latent_dim]

    def get_config(self) -> dict:
        return {
            "model_type": CSPNType.SPFLOW,
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
            "num_sums": self.num_sums,
            "num_leaves": self.num_leaves,
            "depth": self.depth,
            "num_repetitions": self.num_repetitions,
            "nn_layers": self.nn_layers,
            "nn_hidden_dim": self.nn_hidden_dim,
        }
