import torch

from models.cspn import AbstractCSPN
from models.cspn.abstract_cspn import CSPNType
from models.cspn.psinet.einsum_network import EinsumNetwork, Args
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.graph import random_binary_trees
from models.cspn.psinet.conditioning_nn import build_conditioning_mlp_for


# TODO add more parameters
class PsiNetCSPN(AbstractCSPN):
    def __init__(self, latent_dim: int, num_classes: int, h_dims=None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        if h_dims is None:
            h_dims = [100]

        self.graph = random_binary_trees(
            num_var=latent_dim, depth=4, num_repetitions=10
        )

        self.args = Args(
            num_var=latent_dim,
            num_dims=1,
            num_input_distributions=10,
            num_sums=10,
            num_classes=1,
            exponential_family=NormalArray,
            exponential_family_args={"min_var": 1e-3, "max_var": 1.0},
            use_em=False,
        )

        self.einet = EinsumNetwork(
            graph=self.graph,
            param_nn=None,
            args=self.args,
        )
        self.einet.initialize()

        conditioning_network = build_conditioning_mlp_for(
            self.einet, num_classes=num_classes, h_dims=h_dims
        )

        self.einet.param_nn = conditioning_network

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        results = torch.zeros(z.shape[0], 1, device=z.device)
        for cls in labels.unique():
            mask = labels == cls
            results[mask] = self.einet.forward(x=z[mask], y=labels[mask])
        return results

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        results = torch.zeros(labels.shape[0], self.latent_dim, device=labels.device)
        for cls in labels.unique():
            mask = labels == cls
            cls_labels = labels[mask]
            results[mask] = self.einet.sample(y=cls_labels)
        return results

    def get_config(self) -> dict:
        return {
            "model_type": CSPNType.PSINET,
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
        }
