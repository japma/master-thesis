import torch

from models.cspn import AbstractCSPN
from models.cspn.psinet.einsum_network import EinsumNetwork, Args
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.graph import random_binary_trees
from models.cspn.psinet.nns import MLP


# TODO add more parameters
class PsiNetCSPN(AbstractCSPN):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

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
            use_em=True,
        )

        conditioning_network = MLP(
            in_dim=num_classes,
            out_dims=[(self.args.num_input_distributions, self.args.num_classes)],
            h_dims=[100],
        )

        self.einet = EinsumNetwork(
            graph=self.graph,
            param_nn=conditioning_network,
            args=self.args,
        )

        self.einet.initialize()

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.einet.forward(x=z, y=labels)

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        sample = self.einet.sample(y=labels)

        if sample is None:
            raise ValueError

        return sample

    def get_config(self) -> dict:
        return {
            "model_type": "PsiNetCSPN",
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
        }
