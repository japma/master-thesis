import torch
import math

from models.cspn import AbstractCSPN
from models.cspn.abstract_cspn import CSPNType
from models.cspn.psinet.einsum_network import EinsumNetwork, Args
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.graph import random_binary_trees
from models.cspn.psinet.conditioning_nn import build_conditioning_mlp_for


# TODO add more parameters
class PsiNetCSPN(AbstractCSPN):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        num_repetitions: int = 10,
        num_input_distributions: int = 10,
        num_sums: int = 10,
        min_var: float = 1e-3,
        max_var: float = 1.0,
        h_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.config = {
            "model_type": CSPNType.PSINET,
            "latent_dim": latent_dim,
            "num_classes": num_classes,
            "num_repetitions": num_repetitions,
            "num_input_distributions": num_input_distributions,
            "num_sums": num_sums,
            "min_var": min_var,
            "max_var": max_var,
            "h_dims": h_dims,
        }

        if h_dims is None:
            h_dims = [100]

        depth = math.floor(math.log2(latent_dim))

        self.graph = random_binary_trees(
            num_var=latent_dim, depth=depth, num_repetitions=num_repetitions
        )

        self.args = Args(
            num_var=latent_dim,
            num_dims=1,
            num_input_distributions=num_input_distributions,
            num_sums=num_sums,
            num_classes=1,
            exponential_family=NormalArray,
            exponential_family_args={"min_var": min_var, "max_var": max_var},
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
        # results = torch.zeros(z.shape[0], 1, device=z.device)
        # for cls in labels.unique():
        #     mask = labels == cls
        #     results[mask] = self.einet.forward(x=z[mask], y=labels[mask])
        # return results
        return self.einet.forward(x=z, y=labels)

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        # results = torch.zeros(labels.shape[0], self.latent_dim, device=labels.device)
        # for cls in labels.unique():
        #    mask = labels == cls
        #    cls_labels = labels[mask]
        #    results[mask] = self.einet.sample(y=cls_labels)
        return self.einet.sample(y=labels)

    def get_config(self) -> dict:
        return self.config
