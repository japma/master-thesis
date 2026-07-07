import math

import torch

from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.psinet.conditioning_nn import build_conditioning_mlp_for
from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.graph import random_binary_trees
from utils.config import CSPNConfig


# TODO add more parameters
class PsiNetCSPN(AbstractCSPN):
    def __init__(
        self,
        config: CSPNConfig,
    ) -> None:
        super().__init__()

        self.config = config

        depth = math.floor(math.log2(config.num_vars))

        self.graph = random_binary_trees(
            num_var=config.num_vars, depth=depth, num_repetitions=config.num_repetitions
        )

        self.args = Args(
            num_var=config.num_vars,
            num_dims=1,
            num_input_distributions=config.num_input_distributions,
            num_sums=config.num_sums,
            num_classes=1,
            exponential_family=NormalArray,
            exponential_family_args={
                "min_var": config.min_var,
                "max_var": config.max_var,
            },
            use_em=False,
        )

        self.einet = EinsumNetwork(
            graph=self.graph,
            param_nn=None,
            args=self.args,
        )
        self.einet.initialize()

        conditioning_network = build_conditioning_mlp_for(
            self.einet, num_classes=config.num_classes, h_dims=config.h_dims
        )

        self.einet.param_nn = conditioning_network

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.einet.forward(x=z, y=labels).squeeze(-1)

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        samples = self.einet.sample(y=labels)
        assert samples is not None
        return samples

    def get_config(self) -> dict:
        return self.config.model_dump()
