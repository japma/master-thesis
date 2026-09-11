"""VAE whose label blocks are pinned to fixed latent coordinates."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.config import AutoencoderConfig

from .variational_autoencoder import VAEForwardOutput, VariationalAutoencoder


@dataclass
class AnchoredVAEForwardOutput(VAEForwardOutput):
    """`logits` holds one tensor per *head* factor, in target-vector order. Anchored
    factors have no head: their supervision is the conditional prior, which needs the
    labels and so lives in the loss."""

    logits: list[torch.Tensor]


class AnchoredVAE(VariationalAutoencoder):
    """A VAE in which some label factors are anchored and the rest keep a linear head.    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__(config)
        supervision = config.supervision
        if supervision is None:
            raise ValueError("AnchoredVAE needs config.supervision")

        self.supervision = supervision
        self.label_slices: list[slice] = supervision.slices()
        self.head_factors: list[int] = supervision.head_factors
        self.anchored_factors: list[int] = supervision.anchored_factors

        heads = [
            nn.Linear(supervision.dims[i], supervision.cardinalities[i])
            for i in self.head_factors
        ]
        for head in heads:
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
        self.heads = nn.ModuleList(heads)

    def classify(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [
            head(z[:, self.label_slices[i]])
            for head, i in zip(self.heads, self.head_factors, strict=True)
        ]

    def forward(self, x: torch.Tensor) -> AnchoredVAEForwardOutput:
        outputs = super().forward(x)
        assert isinstance(outputs, VAEForwardOutput)
        return AnchoredVAEForwardOutput(
            reconstructed=outputs.reconstructed,
            latent=outputs.latent,
            mu=outputs.mu,
            log_var=outputs.log_var,
            logits=self.classify(outputs.latent),
        )
