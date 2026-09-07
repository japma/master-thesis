"""VAE whose latent blocks are forced to carry named label factors."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.config import AutoencoderConfig

from .variational_autoencoder import VAEForwardOutput, VariationalAutoencoder


@dataclass
class SupervisedVAEForwardOutput(VAEForwardOutput):
    """`logits` holds one tensor per label factor, in target-vector order."""

    logits: list[torch.Tensor]


class SupervisedVAE(VariationalAutoencoder):
    """A VAE with one linear classification head per label factor, each reading only
    its own contiguous block of the latent.

    Encoder and decoder are untouched, so `encode` stays exactly the interface the
    CSPN and the probes consume -- the heads exist only to shape the latent during
    training, and the linear (rather than MLP) head is what makes "this attribute
    lives in these dimensions" the thing being optimised.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__(config)
        supervision = config.supervision
        if supervision is None:
            raise ValueError("SupervisedVAE needs config.supervision")

        self.supervision = supervision
        self.label_slices: list[slice] = supervision.slices()
        heads = [
            nn.Linear(dim, cardinality)
            for dim, cardinality in zip(
                supervision.dims, supervision.cardinalities, strict=True
            )
        ]
        for head in heads:
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
        self.heads = nn.ModuleList(heads)

    def classify(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [
            head(z[:, block])
            for head, block in zip(self.heads, self.label_slices, strict=True)
        ]

    def forward(self, x: torch.Tensor) -> SupervisedVAEForwardOutput:
        outputs = super().forward(x)
        assert isinstance(outputs, VAEForwardOutput)
        return SupervisedVAEForwardOutput(
            reconstructed=outputs.reconstructed,
            latent=outputs.latent,
            mu=outputs.mu,
            log_var=outputs.log_var,
            logits=self.classify(outputs.latent),
        )
