"""Loading the VAE and the conditional models behind the evaluation protocols."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from models.autoencoder import AbstractAutoencoder
from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.joint_pc import JointPC
from models.neural_baseline.abstract_nn import AbstractNeuralBaseline
from utils.checkpoints import (
    load_ae_from_path,
    load_cspn_from_path,
    load_joint_pc_from_path,
    load_nn_baseline_from_path,
)
from utils.wandb_utils import download_artifact

MODEL_TYPES: tuple[str, ...] = ("cspn", "joint_pc", "nn_baseline")


class LabelConditionedModel(Protocol):
    """What the CSPN, JointPC and the neural baselines already share."""

    def sample(self, labels: Tensor, std_correction: float = 1.0) -> Tensor: ...


@dataclass(frozen=True)
class StdCorrectedSampler:
    """A `ConditionalSampler` over a model that samples one latent per label row."""

    model: LabelConditionedModel
    std_correction: float

    def sample(self, y: Tensor, n_per_label: int) -> Tensor:
        labels = y.repeat_interleave(n_per_label, dim=0)
        latents = self.model.sample(labels, std_correction=self.std_correction)
        return latents.reshape(y.shape[0], n_per_label, -1)


def load_vae(artifact: str, device: torch.device) -> tuple[AbstractAutoencoder, str]:
    """The VAE behind a wandb `name[:version]`, and the exact `name:vN` it resolved to."""
    path, resolved = download_artifact(artifact)
    vae = load_ae_from_path(path, device=device)
    return vae.to(device).eval(), resolved


def load_model(
    model_type: str, artifact: str, device: torch.device
) -> tuple[AbstractCSPN | JointPC | AbstractNeuralBaseline, str]:
    """The model behind a wandb `name[:version]`, and the exact `name:vN` it resolved to."""
    path, resolved = download_artifact(artifact)
    match model_type:
        case "cspn":
            model = load_cspn_from_path(path, device=device)
        case "joint_pc":
            model = load_joint_pc_from_path(path, device=device)
        case "nn_baseline":
            model = load_nn_baseline_from_path(path, device=device)
        case _:
            raise ValueError(
                f"unknown model type {model_type!r}, expected {MODEL_TYPES}"
            )
    return model.to(device).eval(), resolved


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
