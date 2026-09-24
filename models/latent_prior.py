"""Unconditional samplers over a VAE's latent space: baselines that ignore the labels."""

import torch
import torch.nn as nn


class StandardNormalPrior(nn.Module):
    """The VAE's own prior, N(0, I). `labels` only sets the batch size."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        shape = (labels.shape[0], self.latent_dim)
        return torch.randn(shape, device=labels.device) * std_correction


class GaussianMixturePrior(nn.Module):
    """Ex-post density estimation (Ghosh et al., 2020): a full-covariance Gaussian
    mixture fitted to the encoded training set, sampled in place of the prior."""

    weights: torch.Tensor
    means: torch.Tensor
    scale_tril: torch.Tensor

    def __init__(self, num_components: int, latent_dim: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.register_buffer("weights", torch.full((num_components,), 1 / num_components))
        self.register_buffer("means", torch.zeros(num_components, latent_dim))
        self.register_buffer(
            "scale_tril", torch.eye(latent_dim).expand(num_components, -1, -1).clone()
        )

    def sample(self, labels: torch.Tensor, std_correction: float = 1.0) -> torch.Tensor:
        n = labels.shape[0]
        component = torch.multinomial(self.weights, n, replacement=True)
        noise = torch.randn(n, self.latent_dim, 1, device=self.means.device)
        spread = (self.scale_tril[component] @ noise).squeeze(-1)
        return self.means[component] + std_correction * spread

    def get_config(self) -> dict:
        return {"num_components": self.num_components, "latent_dim": self.latent_dim}
