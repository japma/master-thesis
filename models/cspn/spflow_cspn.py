from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_cspn import AbstractCSPN
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.product import Product


class _ConditionalNormalParams(nn.Module):
    """Neural parameter function for SPFlow conditional Normal leaves."""

    def __init__(
        self,
        latent_size: int,
        num_labels: int,
        num_components: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.num_components = num_components

        self.label_embedding = nn.Embedding(num_labels, embedding_dim)

        layers = []
        in_dim = embedding_dim
        for _ in range(max(1, num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.loc_head = nn.Linear(hidden_dim, latent_size * num_components)
        self.scale_head = nn.Linear(hidden_dim, latent_size * num_components)

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        if evidence.dim() == 1:
            evidence = evidence.unsqueeze(1)

        labels = evidence[:, 0].long().clamp(min=0, max=self.num_labels - 1)
        h = self.backbone(self.label_embedding(labels))

        loc = self.loc_head(h).view(-1, self.latent_size, self.num_components, 1)
        raw_scale = self.scale_head(h).view(
            -1, self.latent_size, self.num_components, 1
        )
        scale = F.softplus(raw_scale) + 1e-3

        return {"loc": loc, "scale": scale}


class SPFlowCSPN(AbstractCSPN):
    """Conditional Sum Product Network."""

    def __init__(
        self,
        latent_size,
        num_labels,
        label_embedding_dim=32,
        context_hidden_dim=128,
        context_num_layers=3,
        num_mixture_components=4,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.num_labels = num_labels
        self.label_embedding_dim = label_embedding_dim
        self.context_hidden_dim = context_hidden_dim
        self.context_num_layers = context_num_layers
        self.num_mixture_components = num_mixture_components

        self.label_feature_index = latent_size
        y_scope = Scope(query=[self.label_feature_index])

        self.num_joint_channels = max(1, num_mixture_components)

        self.label_prior = Categorical(
            scope=y_scope,
            out_channels=self.num_joint_channels,
            K=num_labels,
        )

        self.conditional_param_fn = _ConditionalNormalParams(
            latent_size=latent_size,
            num_labels=num_labels,
            num_components=self.num_joint_channels,
            embedding_dim=label_embedding_dim,
            hidden_dim=context_hidden_dim,
            num_layers=context_num_layers,
        )

        self.z_given_y = Normal(
            scope=Scope(
                query=list(range(latent_size)),
                evidence=[self.label_feature_index],
            ),
            out_channels=self.num_joint_channels,
            parameter_fn=self.conditional_param_fn,
        )

        self.joint_model = Product(inputs=[self.z_given_y, self.label_prior])

    def _flatten_ll(self, ll: torch.Tensor) -> torch.Tensor:
        if ll.dim() <= 1:
            return ll
        return ll.reshape(ll.shape[0], -1).mean(dim=1)

    def _labels_to_tensor(self, labels) -> torch.Tensor:
        device = next(self.parameters()).device
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device=device, dtype=torch.long)

        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        return labels

    def _pack_joint(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = self._labels_to_tensor(labels)
        if z.dim() != 2 or z.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected z with shape (batch_size, {self.latent_size}), got {tuple(z.shape)}"
            )
        if z.shape[0] != labels.shape[0]:
            raise ValueError(
                "Batch mismatch between z and labels: "
                f"{z.shape[0]} != {labels.shape[0]}"
            )

        if labels.min() < 0 or labels.max() >= self.num_labels:
            raise ValueError(
                f"labels must be in [0, {self.num_labels - 1}], "
                f"got min={int(labels.min())}, max={int(labels.max())}"
            )

        y = labels.to(dtype=z.dtype).unsqueeze(1)
        return torch.cat([z, y], dim=1)

    def log_joint(self, z, labels):
        joint_data = self._pack_joint(z, labels)
        return self._flatten_ll(self.joint_model.log_likelihood(joint_data))

    def log_marginal_labels(self, labels):
        labels = self._labels_to_tensor(labels)
        device = labels.device
        z_nan = torch.full(
            (labels.shape[0], self.latent_size),
            torch.nan,
            dtype=torch.float32,
            device=device,
        )
        joint_data = self._pack_joint(z_nan, labels)
        return self._flatten_ll(self.joint_model.log_likelihood(joint_data))

    def forward(self, z, labels):
        log_joint = self.log_joint(z, labels)
        log_p_y = self.log_marginal_labels(labels)
        return log_joint - log_p_y

    def predict_latent(self, labels):
        labels = self._labels_to_tensor(labels)

        # Estimate E[z | y] directly from the conditional leaf distribution.
        evidence = labels.to(dtype=torch.float32).unsqueeze(1)
        cond_dist = self.z_given_y.conditional_distribution(evidence)
        loc = cond_dist.mean

        while loc.dim() > 2:
            loc = loc.mean(dim=-1)
        if loc.dim() == 1:
            loc = loc.unsqueeze(1)
        if loc.shape[1] != self.latent_size:
            raise RuntimeError(
                "Unexpected conditional mean shape from SPFlow leaf: "
                f"{tuple(loc.shape)}"
            )

        return loc

    def transform_latent(self, z, source_labels, target_labels, strength=1.0):
        source_center = self.predict_latent(source_labels).to(
            device=z.device, dtype=z.dtype
        )
        target_center = self.predict_latent(target_labels).to(
            device=z.device, dtype=z.dtype
        )
        return z + strength * (target_center - source_center)

    def sample(self, labels, num_samples=1):
        labels = self._labels_to_tensor(labels)

        if labels.numel() == 1 and num_samples > 1:
            labels = labels.repeat(num_samples)
        elif num_samples == 1:
            num_samples = labels.numel()
        elif labels.numel() != num_samples:
            raise ValueError(
                "labels length must equal num_samples, unless one label is provided "
                "to broadcast across all samples."
            )

        evidence = torch.full(
            (num_samples, self.latent_size + 1),
            torch.nan,
            dtype=torch.float32,
            device=labels.device,
        )
        evidence[:, self.label_feature_index] = labels.to(dtype=torch.float32)

        samples = torch.as_tensor(
            self.joint_model.sample_with_evidence(evidence=evidence)
        )
        return samples[:, : self.latent_size]
