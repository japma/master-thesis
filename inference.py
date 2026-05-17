"""Inference pipeline for loading checkpoints and generating visualizations."""

from models.autoencoder import AbstractAutoencoder
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Any

from models.cspn import AbstractCSPN
from utils.visualization import save_latent_umap


NUM_CLASSES = 10
SAMPLES_PER_CLASS = 100


def _collect_labeled_batch(
    data_loader: DataLoader, target_count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    images_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    collected = 0

    for images, labels in data_loader:
        remaining = target_count - collected
        if remaining <= 0:
            break
        take = min(images.shape[0], remaining)
        images_parts.append(images[:take].to(device))
        labels_parts.append(labels[:take].to(device))
        collected += take

    if collected < target_count:
        raise ValueError(
            f"Requested {target_count} samples, but data_loader only provided {collected}."
        )

    return torch.cat(images_parts, dim=0), torch.cat(labels_parts, dim=0)


def _build_cspn_context(model: nn.Module, labels: torch.Tensor) -> torch.Tensor:
    cond_net: Any = getattr(model, "cond_net", None)
    if cond_net is not None:
        mlp: Any = getattr(cond_net, "mlp", None)
        context_dim = mlp[0].in_features
        return F.one_hot(labels, num_classes=context_dim).float()

    return labels.view(-1, 1).float()


def run_ae_inference(
    model: AbstractAutoencoder, data_loader: DataLoader | None, device: torch.device
):
    if data_loader is None:
        raise ValueError("run_ae_inference requires a data_loader with class labels.")

    model.eval()
    target_count = NUM_CLASSES * SAMPLES_PER_CLASS

    with torch.no_grad():
        sample_images, sample_labels = _collect_labeled_batch(
            data_loader=data_loader,
            target_count=target_count,
            device=device,
        )
        sampled_latents = model.encode(sample_images)

    save_latent_umap(sampled_latents, labels=sample_labels, path="ae.png")


def run_cspn_inference(
    model: AbstractCSPN, data_loader: DataLoader | None, device: torch.device
):
    model.eval()
    sample_labels = torch.arange(NUM_CLASSES, device=device).repeat(SAMPLES_PER_CLASS)
    sample_context = _build_cspn_context(model, sample_labels).to(device)
    with torch.no_grad():
        sampled_latents = model.sample(sample_context)

    save_latent_umap(sampled_latents, labels=sample_labels, path="cspn.png")
