"""Inference pipeline for loading checkpoints and generating visualizations."""

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset_loaders import get_data_loaders
from models.autoencoder import VariationalAutoencoder, AbstractAutoencoder
from models import SPFlowCSPN

from utils.io import build_ae_path, load_checkpoint, build_cspn_path
from utils.train import resolve_device
from utils.visualization import save_reconstructions, save_latent_umap


def _build_reconstructions(
    ae: AbstractAutoencoder,
    dataloader: DataLoader,
    num_images: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_images <= 0:
        raise ValueError("inference.image_count must be > 0")

    originals = []
    reconstructed = []
    labels_list = []
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            recon, _, _ = ae(images)
            originals.append(images.cpu())
            reconstructed.append(recon.cpu())
            labels_list.append(labels.cpu())
            total += images.size(0)
            if total >= num_images:
                break

    if not originals:
        raise ValueError("No images available in dataloader for inference")

    originals_tensor = torch.cat(originals, dim=0)[:num_images]
    reconstructed_tensor = torch.cat(reconstructed, dim=0)[:num_images]
    labels_tensor = torch.cat(labels_list, dim=0)[:num_images].view(-1)
    return originals_tensor, reconstructed_tensor, labels_tensor


def _extract_latents(
    ae: AbstractAutoencoder,
    dataloader: DataLoader,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    latents_list = []
    labels_list = []
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            z = ae.encode(images)
            latents_list.append(z.cpu())
            labels_list.append(labels.cpu())
            total += images.size(0)
            if total >= num_samples:
                break

    if not latents_list:
        raise ValueError("No images available in dataloader")

    latents_tensor = torch.cat(latents_list, dim=0)[:num_samples]
    labels_tensor = torch.cat(labels_list, dim=0)[:num_samples].view(-1)
    return latents_tensor, labels_tensor


def run_inference(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset

    inference_cfg = cfg.inference
    hydra_cfg = HydraConfig().get()
    device = resolve_device()

    ae_cfg = cfg.autoencoder
    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

    ae_path = build_ae_path(cfg)
    ae_checkpoint = load_checkpoint(ae_path, map_location="cpu")

    ae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_size=dataset_cfg.latent_size,
        base_channels=ae_cfg.base_channels,
        num_blocks=ae_cfg.num_blocks,
        res_blocks=ae_cfg.res_blocks,
    )
    ae.load_state_dict(ae_checkpoint)
    ae.to(device)
    ae.eval()

    _, test = get_data_loaders(dataset_cfg)

    originals, reconstructed, labels = _build_reconstructions(
        ae=ae,
        dataloader=test,
        num_images=inference_cfg.image_count,
        device=device,
    )

    recon_path = f"{hydra_cfg.runtime.output_dir}/reconstructions.png"
    save_reconstructions(originals, reconstructed, labels, recon_path)

    ae_latents, ae_labels = _extract_latents(
        ae=ae,
        dataloader=test,
        num_samples=inference_cfg.umap_count,
        device=device,
    )

    ae_umap_path = f"{hydra_cfg.runtime.output_dir}/ae_latents_umap.png"
    save_latent_umap(
        ae_latents,
        labels=ae_labels,
        path=ae_umap_path,
        title="Autoencoder Latent Space",
    )

    cspn_path = build_cspn_path(cfg)
    if cspn_path.exists():
        cspn_checkpoint = load_checkpoint(cspn_path, map_location="cpu")

        cspn = SPFlowCSPN(
            latent_dim=dataset_cfg.latent_size,
            num_classes=dataset_cfg.num_classes,
        )
        cspn.load_state_dict(cspn_checkpoint)
        cspn.to(device)
        cspn.eval()
