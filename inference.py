"""Inference pipeline for loading checkpoints and generating visualizations."""

import logging

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset_loaders import get_data_loaders
from models.autoencoder import VariationalAutoencoder, AbstractAutoencoder

from utils.io import build_ae_path, load_checkpoint
from utils.train import resolve_device
from utils.visualization import save_reconstructions

logger = logging.getLogger(__name__)


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


def run_inference(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset
    logger.info(f"Dataset: {dataset_cfg.name}")

    inference_cfg = cfg.inference
    hydra_cfg = HydraConfig().get()
    device = resolve_device()
    logger.info(f"Inference device: {device}")

    ae_cfg = cfg.autoencoder
    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

    ae_path = build_ae_path(cfg)
    logger.info(f"Autoencoder checkpoint: {ae_path}")
    ae_checkpoint = load_checkpoint(ae_path, map_location="cpu")

    ae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_size=dataset_cfg.latent_size,
        base_channels=ae_cfg.base_channels,
    )
    ae.load_state_dict(ae_checkpoint)
    ae.to(device)
    ae.eval()

    _, test = get_data_loaders(dataset_cfg)

    logger.info("Reconstructing...")
    originals, reconstructed, labels = _build_reconstructions(
        ae=ae,
        dataloader=test,
        num_images=inference_cfg.image_count,
        device=device,
    )

    recon_path = f"{hydra_cfg.runtime.output_dir}/reconstructions.png"
    save_reconstructions(originals, reconstructed, labels, recon_path)
