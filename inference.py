"""Inference pipeline for loading checkpoints and generating visualizations."""

import logging
from pathlib import Path

import torch

from models.autoencoder.variational_autoencoder import VariationalAutoencoder
from models.cspn import SPFlowCSPN
from utils.config import infer_image_shape_from_input_size
from utils import (
    get_data_loaders,
    load_checkpoint,
    visualize_autoencoder,
    visualize_cspn,
    visualize_cspn_latent_space,
    visualize_latent_space,
)

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_inference_output_dir(base_dir: str | Path) -> Path:
    output_dir = Path(base_dir) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_inference(cfg) -> None:
    """Load trained checkpoints and generate inference visualizations."""
    device = _resolve_device()

    input_size = cfg.data.input_size
    channels = cfg.data.channels
    height = cfg.data.height
    width = cfg.data.width

    if channels is None or height is None or width is None:
        channels, height, width = infer_image_shape_from_input_size(input_size)

    latent_size = cfg.data.latent_size
    cspn_num_labels = cfg.data.num_classes

    autoencoder = VariationalAutoencoder(
        input_size=input_size,
        latent_size=latent_size,
        image_shape=(channels, height, width),
    ).to(device)

    cspn = SPFlowCSPN(
        latent_size=latent_size,
        num_labels=cspn_num_labels,
    ).to(device)

    autoencoder_state_dict = load_checkpoint(
        cfg.checkpoint_dir,
        "autoencoder",
        map_location=device,
    )
    cspn_state_dict = load_checkpoint(
        cfg.checkpoint_dir,
        "cspn",
        map_location=device,
    )

    autoencoder.load_state_dict(autoencoder_state_dict)
    cspn.load_state_dict(cspn_state_dict)

    _, test_loader = get_data_loaders(
        cfg.data.name,
        cfg.model.training.batch_size,
        dataset_kwargs=cfg.data.dataset_kwargs,
    )

    output_dir = _create_inference_output_dir(cfg.run_dir)
    logger.info("Using device: %s", device)
    logger.info("Inference output directory: %s", output_dir)

    if cfg.visualize.autoencoder:
        visualize_autoencoder(
            autoencoder,
            test_loader,
            device,
            output_dir,
            num_samples=cfg.num_samples,
        )

    if cfg.visualize.latent_space:
        visualize_latent_space(
            autoencoder=autoencoder,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            num_labels=cspn_num_labels,
            max_points=cfg.max_points,
        )

    if cfg.visualize.cspn:
        visualize_cspn(
            autoencoder=autoencoder,
            cspn=cspn,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            num_labels=cspn_num_labels,
            num_samples=cfg.num_samples,
        )

    if cfg.visualize.cspn_latent_space:
        visualize_cspn_latent_space(
            autoencoder=autoencoder,
            cspn=cspn,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            num_labels=cspn_num_labels,
            max_points=cfg.max_points,
            samples_per_label=cfg.samples_per_label,
        )
