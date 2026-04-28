"""Inference pipeline for loading checkpoints and generating visualizations."""

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


def _resolve_label_transform(label_mode: str, dataset_name: str):
    if label_mode == "dataset":
        return lambda labels: labels

    if label_mode == "mnist_parity":
        if dataset_name != "MNIST":
            raise ValueError(
                "label_mode='mnist_parity' is only supported with data.name='MNIST'."
            )
        return lambda labels: labels % 2

    raise ValueError(
        f"Unsupported CSPN label_mode '{label_mode}'. Supported modes: "
        "'dataset', 'mnist_parity'."
    )


def run_inference(cfg) -> None:
    """Load trained checkpoints and generate inference visualizations."""
    device = _resolve_device()

    dataset_name = cfg.data.name
    input_size = cfg.data.input_size
    channels = cfg.data.channels
    height = cfg.data.height
    width = cfg.data.width

    if channels is None or height is None or width is None:
        channels, height, width = infer_image_shape_from_input_size(input_size)

    latent_size = cfg.model.training.latent_size
    cspn_cfg = cfg.model.cspn
    cspn_label_config = cfg.cspn_label
    cspn_label_transform = _resolve_label_transform(
        cspn_label_config.label_mode,
        dataset_name,
    )
    cspn_num_labels = cspn_label_config.num_labels
    cspn_class_names = cspn_label_config.class_names

    autoencoder = VariationalAutoencoder(
        input_size=input_size,
        latent_size=latent_size,
        image_shape=(channels, height, width),
    ).to(device)

    cspn = SPFlowCSPN(
        latent_size=latent_size,
        num_labels=cspn_num_labels,
        label_embedding_dim=cspn_cfg.label_embedding_dim,
        context_hidden_dim=cspn_cfg.context_hidden_dim,
        context_num_layers=cspn_cfg.context_num_layers,
        num_mixture_components=cspn_cfg.num_mixture_components,
        num_sum_components=cspn_cfg.num_sum_components,
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
        dataset_name,
        cfg.model.training.batch_size,
        dataset_kwargs=cfg.data.dataset_kwargs,
    )

    output_dir = _create_inference_output_dir(cfg.run_dir)
    print(f"Using device: {device}")
    print(f"Inference output directory: {output_dir}")

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
            class_names=cspn_class_names,
            label_transform=cspn_label_transform,
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
            class_names=cspn_class_names,
            label_transform=cspn_label_transform,
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
            class_names=cspn_class_names,
            label_transform=cspn_label_transform,
        )
