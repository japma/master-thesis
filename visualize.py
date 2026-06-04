"""Inference and visualization entrypoint."""

from pathlib import Path

from config import load_config
from models.autoencoder import create_autoencoder
from models.cspn.einet import Einet
from dataset_loaders import build_data_loaders
from inference import (
    run_ae_inference,
    run_cspn_inference,
    save_combined_latent_umap,
)
from utils import seed_everything, resolve_device, load_checkpoint


def main():
    """Run inference and visualization for both AE and CSPN models."""
    cfg = load_config()

    # Setup
    seed = seed_everything(cfg.seed)
    device = resolve_device()
    dataset_cfg = cfg.dataset
    dataset_name = dataset_cfg.name

    print(f"Running inference on {dataset_name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # Initialize autoencoder
    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)
    ae = create_autoencoder(
        model_type=cfg.autoencoder.model_type,
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        base_channels=cfg.autoencoder.base_channels,
        num_blocks=cfg.autoencoder.num_blocks,
        res_blocks=cfg.autoencoder.res_blocks,
    )

    # Load pretrained autoencoder
    ae_ckpt_path = Path(f"checkpoints/{dataset_name}/autoencoder.pt")
    if not ae_ckpt_path.exists():
        raise FileNotFoundError(
            f"Autoencoder checkpoint not found at {ae_ckpt_path}. "
            f"Please train the autoencoder first: python train_ae.py dataset={dataset_name}"
        )
    # ae_ckpt = load_checkpoint(ae_ckpt_path, device)
    # ae.load_state_dict(ae_ckpt)
    print(f"Loaded autoencoder from {ae_ckpt_path}")

    # Initialize CSPN model
    cspn = Einet(
        num_vars=cfg.dataset.latent_size,
        context_dim=cfg.dataset.num_classes,
        num_leaves=cfg.cspn.num_leaves,
        num_nodes=cfg.cspn.num_nodes,
        nn_hidden_dim=cfg.cspn.nn_hidden_dim,
        nn_num_hidden_layers=cfg.cspn.nn_num_hidden_layers,
    )

    # Load pretrained CSPN
    cspn_ckpt_path = Path(f"checkpoints/{dataset_name}/cspn.pt")
    if not cspn_ckpt_path.exists():
        raise FileNotFoundError(
            f"CSPN checkpoint not found at {cspn_ckpt_path}. "
            f"Please train the CSPN first: python train_cspn.py dataset={dataset_name}"
        )
    cspn_ckpt = load_checkpoint(cspn_ckpt_path, device)
    cspn.load_state_dict(cspn_ckpt)
    print(f"Loaded CSPN from {cspn_ckpt_path}")

    # Load data
    _, test_loader = build_data_loaders(dataset_cfg, batch_size=cfg.training.batch_size)

    # Load test dataset separately to get class_names if available
    from dataset_loaders.helpers import _DATASETS

    loader_fn = _DATASETS.get(dataset_cfg.name)
    if loader_fn is None:
        raise ValueError(f"Unsupported dataset '{dataset_cfg.name}'")
    test_dataset = loader_fn(train=False)

    # Get class names if available
    class_names = getattr(test_dataset, "class_names", None)

    # Run inference
    print("\nRunning AE inference...")
    ae_latents, ae_labels = run_ae_inference(
        model=ae, data_loader=test_loader, device=device
    )

    print("Running CSPN inference...")
    cspn_latents, cspn_labels = run_cspn_inference(
        model=cspn,
        data_loader=test_loader,
        device=device,
        autoencoder=ae,
        class_names=class_names,
    )

    # Create combined visualization
    print("Creating combined UMAP visualization...")
    save_combined_latent_umap(
        ae_latents=ae_latents,
        cspn_latents=cspn_latents,
        ae_labels=ae_labels,
        cspn_labels=cspn_labels,
        path="combined_ae_cspn_umap.png",
    )

    print("\nInference complete! Visualizations saved.")


if __name__ == "__main__":
    main()
