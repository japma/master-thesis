"""Entry point for autoencoder training."""

from pathlib import Path

import torch
import wandb
from rtpt import RTPT
from torch import nn
from torchinfo import summary

from dataset_loaders import build_data_loaders
from models.autoencoder import create_autoencoder
from training.train_ae import train_autoencoder
from utils.config import load_config
from utils.reproducibility import resolve_device, seed_everything


def main():
    cfg = load_config()

    seed = seed_everything(cfg.seed)
    device = resolve_device()
    dataset_name = cfg.dataset.name
    run_name = f"Autoencoder_{dataset_name}_seed{seed}"

    print(f"Training Autoencoder on {dataset_name} | device={device} | seed={seed}")

    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)
    ae = create_autoencoder(
        model_type=cfg.autoencoder.model_type,
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        device=device,
        base_channels=cfg.autoencoder.base_channels,
        num_blocks=cfg.autoencoder.num_blocks,
        res_blocks=cfg.autoencoder.res_blocks,
    )
    print("Autoencoder Architecture:")
    summary(ae)

    train_loader, test_loader = build_data_loaders(
        cfg.dataset, batch_size=cfg.training.batch_size
    )

    optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.training.learning_rate)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss()
    # loss_fn = HybridLoss()
    # loss_fn = lpips.LPIPS(net="vgg").to(device)

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(cfg.training.epochs, 1),
    )
    rtpt.start()

    wandb.init(
        entity="jmartini-tu-darmstadt",
        project="master-thesis",
        name=run_name,
        config={
            "dataset": dataset_name,
            "model": "Autoencoder",
            "model_type": str(cfg.autoencoder.model_type),
            "epochs": cfg.training.epochs,
            "latent_dim": cfg.dataset.latent_size,
            "learning_rate": cfg.training.learning_rate,
            "beta_start": cfg.training.beta_start,
            "beta_end": cfg.training.beta_end,
            "beta_anneal_epochs": cfg.training.beta_anneal_epochs,
            "seed": seed,
            "base_channels": cfg.autoencoder.base_channels,
            "num_blocks": cfg.autoencoder.num_blocks,
        },
        mode=cfg.wandb,
    )

    train_autoencoder(
        model=ae,
        device=device,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        rtpt=rtpt,
    )

    ckpt_path = Path(f"checkpoints/{dataset_name}/autoencoder.pt")
    # save_checkpoint(ae.state_dict(), ckpt_path)

    artifact = wandb.Artifact(name=run_name, type="autoencoder")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
