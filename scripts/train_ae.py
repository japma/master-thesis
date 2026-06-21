"""Entry point for autoencoder training."""

from pathlib import Path

import torch
import wandb
from rtpt import RTPT
from torch import nn
from torchinfo import summary

from dataset_loaders import build_data_loaders
from models import VariationalAutoencoder
from training.train_ae import train_autoencoder
from utils.checkpoints import save_autoencoder
from utils.config import load_config, VariationalAutoencoderConfig
from utils.reproducibility import resolve_device, seed_everything


def main():
    cfg = load_config()
    dataset_cfg = cfg.dataset
    autoencoder_cfg = cfg.autoencoder
    assert isinstance(autoencoder_cfg, VariationalAutoencoderConfig), (
        "Only VariationalAutoencoderConfig is supported in this script"
    )
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg.seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name
    run_name = f"autoencoder_{dataset_name}"

    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=run_name,
        config={
            "dataset": dataset_name,
            "model": "Autoencoder",
            "model_type": "VariationalAutoencoder",
            "epochs": training_cfg.epochs,
            "latent_dim": autoencoder_cfg.latent_dim,
            "learning_rate": training_cfg.learning_rate,
            "beta_start": training_cfg.beta_start,
            "beta_end": training_cfg.beta_end,
            "beta_anneal_epochs": training_cfg.beta_anneal_epochs,
            "seed": seed,
            "base_channels": autoencoder_cfg.base_channels,
            "num_blocks": autoencoder_cfg.num_blocks,
        },
        mode=wandb_cfg.mode,
    )

    print(f"Training Autoencoder on {dataset_name} | device={device} | seed={seed}")

    input_shape = (dataset_cfg.channels, dataset_cfg.height, dataset_cfg.width)

    ae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_dim=autoencoder_cfg.latent_dim,
        base_channels=autoencoder_cfg.base_channels,
        num_blocks=autoencoder_cfg.num_blocks,
        res_blocks=autoencoder_cfg.res_blocks,
    )
    print("Autoencoder Architecture:")
    summary(ae)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer = torch.optim.Adam(ae.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )
    loss_fn = nn.BCELoss()
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss()
    # loss_fn = HybridLoss()
    # loss_fn = lpips.LPIPS(net="vgg").to(device)

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    train_autoencoder(
        model=ae,
        device=device,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        rtpt=rtpt,
    )

    ckpt_path = Path(cfg.paths.autoencoder_path)
    save_autoencoder(ae, ckpt_path)

    artifact = wandb.Artifact(name=run_name, type="autoencoder")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
