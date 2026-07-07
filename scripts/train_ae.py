"""Entry point for autoencoder training."""

from pathlib import Path

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models import VariationalAutoencoder
from training.ae_trainer import train_autoencoder
from training.losses import BetaVAELoss, VAELoss, BetaTCVAELoss
from utils.checkpoints import save_autoencoder
from utils.config import AERunConfig, load_config
from utils.reproducibility import resolve_device, seed_everything


def main() -> None:
    cfg, cfg_seed = load_config()
    assert isinstance(cfg, AERunConfig)
    dataset_cfg = cfg.dataset
    autoencoder_cfg = cfg.model
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg_seed)
    beta = training_cfg.beta
    device = resolve_device()
    dataset_name = dataset_cfg.name
    model_name = f"autoencoder_{dataset_name}"
    run_name = f"{model_name}_beta{beta}"

    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=run_name,
        config=cfg.model_dump(),
        mode=wandb_cfg.mode,
    )

    print(f"Training Autoencoder on {dataset_name} | device={device} | seed={seed}")

    ae = VariationalAutoencoder(config=autoencoder_cfg).to(device)
    print("Autoencoder Architecture:")
    summary(ae)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer = torch.optim.Adam(ae.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    # loss_fn = VAELoss(
    #    beta_vae_loss=BetaVAELoss(beta=beta, free_bits=training_cfg.free_bits),
    #    lambda_perceptual=training_cfg.lambda_perceptual,
    # )

    loss_fn = BetaTCVAELoss(
        dataset_size=len(train_loader),
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        free_bits=training_cfg.free_bits,
    )

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

    ckpt_path = Path("checkpoints") / f"{run_name}.pt"
    save_autoencoder(ae, ckpt_path)

    artifact = wandb.Artifact(name=model_name, type="autoencoder")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
