"""Entry point for autoencoder training."""

from pathlib import Path

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models import VariationalAutoencoder
from training.ae_trainer import train_autoencoder
from training.losses import BetaVAELoss, PatchDiscriminator, VAELoss
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
        config={
            "dataset": dataset_name,
            "model": "Autoencoder",
            "model_type": "VariationalAutoencoder",
            "epochs": training_cfg.epochs,
            "latent_dim": autoencoder_cfg.latent_dim,
            "learning_rate": training_cfg.learning_rate,
            "beta": beta,
            "seed": seed,
            # "base_channels": autoencoder_cfg.base_channels,
            "num_blocks": autoencoder_cfg.num_blocks,
            "lambda_perceptual": training_cfg.lambda_perceptual,
            "lambda_adversarial": training_cfg.lambda_adversarial,
            "adversarial_warmup_steps": training_cfg.adversarial_warmup_steps,
        },
        mode=wandb_cfg.mode,
    )

    print(f"Training Autoencoder on {dataset_name} | device={device} | seed={seed}")

    ae = VariationalAutoencoder(config=autoencoder_cfg).to(device)
    print("Autoencoder Architecture:")
    summary(ae)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer_g = torch.optim.Adam(ae.parameters(), lr=training_cfg.learning_rate)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=training_cfg.epochs
    )

    discriminator = PatchDiscriminator().to(device)
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=training_cfg.learning_rate * 0.5,
        betas=(0.5, 0.999),  # standard GAN discriminator betas
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=training_cfg.epochs
    )

    loss_fn = VAELoss(
        beta_vae_loss=BetaVAELoss(beta=beta, free_bits=training_cfg.free_bits),
        discriminator=discriminator,
        lambda_perceptual=training_cfg.lambda_perceptual,
        lambda_adversarial=training_cfg.lambda_adversarial,
        adversarial_warmup_steps=training_cfg.adversarial_warmup_steps,
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
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
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
