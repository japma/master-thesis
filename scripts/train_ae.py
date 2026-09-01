"""Entry point for autoencoder training."""

from collections.abc import Sized

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models import VariationalAutoencoder
from training.loop import CheckpointSpec, run_training_loop
from training.losses.tcvae import BetaTCVAELoss
from training.losses.vae import VAELoss
from training.objectives.beta_vae import BetaVAEObjective
from training.objectives.tcvae import TCVAEObjective
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import (
    final_checkpoint_path,
    intermediate_checkpoint_path,
    load_ae_from_path,
)
from utils.compilation import maybe_compile
from utils.config import AERunConfig, VAETrainingType, load_config
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, log_images


def main() -> None:
    cfg, cfg_seed, resume = load_config()
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
    run_name = f"{model_name}_{training_cfg.vae_type}"

    init_run(wandb_cfg, run_name, cfg.model_dump())

    print(f"Training Autoencoder on {dataset_name} | device={device} | seed={seed}")

    ae_ckpt_path = intermediate_checkpoint_path(
        autoencoder_cfg.model_type, dataset_cfg.name
    )
    if resume and ae_ckpt_path.exists():
        ae = load_ae_from_path(ae_ckpt_path, device=device).to(device)
        print(f"Resumed model weights from {ae_ckpt_path}")
    else:
        if resume:
            print(
                f"--resume given but no checkpoint found at {ae_ckpt_path}; "
                "starting from scratch"
            )
        ae = VariationalAutoencoder(config=autoencoder_cfg).to(device)
    assert isinstance(ae, VariationalAutoencoder)
    print("Autoencoder Architecture:")
    summary(ae)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )
    test_dataset = test_loader.dataset
    assert isinstance(train_loader.dataset, Sized)
    assert isinstance(test_loader.dataset, Sized)

    test_data_size = len(test_loader.dataset)
    train_data_size = len(train_loader.dataset)

    optimizer = torch.optim.Adam(ae.parameters(), lr=training_cfg.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    ae = maybe_compile(ae, training_cfg.compile, training_cfg.compile_mode)

    if training_cfg.vae_type == VAETrainingType.BETA:
        beta_scheduler = BetaAnnealingScheduler(
            beta_start=training_cfg.beta_start,
            beta_end=training_cfg.beta_end,
            num_steps=len(train_loader) * training_cfg.kl_warmup_epochs,
        )

        loss_fn = VAELoss(
            beta=beta,
            free_bits=training_cfg.free_bits,
            lambda_perceptual=training_cfg.lambda_perceptual,
        ).to(device)

        objective = BetaVAEObjective(
            model=ae,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            beta_scheduler=beta_scheduler,
        )
    elif training_cfg.vae_type == VAETrainingType.TCVAE:
        assert training_cfg.tcvae_alpha is not None
        assert training_cfg.tcvae_beta is not None
        assert training_cfg.tcvae_gamma is not None

        tcvae_beta_scheduler = BetaAnnealingScheduler(
            beta_start=training_cfg.beta_start,
            beta_end=training_cfg.tcvae_beta,
            num_steps=len(train_loader) * training_cfg.kl_warmup_epochs,
        )

        loss_fn = BetaTCVAELoss(
            alpha=training_cfg.tcvae_alpha,
            beta=training_cfg.tcvae_beta,
            gamma=training_cfg.tcvae_gamma,
            free_bits=training_cfg.free_bits,
            lambda_perceptual=training_cfg.lambda_perceptual,
        ).to(device)
        objective = TCVAEObjective(
            model=ae,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            beta_scheduler=tcvae_beta_scheduler,
            train_data_size=train_data_size,
            test_data_size=test_data_size,
        )
    else:
        raise ValueError(f"Unknown training type: {training_cfg.vae_type}")

    sample_indices = torch.randperm(test_data_size)[: min(10, test_data_size)]
    sample_images = torch.stack(
        [test_dataset[i.item()][0] for i in sample_indices]
    ).to(device)
    log_images("samples/input", sample_images, step=0)

    checkpoint = CheckpointSpec(
        intermediate_path=ae_ckpt_path,
        final_path=final_checkpoint_path(autoencoder_cfg.model_type, dataset_name),
        artifact_type="autoencoder",
    )

    run_training_loop(
        objective=objective,
        device=device,
        epochs=training_cfg.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        rtpt=rtpt,
        checkpoint=checkpoint,
        resume=resume,
        sample_probe=sample_images,
        sample_log_key="samples/reconstructed",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
