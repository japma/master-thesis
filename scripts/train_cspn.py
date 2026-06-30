"""Entry point for CSPN training."""

from pathlib import Path

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models.autoencoder.utils import load_pretrained_autoencoder
from models.cspn.psinet_cspn import PsiNetCSPN
from training.cspn_trainer import train_cspn
from utils.checkpoints import load_ae_from_path, load_from_wandb, save_cspn
from utils.config import CSPNRunConfig, CSPNType, load_config
from utils.reproducibility import resolve_device, seed_everything


def main():
    cfg, cfg_seed = load_config()
    assert isinstance(cfg, CSPNRunConfig)
    dataset_cfg = cfg.dataset
    cspn_cfg = cfg.model
    assert cspn_cfg is not None
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name

    run_name = f"cspn_{dataset_name}_{cspn_cfg.model_type}"

    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=run_name,
        config={
            "dataset": dataset_name,
            "model": "CSPN",
            "model_type": "Einet",
            "epochs": training_cfg.epochs,
            # "latent_dim": ae.get_latent_dim(), # commented out because it is not known and only used for logging
            "learning_rate": training_cfg.learning_rate,
            "seed": seed,
            # "num_leaves": cspn_cfg.num_leaves,
            "num_sums": cspn_cfg.num_sums,
            # "depth": cspn_cfg.depth,
            "num_repetitions": cspn_cfg.num_repetitions,
        },
        mode=wandb_cfg.mode,
    )

    ae_cfg = cfg.autoencoder
    if ae_cfg.external:
        ae = load_pretrained_autoencoder(ae_cfg.name)
    else:
        ae_path = load_from_wandb(ckpt_name=ae_cfg.name, tag="best")
        ae = load_ae_from_path(ae_path, device=device)

    if ae.get_latent_dim() != ae_cfg.latent_dim:
        raise ValueError(
            f"Latent dimension of autoencoder checkpoint ({ae.get_latent_dim()}) does not match the expected latent dimension ({ae_cfg.latent_dim})"
        )

    print(f"Training CSPN on {dataset_name} | device={device} | seed={seed}")

    if cspn_cfg.model_type == CSPNType.PSINET:
        cspn = PsiNetCSPN(config=cspn_cfg)
    else:
        raise ValueError(f"Unknown model type {cspn_cfg.model_type}")

    print("CSPN architecture:")
    summary(cspn)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer = torch.optim.Adam(cspn.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    train_cspn(
        model=cspn,
        autoencoder=ae,
        device=device,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        rtpt=rtpt,
    )

    ckpt_path = Path("checkpoints") / f"{run_name}.pt"
    save_cspn(cspn, ckpt_path)

    artifact = wandb.Artifact(name=run_name, type="cspn")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
