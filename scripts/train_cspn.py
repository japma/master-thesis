"""Entry point for CSPN training."""

from sympy import true

from models import SPFlowCSPN
from models.autoencoder import AutoencoderType
from models.cspn.abstract_cspn import CSPNType
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.checkpoints import load_ae_from_path, load_from_wandb
from models.autoencoder.utils import load_pretrained_autoencoder
from torchinfo import summary

from pathlib import Path

import torch
import wandb
from rtpt import RTPT

from utils.checkpoints import save_cspn
from utils.config import load_config, CSPNConfig, CSPNRunConfig
from utils.reproducibility import seed_everything, resolve_device
from models.cspn.CustomEinet.einet import Einet
from dataset_loaders import build_data_loaders
from training.train_cspn import train_cspn


def main():
    cfg = load_config()
    assert isinstance(cfg, CSPNRunConfig)
    dataset_cfg = cfg.dataset
    cspn_cfg = cfg.model
    assert cspn_cfg is not None
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg.seed)
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
            "num_leaves": cspn_cfg.num_leaves,
            "num_sums": cspn_cfg.num_sums,
            "depth": cspn_cfg.depth,
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

    print(f"Training CSPN on {dataset_name} | device={device} | seed={seed}")

    if cspn_cfg.model_type == CSPNType.SPFLOW:
        cspn = SPFlowCSPN(
            latent_dim=ae.get_latent_dim(),
            num_classes=dataset_cfg.num_classes,
            num_sums=cspn_cfg.num_sums,
            num_leaves=cspn_cfg.num_leaves,
            depth=cspn_cfg.depth,
            num_repetitions=5,
            nn_layers=cspn_cfg.nn_num_hidden_layers,
            nn_hidden_dim=cspn_cfg.nn_hidden_dim,
        )
    elif cspn_cfg.model_type == CSPNType.CUSTOM:
        cspn = Einet(
            num_vars=ae.get_latent_dim(),
            context_dim=dataset_cfg.num_classes,
            num_leaves=cspn_cfg.num_leaves,
            num_nodes=cspn_cfg.num_sums,
            nn_hidden_dim=cspn_cfg.nn_hidden_dim,
            nn_num_hidden_layers=cspn_cfg.nn_num_hidden_layers,
        )
    elif cspn_cfg.model_type == CSPNType.PSINET:
        cspn = PsiNetCSPN(
            latent_dim=ae.get_latent_dim(),
            num_classes=dataset_cfg.num_classes,
        )
    else:
        raise ValueError(f"Unknown model type {cspn_cfg.model_type}")

    print("CSPN architecture:")
    summary(cspn)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size, homogeneous=true
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
