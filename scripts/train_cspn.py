"""Entry point for CSPN training."""

from models.autoencoder.utils import load_pretrained_autoencoder
from torchinfo import summary

from pathlib import Path

import torch
import wandb
from rtpt import RTPT

from utils.checkpoints import save_cspn
from utils.config import load_config
from utils.reproducibility import seed_everything, resolve_device
from models.cspn.einet import Einet
from dataset_loaders import build_data_loaders
from training.train_cspn import train_cspn


def main():
    cfg = load_config()
    dataset_cfg = cfg.dataset
    cspn_cfg = cfg.cspn
    assert cspn_cfg is not None
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg.seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name
    run_name = f"CSPN_{dataset_name}_seed{seed}"

    print(f"Training CSPN on {dataset_name} | device={device} | seed={seed}")

    # TODO fix loading
    ae = load_pretrained_autoencoder("madebyollin/taesd")

    cspn = Einet(
        num_vars=32,
        context_dim=cfg.dataset.num_classes,
        num_leaves=cspn_cfg.num_leaves,
        num_nodes=cspn_cfg.num_nodes,
        nn_hidden_dim=cspn_cfg.nn_hidden_dim,
        nn_num_hidden_layers=cspn_cfg.nn_num_hidden_layers,
    )
    print("CSPN architecture:")
    summary(cspn)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer = torch.optim.Adam(cspn.parameters(), lr=training_cfg.learning_rate)

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=run_name,
        config={
            "dataset": dataset_name,
            "model": "CSPN",
            "model_type": "Einet",
            "epochs": training_cfg.epochs,
            "latent_dim": ae.get_latent_dim(),
            "learning_rate": training_cfg.learning_rate,
            "seed": seed,
            "num_leaves": cspn_cfg.num_leaves,
            "num_nodes": cspn_cfg.num_nodes,
            "nn_hidden_dim": cspn_cfg.nn_hidden_dim,
            "nn_num_hidden_layers": cspn_cfg.nn_num_hidden_layers,
        },
        mode=wandb_cfg.mode,
    )

    train_cspn(
        model=cspn,
        autoencoder=ae,
        device=device,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        rtpt=rtpt,
    )

    ckpt_path = Path(cfg.paths.cspn_path)
    save_cspn(cspn, ckpt_path)

    artifact = wandb.Artifact(name=run_name, type="cspn")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
