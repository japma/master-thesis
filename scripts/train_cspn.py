"""Entry point for CSPN training."""

from models.autoencoder import AutoencoderType
from pathlib import Path

import torch
import wandb
from rtpt import RTPT

from models.autoencoder.utils import load_pretrained_autoencoder
from utils.checkpoints import save_cspn
from utils.config import load_config
from utils.reproducibility import seed_everything, resolve_device
from models.cspn.einet import Einet
from dataset_loaders import build_data_loaders
from training.train_cspn import train_cspn


def main():
    cfg = load_config()

    seed = seed_everything(cfg.seed)
    device = resolve_device()

    dataset_name = cfg.dataset.name
    run_name = f"CSPN_{dataset_name}_seed{seed}"

    print(f"Training CSPN on {dataset_name} | device={device} | seed={seed}")

    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

    # TODO fix loading
    ae = load_pretrained_autoencoder("test", AutoencoderType.VARIATIONAL)

    cspn = Einet(
        num_vars=cfg.dataset.latent_dim,
        context_dim=cfg.dataset.num_classes,
        num_leaves=cfg.cspn.num_leaves,
        num_nodes=cfg.cspn.num_nodes,
        nn_hidden_dim=cfg.cspn.nn_hidden_dim,
        nn_num_hidden_layers=cfg.cspn.nn_num_hidden_layers,
    )

    train_loader, test_loader = build_data_loaders(
        cfg.dataset, batch_size=cfg.training.batch_size
    )

    optimizer = torch.optim.Adam(cspn.parameters(), lr=cfg.training.learning_rate)

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
            "model": "CSPN",
            "epochs": cfg.training.epochs,
            "latent_dim": cfg.dataset.latent_dim,
            "learning_rate": cfg.training.learning_rate,
            "seed": seed,
            "num_leaves": cfg.cspn.num_leaves,
            "num_nodes": cfg.cspn.num_nodes,
            "nn_hidden_dim": cfg.cspn.nn_hidden_dim,
        },
        mode=cfg.wandb,
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

    ckpt_path = Path(f"checkpoints/{dataset_name}/cspn.pt")
    save_cspn(cspn, ckpt_path)

    artifact = wandb.Artifact(name=run_name, type="cspn")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
