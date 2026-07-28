"""Entry point for CSPN training."""

import torch
from rtpt import RTPT
from torch.profiler import ProfilerActivity, profile, record_function
from torchinfo import summary

from dataset_loaders import build_data_loaders
from models.autoencoder.utils import load_pretrained_autoencoder
from models.cspn.psinet_cspn import PsiNetCSPN
from training.cspn_trainer import train_cspn
from training.objectives.cspn import CSPNObjective
from utils.checkpoints import load_ae_from_path, load_from_wandb
from utils.config import CSPNRunConfig, CSPNType, load_config
from utils.reproducibility import resolve_device, seed_everything


def main() -> None:
    cfg, cfg_seed = load_config()
    assert isinstance(cfg, CSPNRunConfig)
    dataset_cfg = cfg.dataset
    cspn_cfg = cfg.model
    assert cspn_cfg is not None
    training_cfg = cfg.training
    training_cfg.epochs = 1

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name

    run_name = f"cspn_{dataset_name}_{cspn_cfg.model_type}"

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

    objective = CSPNObjective(
        model=cspn.to(device),
        autoencoder=ae.to(device),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof,
        record_function("train_cspn"),
    ):
        train_cspn(
            objective=objective,
            device=device,
            cfg=cfg,
            train_loader=train_loader,
            test_loader=test_loader,
            rtpt=rtpt,
        )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
