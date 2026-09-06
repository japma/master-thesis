"""Entry point for neural baseline training."""

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models.autoencoder.pretrained import PretrainedVAE
from models.neural_baseline import build_neural_baseline
from training.early_stopping import EarlyStopping
from training.loop import CheckpointSpec, run_training_loop
from training.objectives.nn_baseline import NeuralBaselineObjective
from utils.checkpoints import (
    final_checkpoint_path,
    intermediate_checkpoint_path,
    load_ae_from_path,
    load_nn_baseline_from_path,
)
from utils.compilation import maybe_compile
from utils.config import NeuralBaselineRunConfig, load_config
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, load_from_wandb


def main() -> None:
    cfg, cfg_seed, resume = load_config()
    assert isinstance(cfg, NeuralBaselineRunConfig)
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    training_cfg = cfg.training

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name
    variant = str(model_cfg.model_type)

    run_name = f"nn_baseline_{dataset_name}_{variant}"
    init_run(cfg.wandb, run_name, cfg.model_dump())

    ae_cfg = cfg.autoencoder
    if ae_cfg.external:
        ae = PretrainedVAE(
            name=ae_cfg.name, height=dataset_cfg.height, width=dataset_cfg.width
        )
    else:
        ae = load_ae_from_path(
            load_from_wandb(ckpt_name=ae_cfg.name, tag=ae_cfg.tag), device=device
        )
    ae = ae.to(device)

    if ae.get_latent_dim().numel() != model_cfg.num_vars:
        raise ValueError(
            f"AE latent dim {ae.get_latent_dim()} ({ae.get_latent_dim().numel()}) does "
            f"not match baseline num_vars {model_cfg.num_vars}"
        )

    print(f"Training neural baseline on {dataset_name} | device={device} | seed={seed}")

    ckpt_path = intermediate_checkpoint_path("nn_baseline", dataset_name, variant)
    if resume and ckpt_path.exists():
        model = load_nn_baseline_from_path(ckpt_path, device=device).to(device)
        print(f"Resumed model weights from {ckpt_path}")
    else:
        if resume:
            print(
                f"--resume given but no checkpoint found at {ckpt_path}; "
                "starting from scratch"
            )
        model = build_neural_baseline(model_cfg).to(device)

    print("Neural baseline architecture:")
    summary(model)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    model = maybe_compile(model, training_cfg.compile, training_cfg.compile_mode)

    objective = NeuralBaselineObjective(
        model=model,
        autoencoder=ae,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    early_stopping = EarlyStopping(
        patience=training_cfg.early_stopping_patience,
        min_delta=training_cfg.early_stopping_min_delta,
    )
    print(
        f"Early stopping on val total: patience "
        f"{training_cfg.early_stopping_patience}, min_delta "
        f"{training_cfg.early_stopping_min_delta}"
    )

    checkpoint = CheckpointSpec(
        intermediate_path=ckpt_path,
        final_path=final_checkpoint_path("nn_baseline", dataset_name, variant),
        artifact_type="nn_baseline",
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
        sample_probe=_sample_labels(cfg, device),
        sample_log_key="samples/nn_baseline_generated_images",
        early_stopping=early_stopping,
    )

    wandb.finish()


def _sample_labels(cfg: NeuralBaselineRunConfig, device: torch.device) -> torch.Tensor:
    return torch.tensor([[digit, 0, 0] for digit in range(10)]).to(
        device, non_blocking=True
    )


if __name__ == "__main__":
    main()
