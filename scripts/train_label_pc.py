"""Entry point for LabelPC training."""

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models.cspn.psinet.label_pc import LabelPC
from training.loop import CheckpointSpec, run_training_loop
from training.objectives.label_pc_objective import LabelPCObjective
from utils.checkpoints import (
    intermediate_checkpoint_path,
    label_pc_checkpoint_path,
    load_label_pc_from_path,
)
from utils.config import CSPNRunConfig, load_config
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run


def main() -> None:
    cfg, cfg_seed, resume = load_config()
    assert isinstance(cfg, CSPNRunConfig)
    dataset_cfg = cfg.dataset
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name

    label_pc_cfg = cfg.label_pc
    # Derived, not label_pc.name: that field points at an artifact to *load*, while this
    # run *produces* one, and its checkpoint stem is what becomes the artifact name.
    run_name = f"label_pc_{dataset_name}"

    init_run(wandb_cfg, run_name, cfg.model_dump())

    # Raises for label spaces the LabelPC cannot represent (anything non-binary),
    # rather than failing later on a shape mismatch inside the circuit.
    num_attributes: int = label_pc_cfg.resolve_num_attributes(
        cfg.model.encoder_config
    )

    print(
        f"Training LabelPC on {dataset_name} | {num_attributes} attributes | "
        f"device={device} | seed={seed}"
    )

    label_pc_ckpt_path = intermediate_checkpoint_path("label_pc", dataset_name)
    if resume and label_pc_ckpt_path.exists():
        label_pc = load_label_pc_from_path(label_pc_ckpt_path, device=device)
        print(f"Resumed model weights from {label_pc_ckpt_path}")
    else:
        if resume:
            print(
                f"--resume given but no checkpoint found at {label_pc_ckpt_path}; "
                "starting from scratch"
            )
        label_pc = LabelPC(
            num_attributes=num_attributes,
            num_input_distributions=label_pc_cfg.num_input_distributions,
            num_sums=label_pc_cfg.num_sums,
            num_repetitions=label_pc_cfg.num_repetitions,
        )
    label_pc = label_pc.to(device)

    print("LabelPC architecture:")
    summary(label_pc)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    optimizer = torch.optim.Adam(label_pc.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    objective = LabelPCObjective(
        model=label_pc,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    checkpoint = CheckpointSpec(
        intermediate_path=label_pc_ckpt_path,
        final_path=label_pc_checkpoint_path(dataset_name),
        artifact_type="label_pc",
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
        needs_images=False,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
