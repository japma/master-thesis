"""Entry point for LabelPC training."""

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models.cspn.psinet.label_pc import LabelPC
from training.label_pc_trainer import train_label_pc
from training.objectives.label_pc_objective import LabelPCObjective
from utils.checkpoints import label_pc_checkpoint_path, save_label_pc
from utils.config import CSPNRunConfig, load_config
from utils.reproducibility import resolve_device, seed_everything


def main() -> None:
    cfg, cfg_seed, _resume = load_config()
    assert isinstance(cfg, CSPNRunConfig)
    dataset_cfg = cfg.dataset
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name

    run_name = f"label_pc_{dataset_name}"

    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=run_name,
        config=cfg.model_dump(),
        mode=wandb_cfg.mode,
    )

    # TODO: hardcoded until LabelPC gets its own config section
    NUM_INPUT_DISTRIBUTIONS: int = 10
    NUM_SUMS: int = 10
    NUM_REPETITIONS: int = 5

    # TODO: assumes dataset_cfg.num_classes holds the attribute count for
    # multi-binary label datasets (CelebA)
    num_attributes: int = dataset_cfg.num_classes

    print(f"Training LabelPC on {dataset_name} | device={device} | seed={seed}")

    label_pc = LabelPC(
        num_attributes=num_attributes,
        num_input_distributions=NUM_INPUT_DISTRIBUTIONS,
        num_sums=NUM_SUMS,
        num_repetitions=NUM_REPETITIONS,
    )

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
        model=label_pc.to(device),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    train_label_pc(
        objective=objective,
        device=device,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        rtpt=rtpt,
    )

    ckpt_path = label_pc_checkpoint_path(dataset_name)
    save_label_pc(label_pc, ckpt_path)

    artifact = wandb.Artifact(name=run_name, type="label_pc")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
