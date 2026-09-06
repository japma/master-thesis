"""Entry point for joint latent+label PC training."""

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from dataset_loaders.latent_normalizer import LatentNormalizer
from models.autoencoder.pretrained import PretrainedVAE
from models.cspn.joint_pc import JointPC
from training.loop import CheckpointSpec, run_training_loop
from training.objectives.joint_pc import JointPCObjective
from utils.checkpoints import (
    final_checkpoint_path,
    intermediate_checkpoint_path,
    load_ae_from_path,
    load_joint_pc_from_path,
)
from utils.compilation import maybe_compile
from utils.config import JointPCRunConfig, load_config
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, load_from_wandb


def _build_sample_labels(
    model: JointPC, device: torch.device, per_factor: int = 16
) -> torch.Tensor:
    """One label vector per value of the first factor, remaining factors at 0."""
    count = min(per_factor, model.label_cardinalities[0])
    labels = torch.zeros(count, len(model.label_cardinalities), dtype=torch.long)
    labels[:, 0] = torch.arange(count)
    return labels.to(device, non_blocking=True)


def main() -> None:
    cfg, cfg_seed, resume = load_config()
    assert isinstance(cfg, JointPCRunConfig)
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    training_cfg = cfg.training

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name
    run_name = f"joint_pc_{dataset_name}"

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

    if ae.get_latent_dim().numel() != model_cfg.num_latents:
        raise ValueError(
            f"AE latent dim {ae.get_latent_dim()} "
            f"({ae.get_latent_dim().numel()}) does not match JointPC num_latents "
            f"{model_cfg.num_latents}"
        )

    print(
        f"Training JointPC on {dataset_name} | {model_cfg.num_latents} latents + "
        f"{model_cfg.label_cardinalities} labels | device={device} | seed={seed}"
    )

    ckpt_path = intermediate_checkpoint_path("joint_pc", dataset_name)
    resumed = False
    if resume and ckpt_path.exists():
        model = load_joint_pc_from_path(ckpt_path, device=device)
        resumed = True
        print(f"Resumed model weights from {ckpt_path}")
    else:
        if resume:
            print(
                f"--resume given but no checkpoint found at {ckpt_path}; "
                "starting from scratch"
            )
        model = JointPC(config=model_cfg)
    model = model.to(device)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    if model_cfg.normalize_latents and not resumed:
        normalizer = LatentNormalizer()
        normalizer.fit(ae, train_loader, device)
        assert normalizer.mean is not None
        assert normalizer.std is not None
        model.set_latent_stats(normalizer.mean, normalizer.std)

    print("JointPC architecture:")
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    sample_labels = _build_sample_labels(model, device)
    model = maybe_compile(model, training_cfg.compile, training_cfg.compile_mode)

    objective = JointPCObjective(
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

    checkpoint = CheckpointSpec(
        intermediate_path=ckpt_path,
        final_path=final_checkpoint_path("joint_pc", dataset_name),
        artifact_type="joint_pc",
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
        sample_probe=sample_labels,
        sample_log_key="samples/joint_pc_generated_images",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
