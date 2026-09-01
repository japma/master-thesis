"""Entry point for CSPN training."""

from pathlib import Path

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from dataset_loaders.latent_normalizer import LatentNormalizer
from models.autoencoder.pretrained import PretrainedVAE
from models.cspn.psinet.label_pc import LabelPC
from models.cspn.psinet_cspn import PsiNetCSPN
from training.loop import CheckpointSpec, run_training_loop
from training.objectives.cspn import CSPNObjective
from utils.checkpoints import (
    final_checkpoint_path,
    intermediate_checkpoint_path,
    label_pc_checkpoint_path,
    load_ae_from_path,
    load_cspn_from_path,
    load_label_pc_from_path,
)
from utils.compilation import maybe_compile
from utils.config import (
    ConditioningType,
    CSPNEncoderType,
    CSPNRunConfig,
    CSPNType,
    load_config,
)
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, load_from_wandb


def _themed_multi_binary_labels(
    themes: list[dict[int, float]],
    num_attributes: int,
    device: torch.device,
    label_pc: LabelPC | None,
) -> torch.Tensor:
    """Builds one label vector per theme (a sparse {attribute_idx: value} spec)."""
    if label_pc is not None:
        return torch.cat(
            [
                label_pc.complete_partial(known, batch_size=1, device=device)
                for known in themes
            ],
            dim=0,
        )

    print(
        "No LabelPC available -- falling back to zero-filled (attribute=off) labels "
        "for sample logging."
    )
    vectors = []
    for known in themes:
        vector = torch.zeros(num_attributes)
        for idx, value in known.items():
            vector[idx] = value
        vectors.append(vector)
    return torch.stack(vectors).to(device, non_blocking=True)


def _load_label_pc(
    cfg: CSPNRunConfig, dataset_name: str, device: torch.device
) -> LabelPC | None:
    """Load the pretrained LabelPC that completes unspecified attributes when logging
    samples.

    Tries the wandb artifact first — the autoencoder is loaded the same way, and the
    checkpoint generally lives on whichever machine trained it rather than the one
    running this — then falls back to a local checkpoint. Missing is not fatal: the
    LabelPC only decorates sample logging, so training proceeds with zero-filled
    attribute vectors instead.
    """
    if cfg.model.encoder_config.encoder_type != CSPNEncoderType.MULTI_BINARY:
        return None

    local_path = label_pc_checkpoint_path(dataset_name)
    # run_training_loop logs every checkpoint under its filename stem, which is what
    # resolve_name reproduces.
    artifact_name = cfg.label_pc.resolve_name(dataset_name)
    tag = cfg.label_pc.tag

    path: Path | None = None
    try:
        path = load_from_wandb(ckpt_name=artifact_name, tag=tag)
    except Exception as error:
        print(f"Could not load LabelPC artifact {artifact_name}:{tag} from wandb ({error})")
        if local_path.exists():
            path = local_path

    if path is None:
        print(
            f"No LabelPC found (wandb {artifact_name}:{tag}, local {local_path}); "
            "sample logging will fall back to zero-filled attributes. Run "
            "scripts/train_label_pc.py first to enable attribute-completion sampling."
        )
        return None

    label_pc = load_label_pc_from_path(path, device=device)
    label_pc.eval()
    print(f"Loaded LabelPC from {path} for attribute-completion sampling")
    return label_pc


def _build_sample_labels(
    cfg: CSPNRunConfig,
    device: torch.device,
    label_pc: LabelPC | None,
) -> torch.Tensor:
    sample_count = max(1, min(16, cfg.dataset.num_classes))

    if cfg.model.encoder_config.encoder_type == CSPNEncoderType.CATEGORICAL:
        return (
            torch.arange(sample_count)
            .repeat_interleave(3)
            .to(device, non_blocking=True)
        )
    elif cfg.model.encoder_config.encoder_type == CSPNEncoderType.MULTI_BINARY:
        glasses_idx = 15
        male_idx = 20
        bald_idx = 4

        # {} = fully unconditional (all attributes marginalized/sampled by LabelPC,
        # or zero-filled in the no-LabelPC fallback).
        themes: list[dict[int, float]] = [
            {},
            {glasses_idx: 1.0},
            {male_idx: 1.0},
            {bald_idx: 1.0},
        ]
        return _themed_multi_binary_labels(
            themes, cfg.dataset.num_classes, device, label_pc
        )
    else:
        # TODO update the hardcoded colourmnist values
        return torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 0, 0],
                [6, 0, 0],
                [7, 0, 0],
                [8, 0, 0],
                [9, 0, 0],
            ]
        ).to(device, non_blocking=True)


def main() -> None:
    cfg, cfg_seed, resume = load_config()
    assert isinstance(cfg, CSPNRunConfig)
    dataset_cfg = cfg.dataset
    cspn_cfg = cfg.model
    assert cspn_cfg is not None
    training_cfg = cfg.training
    wandb_cfg = cfg.wandb

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name

    # Empty for the default joint hypernetwork, so existing checkpoint paths and
    # wandb run names are unchanged; a factorized run gets its own so it does not
    # overwrite the baseline it is being ablated against.
    variant = (
        ""
        if cspn_cfg.conditioning_type is ConditioningType.JOINT
        else str(cspn_cfg.conditioning_type)
    )

    run_name = f"cspn_{dataset_name}_{cspn_cfg.model_type}" + (
        f"_{variant}" if variant else ""
    )

    init_run(wandb_cfg, run_name, cfg.model_dump())

    ae_cfg = cfg.autoencoder
    if ae_cfg.external:
        ae = PretrainedVAE(
            name=ae_cfg.name, height=dataset_cfg.height, width=dataset_cfg.width
        )
    else:
        ae_path = load_from_wandb(ckpt_name=ae_cfg.name, tag=ae_cfg.tag)
        ae = load_ae_from_path(ae_path, device=device)
    ae = ae.to(device)

    if ae.get_latent_dim().numel() != cspn_cfg.num_vars:
        raise ValueError(
            f"AE latent dim {ae.get_latent_dim()} ({ae.get_latent_dim().numel()}) does not match CSPN num_vars {cspn_cfg.num_vars}"
        )

    print(f"Training CSPN on {dataset_name} | device={device} | seed={seed}")

    if cspn_cfg.model_type != CSPNType.PSINET:
        raise ValueError(f"Unknown model type {cspn_cfg.model_type}")

    cspn_ckpt_path = intermediate_checkpoint_path(
        cspn_cfg.model_type, dataset_cfg.name, variant
    )
    resumed_cspn = False
    if resume and cspn_ckpt_path.exists():
        cspn = load_cspn_from_path(cspn_ckpt_path, device=device).to(device)
        resumed_cspn = True
        print(f"Resumed model weights from {cspn_ckpt_path}")
    else:
        if resume:
            print(
                f"--resume given but no checkpoint found at {cspn_ckpt_path}; "
                "starting from scratch"
            )
        cspn = PsiNetCSPN(config=cspn_cfg).to(device)
    assert isinstance(cspn, PsiNetCSPN)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size
    )

    # Skipped when resuming: latent_mean/latent_std were already restored from the
    # checkpoint's state dict above, and re-fitting would just redundantly recompute
    # the same (deterministic) statistics from the same train loader.
    if cspn_cfg.normalize_latents and not resumed_cspn:
        normalizer = LatentNormalizer()
        normalizer.fit(ae, train_loader, device)
        assert normalizer.mean is not None
        assert normalizer.std is not None
        cspn.set_latent_stats(normalizer.mean, normalizer.std)

    label_pc = _load_label_pc(cfg, dataset_name, device)

    print("CSPN architecture:")
    summary(cspn)

    optimizer = torch.optim.Adam(cspn.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    cspn = maybe_compile(cspn, training_cfg.compile, training_cfg.compile_mode)

    objective = CSPNObjective(
        model=cspn,
        autoencoder=ae,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        label_pc=label_pc,
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    sample_labels = _build_sample_labels(cfg, device, label_pc)

    checkpoint = CheckpointSpec(
        intermediate_path=cspn_ckpt_path,
        final_path=final_checkpoint_path(cspn_cfg.model_type, dataset_name, variant),
        artifact_type="cspn",
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
        sample_log_key="samples/cspn_generated_images",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
