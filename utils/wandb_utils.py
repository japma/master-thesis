from pathlib import Path

import torch

import wandb
from utils.config import WANDB_ENTITY as ENTITY
from utils.config import WANDB_PROJECT as PROJECT
from utils.config import WandbConfig


def init_run(wandb_cfg: WandbConfig, run_name: str, config: dict) -> None:
    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=run_name,
        config=config,
        mode=wandb_cfg.mode,
    )


def log_checkpoint_artifact(
    path: Path, name: str, type: str, description: str | None = None
) -> None:
    artifact = wandb.Artifact(name=name, type=type, description=description)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def log_metrics(metrics: dict[str, float | torch.Tensor], step: int) -> None:
    """No-ops without an active run, so an objective that logs can still be driven
    outside a training script -- from a test, or a notebook."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_summary(summary: dict[str, float]) -> None:
    """Run-level values that do not belong on a per-epoch curve."""
    if wandb.run is not None:
        wandb.run.summary.update(summary)


def log_scalar_metrics(
    avg_train_loss: dict[str, float | torch.Tensor],
    avg_val_loss: dict[str, float | torch.Tensor],
    step: int,
) -> None:
    metrics = {f"train/{key}": value for key, value in avg_train_loss.items()}
    metrics.update({f"val/{key}": value for key, value in avg_val_loss.items()})
    log_metrics(metrics, step=step)


def log_images(key: str, images: torch.Tensor, step: int) -> None:
    images_u8 = (images.clamp(0, 1) * 255).byte().cpu()
    wandb.log({key: [wandb.Image(img) for img in images_u8]}, step=step)


def _qualified(ckpt_name: str, tag: str) -> str:
    """Full artifact path. A `ckpt_name` that already names a version (`foo:v3`) keeps
    it, so a reference resolved from lineage survives a round trip through here."""
    name = ckpt_name if ":" in ckpt_name else f"{ckpt_name}:{tag}"
    return f"{ENTITY}/{PROJECT}/{name}"


def artifact_ref(artifact) -> str:
    """`name:version` -- the version-pinned reference, never `:latest`."""
    name = str(artifact.name)
    return name if ":" in name else f"{name}:{artifact.version}"


def download_artifact(ckpt_name: str, tag: str = "latest") -> tuple[Path, str]:
    """The downloaded checkpoint file and the exact `name:version` it came from.

    Works with or without an active run (no `wandb.init()` needed for standalone inference/notebook
    use). If a run is actually tracking to the server, uses `run.use_artifact` so the run's lineage
    records which artifact it consumed; otherwise (no run, or a disabled/offline run such as
    `--dry-run`, whose `use_artifact` is a no-op returning None) falls back to the plain
    `wandb.Api()`. Either way `.download()` (rather than the old `.file()`) lands in a
    version-qualified directory, so downloading a different version no longer overwrites an
    already-downloaded one on disk.
    """
    print(f"Loading checkpoint {ckpt_name}:{tag} from Weights & Biases artifacts...")
    name = _qualified(ckpt_name, tag)
    run = wandb.run
    if run is not None and not run.disabled and not run.offline:
        artifact = run.use_artifact(name)
    else:
        artifact = wandb.Api().artifact(name)
    download_dir = Path(artifact.download())
    files = [f for f in download_dir.rglob("*") if f.is_file()]
    assert len(files) == 1, (
        f"expected exactly one file in artifact {name}, found {len(files)}: {files}"
    )
    file = files[0]
    print(f"Loaded {file} from Weights & Biases artifact {name}")
    return file, artifact_ref(artifact)


def load_from_wandb(ckpt_name: str, tag: str = "latest") -> Path:
    """Load a checkpoint from wandb artifacts. Uses the most recent checkpoint unless tag is provided."""
    return download_artifact(ckpt_name, tag)[0]


def trained_with(
    ckpt_name: str, tag: str = "latest", artifact_type: str = "autoencoder"
) -> str | None:
    """The `name:version` of the `artifact_type` artifact that `ckpt_name:tag` was
    trained against, or None if it cannot be established.

    A training run consumes its autoencoder through `download_artifact`, which records
    the dependency via `run.use_artifact`, so the run that logged a PC checkpoint knows
    exactly which autoencoder version produced its latents. Falls back to the run's
    stored config, which names the autoencoder but not the version it resolved to --
    the caller is told which of the two answered.
    """
    try:
        artifact = wandb.Api().artifact(_qualified(ckpt_name, tag))
        run = artifact.logged_by()
    except Exception as error:  # offline, deleted run, no such artifact
        print(f"Could not reach the wandb lineage for {ckpt_name}:{tag}: {error}")
        return None

    if run is None:
        print(f"Artifact {ckpt_name}:{tag} has no run to trace its inputs to")
        return None

    used = [a for a in run.used_artifacts() if a.type == artifact_type]
    if len(used) > 1:
        print(
            f"Run {run.name} consumed {len(used)} {artifact_type} artifacts "
            f"({', '.join(artifact_ref(a) for a in used)}); taking the first"
        )
    if used:
        ref = artifact_ref(used[0])
        print(f"{ckpt_name}:{tag} was trained with {ref} (wandb lineage)")
        return ref

    # Runs from before the dependency was recorded, and the `external: true`
    # autoencoders that never become artifacts at all.
    configured = run.config.get(artifact_type, {})
    name = configured.get("name") if isinstance(configured, dict) else None
    if name is None:
        print(f"Run {run.name} records no {artifact_type} input or config entry")
        return None
    print(
        f"{ckpt_name}:{tag} names {name} in its run config, but did not record which "
        "version it used -- falling back to the configured tag"
    )
    return str(name)
