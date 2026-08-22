from pathlib import Path

import torch

import wandb
from utils.config import WandbConfig

ENTITY = "jmartini-tu-darmstadt"
PROJECT = "master-thesis"


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


def log_scalar_metrics(
    avg_train_loss: dict[str, float | torch.Tensor],
    avg_val_loss: dict[str, float | torch.Tensor],
    step: int,
) -> None:
    metrics = {f"train/{key}": value for key, value in avg_train_loss.items()}
    metrics.update({f"val/{key}": value for key, value in avg_val_loss.items()})
    wandb.log(metrics, step=step)


def log_images(key: str, images: torch.Tensor, step: int) -> None:
    images_u8 = (images.clamp(0, 1) * 255).byte().cpu()
    wandb.log({key: [wandb.Image(img) for img in images_u8]}, step=step)


def load_from_wandb(ckpt_name: str, tag: str = "latest") -> Path:
    """Load a checkpoint from wandb artifacts. Uses the most recent checkpoint unless tag is provided.

    Works with or without an active run (no `wandb.init()` needed for standalone inference/notebook
    use). If a run is actually tracking to the server, uses `run.use_artifact` so the run's lineage
    records which artifact it consumed; otherwise (no run, or a disabled/offline run such as
    `--dry-run`, whose `use_artifact` is a no-op returning None) falls back to the plain
    `wandb.Api()`. Either way `.download()` (rather than the old `.file()`) lands in a
    version-qualified directory, so downloading a different version no longer overwrites an
    already-downloaded one on disk.
    """
    print(f"Loading checkpoint {ckpt_name}:{tag} from Weights & Biases artifacts...")
    name = f"{ENTITY}/{PROJECT}/{ckpt_name}:{tag}"
    run = wandb.run
    tracking = run is not None and not run.disabled and not run.offline
    artifact = run.use_artifact(name) if tracking else wandb.Api().artifact(name)
    download_dir = Path(artifact.download())
    files = [f for f in download_dir.rglob("*") if f.is_file()]
    assert len(files) == 1, (
        f"expected exactly one file in artifact {name}, found {len(files)}: {files}"
    )
    file = files[0]
    print(f"Loaded {file} from Weights & Biases artifact {name}")
    return file
