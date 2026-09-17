"""A run directory: the on-disk contract between generation and evaluation.

<root>/<dataset>__<model_name>__seed<N>/
    manifest.json
    latents.pt    (N, D)
    labels.pt     (N, 3): digit, fg, bg of each latent
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import torch

MANIFEST_FILENAME = "manifest.json"
LATENTS_FILENAME = "latents.pt"
LABELS_FILENAME = "labels.pt"


@dataclass(frozen=True)
class RunManifest:
    model_name: str
    dataset: str
    seed: int
    n_samples: int
    # Exact wandb `name:vN` artifacts; the model's is None for real data.
    checkpoint_path: str | None
    vae_checkpoint: str
    git_commit: str | None
    std_correction: float | None
    sampling_seconds: float | None
    n_parameters: int | None

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(self), indent=2) + "\n"
        (run_dir / MANIFEST_FILENAME).write_text(text)

    @classmethod
    def load(cls, run_dir: Path) -> Self:
        data = json.loads((run_dir / MANIFEST_FILENAME).read_text())
        return cls(**data)


def run_dir_name(dataset: str, model_name: str, seed: int) -> str:
    return f"{dataset}__{model_name}__seed{seed}"


def save_run(
    run_dir: Path, manifest: RunManifest, latents: torch.Tensor, labels: torch.Tensor
) -> None:
    if latents.shape[0] != labels.shape[0]:
        raise ValueError(f"{latents.shape[0]} latents but {labels.shape[0]} labels")
    manifest.save(run_dir)
    torch.save(latents.cpu(), run_dir / LATENTS_FILENAME)
    torch.save(labels.cpu(), run_dir / LABELS_FILENAME)


def load_latents(run_dir: Path) -> torch.Tensor:
    return torch.load(run_dir / LATENTS_FILENAME, weights_only=True)


def load_labels(run_dir: Path) -> torch.Tensor:
    return torch.load(run_dir / LABELS_FILENAME, weights_only=True)


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
