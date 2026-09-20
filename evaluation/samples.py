"""A sample pool directory: the on-disk contract between generation and evaluation.

<pool_dir>/
    manifest.json    what produced the pool, and from which checkpoints
    latents.pt       (N, D)        float32
    images.pt        (N, C, H, W)  uint8, already decoded
    labels.pt        (N, 3)        int64 -- digit, fg, bg
<pool_dir>/reference/
    manifest.json
    latents.pt       encoded real validation images
    images.pt        those latents decoded again -- the VAE round trip
    originals.pt     the real validation images themselves
    labels.pt
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

MANIFEST_FILENAME = "manifest.json"
LATENTS_FILENAME = "latents.pt"
IMAGES_FILENAME = "images.pt"
LABELS_FILENAME = "labels.pt"
ORIGINALS_FILENAME = "originals.pt"

REFERENCE_DIRNAME = "reference"

STRATIFIED_SCHEDULE = "stratified"

SAMPLES_KIND = "samples"
REFERENCE_KIND = "reference"


@dataclass(frozen=True)
class SampleManifest:
    model_checkpoint: str
    model_type: str
    vae_checkpoint: str
    dataset: str
    schedule: str
    n_per_cell: int
    n_samples: int
    seed: int
    std_correction: float
    git_commit: str | None
    kind: str = SAMPLES_KIND


@dataclass(frozen=True)
class ReferenceManifest:
    """What the ceiling pool is: one split of real data, round-tripped through the VAE."""

    vae_checkpoint: str
    dataset: str
    split: str
    n_samples: int
    git_commit: str | None
    kind: str = REFERENCE_KIND


Manifest = SampleManifest | ReferenceManifest


def save_manifest(pool_dir: Path, manifest: Manifest) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(manifest), indent=2) + "\n"
    (pool_dir / MANIFEST_FILENAME).write_text(text)


def _load_manifest[M: SampleManifest | ReferenceManifest](
    pool_dir: Path, cls: type[M]
) -> M:
    path = pool_dir / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No {cls.kind} pool at {pool_dir}: {path} is missing")
    data = json.loads(path.read_text())
    kind = data.get("kind")
    if kind != cls.kind:
        raise ValueError(f"{pool_dir} holds a {kind!r} pool, not a {cls.kind!r} one")
    return cls(**data)


def load_sample_manifest(pool_dir: Path) -> SampleManifest:
    return _load_manifest(pool_dir, SampleManifest)


def load_reference_manifest(pool_dir: Path) -> ReferenceManifest:
    return _load_manifest(pool_dir, ReferenceManifest)


def reference_dir(pool_dir: Path) -> Path:
    return pool_dir / REFERENCE_DIRNAME


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Decoder output in [0, 1] as the 8-bit image it is actually standing in for."""
    return (images.clamp(0, 1) * 255).round().to(torch.uint8)


def to_float(images: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_uint8`, back into the [0, 1] range every model here expects."""
    return images.float() / 255.0


def save_pool(
    pool_dir: Path,
    manifest: Manifest,
    latents: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    originals: torch.Tensor | None = None,
) -> None:
    tensors = {"latents": latents, "images": images, "labels": labels}
    if originals is not None:
        tensors["originals"] = originals

    lengths = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"pool tensors disagree on length: {lengths}")
    if images.dtype != torch.uint8:
        raise ValueError(f"images must be uint8, got {images.dtype}")
    if originals is not None and originals.dtype != torch.uint8:
        raise ValueError(f"originals must be uint8, got {originals.dtype}")
    if labels.shape[1] != 3:
        raise ValueError(f"labels must be (N, 3), got {tuple(labels.shape)}")

    save_manifest(pool_dir, manifest)
    torch.save(latents.cpu(), pool_dir / LATENTS_FILENAME)
    torch.save(images.cpu(), pool_dir / IMAGES_FILENAME)
    torch.save(labels.cpu(), pool_dir / LABELS_FILENAME)
    if originals is not None:
        torch.save(originals.cpu(), pool_dir / ORIGINALS_FILENAME)
    print(f"Wrote {manifest.n_samples} samples to {pool_dir}")


def load_latents(pool_dir: Path) -> torch.Tensor:
    return torch.load(pool_dir / LATENTS_FILENAME, weights_only=True)


def load_images(pool_dir: Path) -> torch.Tensor:
    return torch.load(pool_dir / IMAGES_FILENAME, weights_only=True)


def load_labels(pool_dir: Path) -> torch.Tensor:
    return torch.load(pool_dir / LABELS_FILENAME, weights_only=True)


def load_originals(pool_dir: Path) -> torch.Tensor:
    return torch.load(pool_dir / ORIGINALS_FILENAME, weights_only=True)


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
