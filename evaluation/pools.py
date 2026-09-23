"""A dataset's pool: the on-disk contract between generation and evaluation.

<root>/<dataset>/
    real/                      the val split: images (uint8) and their labels
    stratified_<n>/            conditioning labels of the stratified schedule
    vaes/<vae>/<vN>/           real/ encoded (latents) and decoded again (images)
    seed<s>/<type>/<model>/<vN>/std<x>/
                               generated latents and images, one per conditioning row

Nothing in real/ or vaes/ is random, so only the generated samples sit under a seed.
Conditioning labels are either real/'s own (`real`, sample i conditioned on image i) or
the stratified file. Every directory is written under a temporary name and renamed when
complete, so a directory with a manifest is a finished one.
"""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

MANIFEST_FILENAME = "manifest.json"
IMAGES = "images"
LATENTS = "latents"
LABELS = "labels"

REAL_LABELS = "real"
PARTIAL_SUFFIX = ".partial"


@dataclass(frozen=True)
class RealManifest:
    dataset: str
    split: str
    n: int
    git_commit: str | None
    kind: str = "real"
    # The dataset yields one label per image as `(N,)`, as MNIST does. Stored as
    # `(N, 1)` like every other label table, and handed back flat to the models.
    flat_labels: bool = False


@dataclass(frozen=True)
class ConditioningManifest:
    dataset: str
    labels: str
    n: int
    git_commit: str | None
    kind: str = "conditioning"


@dataclass(frozen=True)
class ReconstructionManifest:
    dataset: str
    vae_checkpoint: str
    n: int
    git_commit: str | None
    kind: str = "reconstruction"


@dataclass(frozen=True)
class ModelManifest:
    dataset: str
    model_type: str
    model_checkpoint: str
    vae_checkpoint: str
    seed: int
    std_correction: float
    # `real` or the stratified directory name, and which of its columns the model saw.
    labels: str
    label_columns: list[int] | None
    labels_sha: str
    n: int
    git_commit: str | None
    kind: str = "samples"


Manifest = RealManifest | ConditioningManifest | ReconstructionManifest | ModelManifest


# --- layout ---
def real_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "real"


def stratified_key(n_per_cell: int) -> str:
    return f"stratified_{n_per_cell}"


def conditioning_dir(dataset_dir: Path, labels: str) -> Path:
    return real_dir(dataset_dir) if labels == REAL_LABELS else dataset_dir / labels


def split_ref(ref: str) -> tuple[str, str]:
    """`name:v3` -> (`name`, `v3`)."""
    name, _, version = ref.rpartition(":")
    if not name or not version.startswith("v"):
        raise ValueError(f"expected a resolved `name:vN`, got {ref!r}")
    return name, version


def vae_dir(dataset_dir: Path, vae_ref: str) -> Path:
    name, version = split_ref(vae_ref)
    return dataset_dir / "vaes" / name / version


def model_root(dataset_dir: Path, seed: int, model_type: str, name: str) -> Path:
    """All versions of one model under one seed."""
    return dataset_dir / f"seed{seed}" / model_type / name


def model_dir(
    dataset_dir: Path, seed: int, model_type: str, model_ref: str, std: float
) -> Path:
    name, version = split_ref(model_ref)
    return model_root(dataset_dir, seed, model_type, name) / version / f"std{std:g}"


# --- writing ---
def is_complete(directory: Path) -> bool:
    return (directory / MANIFEST_FILENAME).is_file()


def write_dir(
    directory: Path, manifest: Manifest, tensors: dict[str, torch.Tensor]
) -> None:
    """Write `tensors` and `manifest` to `directory`, all or nothing."""
    lengths = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(lengths.values()) | {manifest.n}) != 1:
        raise ValueError(
            f"{directory}: lengths disagree: {lengths}, manifest n={manifest.n}"
        )
    if IMAGES in tensors and tensors[IMAGES].dtype != torch.uint8:
        raise ValueError(f"images must be uint8, got {tensors[IMAGES].dtype}")
    if LABELS in tensors and tensors[LABELS].dim() != 2:
        raise ValueError(
            f"labels must be (N, factors), got {tuple(tensors[LABELS].shape)}"
        )

    partial = directory.with_name(directory.name + PARTIAL_SUFFIX)
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for name, tensor in tensors.items():
        torch.save(tensor.cpu(), partial / f"{name}.pt")
    text = json.dumps(asdict(manifest), indent=2) + "\n"
    (partial / MANIFEST_FILENAME).write_text(text)
    shutil.rmtree(directory, ignore_errors=True)
    partial.rename(directory)
    print(f"Wrote {manifest.n} rows to {directory}")


# --- reading ---
def load_manifest[M: Manifest](directory: Path, cls: type[M]) -> M:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No {cls.kind} data at {directory}: {path} is missing")
    data = json.loads(path.read_text())
    if data.get("kind") != cls.kind:
        raise ValueError(
            f"{directory} holds {data.get('kind')!r} data, not {cls.kind!r}"
        )
    return cls(**data)


def load_tensor(directory: Path, name: str) -> torch.Tensor:
    return torch.load(directory / f"{name}.pt", weights_only=True)


def load_conditioning(dataset_dir: Path, labels: str) -> torch.Tensor:
    return load_tensor(conditioning_dir(dataset_dir, labels), LABELS)


def labels_sha(labels: torch.Tensor) -> str:
    return hashlib.sha256(labels.contiguous().numpy().tobytes()).hexdigest()[:16]


def version_number(directory: Path) -> int:
    return int(directory.name.removeprefix("v"))


def latest_version(root: Path) -> Path | None:
    """The highest `vN` directory under `root`, or None."""
    versions = [d for d in root.glob("v*") if d.is_dir() and d.name[1:].isdigit()]
    return max(versions, key=version_number, default=None)
