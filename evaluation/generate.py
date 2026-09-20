"""Stage 1: sample p(z | y), decode through the VAE, write a sample pool.

No metric is computed here. This stage decides *which* labels get sampled and produces
the images `evaluation.evaluate` scores -- including the reference pool that gives those
scores a ceiling, because decoding happens here and only here.
"""

from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loaders import build_data_loaders
from dataset_loaders.colour_mnist import all_combinations
from evaluation.samples import (
    STRATIFIED_SCHEDULE,
    ReferenceManifest,
    SampleManifest,
    current_git_commit,
    reference_dir,
    save_pool,
    to_uint8,
)
from models.autoencoder import AbstractAutoencoder
from models.autoencoder.pretrained import PretrainedVAE
from utils.checkpoints import (
    load_ae_from_path,
    load_cspn_from_path,
    load_joint_pc_from_path,
    load_nn_baseline_from_path,
    read_source_artifact,
)
from utils.config import (
    DatasetConfig,
    EvaluationRunConfig,
    PretrainedAutoencoderConfig,
    load_dataset_config,
)
from utils.reproducibility import seed_everything
from utils.wandb_utils import download_artifact, trained_with

# Every one of these exposes `sample(labels, std_correction)` over the same latent
# space, so generation treats them interchangeably.
MODEL_LOADERS = {
    "cspn": load_cspn_from_path,
    "joint_pc": load_joint_pc_from_path,
    "nn_baseline": load_nn_baseline_from_path,
}
MODEL_TYPES: tuple[str, ...] = tuple(MODEL_LOADERS)

DEFAULT_N_PER_CELL = 100
DEFAULT_BATCH_SIZE = 256

# `build_data_loaders` maps train=False onto colour-MNIST's val split; the reference
# pool names it so a result says which data the ceiling was measured on.
REFERENCE_SPLIT = "val"


class ConditionalSampler(Protocol):
    def sample(
        self, labels: torch.Tensor, std_correction: float = 1.0
    ) -> torch.Tensor: ...


def stratified_labels(n_per_cell: int = DEFAULT_N_PER_CELL) -> torch.Tensor:
    """Every (digit, fg, bg) combination, `n_per_cell` times each.

    All 180 cells, including the ones a skewed variant never trained on: a schedule
    that only asked for what the model saw could not show the generalization gap.
    """
    if n_per_cell < 1:
        raise ValueError(f"n_per_cell must be at least 1, got {n_per_cell}")
    return all_combinations().repeat_interleave(n_per_cell, dim=0)


@torch.no_grad()
def sample_and_decode(
    model: ConditionalSampler,
    vae: AbstractAutoencoder,
    labels: torch.Tensor,
    device: torch.device,
    std_correction: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One latent and one decoded image per row of `labels`, in the same order."""
    latents: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    for batch in tqdm(labels.split(batch_size), desc="sampling"):
        z = model.sample(batch.to(device), std_correction=std_correction)
        x = vae.decode(z)
        latents.append(z.float().cpu())
        images.append(to_uint8(x).cpu())
    return torch.cat(latents), torch.cat(images)


@torch.no_grad()
def encode_and_decode(
    vae: AbstractAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-trip a loader through the VAE: latents, reconstructions, originals, labels."""
    latents: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    originals: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_images, batch_labels in tqdm(loader, desc="encoding"):
        batch_images = batch_images.to(device)
        z = vae.encode(batch_images)
        x = vae.decode(z)
        latents.append(z.float().cpu())
        images.append(to_uint8(x).cpu())
        originals.append(to_uint8(batch_images).cpu())
        labels.append(batch_labels.long())
    return (
        torch.cat(latents),
        torch.cat(images),
        torch.cat(originals),
        torch.cat(labels),
    )


def load_generative_model(
    model_type: str, artifact: str, device: torch.device, tag: str = "latest"
) -> tuple[ConditionalSampler, str, Path]:
    """The model behind a wandb `name[:version]`, the exact `name:vN` it resolved to,
    and the checkpoint file -- which also records the autoencoder it was trained with."""
    loader = MODEL_LOADERS.get(model_type)
    if loader is None:
        raise ValueError(f"unknown model type {model_type!r}, expected {MODEL_TYPES}")
    path, resolved = download_artifact(artifact, tag)
    model = loader(path, device=device)
    return model.to(device).eval(), resolved, path


def resolve_autoencoder(
    cfg: PretrainedAutoencoderConfig | None, model_path: Path, model_ref: str
) -> tuple[str, str, bool]:
    """Which autoencoder decodes this model's latents: `(name, tag, external)`.

    An unpinned config asks the checkpoint itself, then wandb lineage. A model and a
    decoder that were never trained together produce latents the decoder cannot read.
    """
    if cfg is not None:
        return cfg.name, cfg.tag, cfg.external

    source = read_source_artifact(model_path) or trained_with(model_ref)
    if source is None:
        raise ValueError(
            f"{model_ref} records no autoencoder, and wandb lineage did not answer "
            "either. Pin one in the config's `autoencoder:` block."
        )
    print(f"Decoding with {source}, recorded by {model_ref}")
    return source, "latest", False


def load_vae(
    name: str, tag: str, external: bool, dataset: DatasetConfig, device: torch.device
) -> tuple[AbstractAutoencoder, str]:
    """The VAE and the reference that identifies it, resolved as the trainers do:
    a wandb `name:vN`, or the HuggingFace repo named by an `external: true` entry."""
    if external:
        vae = PretrainedVAE(name=name, height=dataset.height, width=dataset.width)
        return vae.to(device).eval(), name
    path, resolved = download_artifact(name, tag)
    vae = load_ae_from_path(path, device=device)
    return vae.to(device).eval(), resolved


def check_latent_dim(
    model: ConditionalSampler,
    vae: AbstractAutoencoder,
    device: torch.device,
    model_ref: str,
    vae_ref: str,
) -> None:
    """Fail naming both artifacts, rather than as a matmul error inside the decoder."""
    with torch.no_grad():
        probe = model.sample(all_combinations()[:1].to(device))
    sampled = int(probe.shape[1])
    expected = int(vae.get_latent_dim().numel())
    if sampled != expected:
        raise ValueError(
            f"{model_ref} samples {sampled}-dim latents but {vae_ref} decodes "
            f"{expected}-dim ones -- they were not trained together. Drop the config's "
            "`autoencoder:` block to use the one the checkpoint recorded."
        )


def write_reference_pool(
    vae: AbstractAutoencoder,
    resolved_vae: str,
    dataset: str,
    pool_dir: Path,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """The ceiling pool: real validation images and their VAE round trip."""
    dataset_cfg = load_dataset_config(dataset)
    _, val_loader = build_data_loaders(
        dataset_cfg, batch_size=batch_size, drop_last=False
    )
    latents, images, originals, labels = encode_and_decode(vae, val_loader, device)

    manifest = ReferenceManifest(
        vae_checkpoint=resolved_vae,
        dataset=dataset,
        split=REFERENCE_SPLIT,
        n_samples=int(labels.shape[0]),
        git_commit=current_git_commit(),
    )
    save_pool(pool_dir, manifest, latents, images, labels, originals=originals)
    return pool_dir


def generate_pool(cfg: EvaluationRunConfig, device: torch.device) -> Path:
    """Sample, decode, write the pool and its reference pool. Returns the directory."""
    generation = cfg.generation
    model, resolved_model, model_path = load_generative_model(
        cfg.model.model_type, cfg.model.name, device, cfg.model.tag
    )
    name, tag, external = resolve_autoencoder(
        cfg.autoencoder, model_path, resolved_model
    )
    vae, resolved_vae = load_vae(name, tag, external, cfg.dataset, device)
    check_latent_dim(model, vae, device, resolved_model, resolved_vae)

    seed = seed_everything(generation.seed)
    labels = stratified_labels(generation.n_per_cell)
    latents, images = sample_and_decode(
        model,
        vae,
        labels,
        device,
        std_correction=generation.std_correction,
        batch_size=generation.batch_size,
    )

    manifest = SampleManifest(
        model_checkpoint=resolved_model,
        model_type=cfg.model.model_type,
        vae_checkpoint=resolved_vae,
        dataset=cfg.dataset.name,
        schedule=STRATIFIED_SCHEDULE,
        n_per_cell=generation.n_per_cell,
        n_samples=int(labels.shape[0]),
        seed=seed,
        std_correction=generation.std_correction,
        git_commit=current_git_commit(),
    )
    pool_dir = cfg.pool_dir
    save_pool(pool_dir, manifest, latents, images, labels)

    write_reference_pool(
        vae,
        resolved_vae,
        cfg.dataset.name,
        reference_dir(pool_dir),
        device,
        batch_size=generation.batch_size,
    )
    return pool_dir
