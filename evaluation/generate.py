"""Stage 1: fill a dataset's pool (see `evaluation.pools`).

    real/     the val split, once per dataset
    vaes/     the val split round-tripped through each VAE a listed model needs
    seed<s>/  each listed model's samples, decoded by the VAE it was trained against

Anything already complete is skipped, and a listed model wandb does not have is
reported, not fatal. No metric is computed here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from rtpt import RTPT
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loaders import build_dataset
from dataset_loaders.colour_mnist import all_combinations, label_columns
from evaluation.pools import (
    IMAGES,
    LABELS,
    LATENTS,
    REAL_LABELS,
    ConditioningManifest,
    ModelManifest,
    RealManifest,
    ReconstructionManifest,
    conditioning_dir,
    is_complete,
    labels_sha,
    load_conditioning,
    load_manifest,
    load_tensor,
    model_dir,
    real_dir,
    stratified_key,
    vae_dir,
    write_dir,
)
from evaluation.samples import current_git_commit, to_float, to_uint8
from models.autoencoder import AbstractAutoencoder
from models.autoencoder.pretrained import PretrainedVAE
from utils.checkpoints import (
    load_ae_from_path,
    load_cspn_from_path,
    load_gmm_from_path,
    load_joint_pc_from_path,
    load_nn_baseline_from_path,
    load_spn_from_path,
    load_vae_prior_from_path,
    read_source_artifact,
)
from utils.config import (
    DatasetConfig,
    GenerativeModelType,
    PoolModelConfig,
    PoolRunConfig,
    PretrainedAutoencoderConfig,
)
from utils.progress import batch_count, start_rtpt
from utils.reproducibility import seed_everything
from utils.wandb_utils import (
    download_artifact,
    is_versioned,
    resolve_artifact,
    trained_with,
)

# Every one of these exposes `sample(labels, std_correction)` over the same latent
# space, so generation treats them interchangeably.
MODEL_LOADERS = {
    GenerativeModelType.CSPN: load_cspn_from_path,
    GenerativeModelType.JOINT_PC: load_joint_pc_from_path,
    GenerativeModelType.NN_BASELINE: load_nn_baseline_from_path,
    GenerativeModelType.VAE_PRIOR: load_vae_prior_from_path,
    GenerativeModelType.GMM: load_gmm_from_path,
    GenerativeModelType.SPN: load_spn_from_path,
}
MODEL_TYPES: tuple[GenerativeModelType, ...] = tuple(GenerativeModelType)

DEFAULT_N_PER_CELL = 100
DEFAULT_BATCH_SIZE = 256

# `build_dataset(train=False)` is the val split for every dataset.
REAL_SPLIT = "val"
LOADER_WORKERS = 8


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
    rtpt: RTPT | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One latent and one decoded image per row of `labels`, in the same order."""
    latents: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    for batch in tqdm(labels.split(batch_size), desc="sampling"):
        if rtpt is not None:
            rtpt.step(subtitle="sampling")
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
    rtpt: RTPT | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-trip a loader through the VAE: latents, reconstructions, originals, labels."""
    latents: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    originals: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_images, batch_labels in tqdm(loader, desc="encoding"):
        if rtpt is not None:
            rtpt.step(subtitle="reference")
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
    A checkpoint's record without a concrete version (`vae:latest`, from before refs
    were always resolved) says only which collection, so lineage answers instead.
    """
    if cfg is not None:
        return cfg.name, cfg.tag, cfg.external

    recorded = read_source_artifact(model_path)
    if recorded is not None and not is_versioned(recorded):
        print(f"{model_ref} records {recorded}, not a version; asking wandb lineage")
        recorded = None
    source = recorded or trained_with(model_ref)
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
    probe_labels: torch.Tensor,
) -> None:
    """Fail naming both artifacts, rather than as a matmul error inside the decoder.

    `probe_labels` is one row of the labels this run will actually use: the label space
    is the dataset's, and CelebA's 40 attributes are not colour-MNIST's three factors.
    """
    with torch.no_grad():
        probe = model.sample(probe_labels[:1].to(device))
    sampled = int(probe.shape[1])
    expected = int(vae.get_latent_dim().numel())
    if sampled != expected:
        raise ValueError(
            f"{model_ref} samples {sampled}-dim latents but {vae_ref} decodes "
            f"{expected}-dim ones -- they were not trained together. Drop the config's "
            "`autoencoder:` block to use the one the checkpoint recorded."
        )


@torch.no_grad()
def round_trip(
    vae: AbstractAutoencoder,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rtpt: RTPT | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents and reconstructions of `(N, C, H, W)` uint8 images, in order."""
    latents: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    for batch in tqdm(images.split(batch_size), desc="round trip"):
        if rtpt is not None:
            rtpt.step(subtitle="round trip")
        z = vae.encode(to_float(batch).to(device))
        latents.append(z.float().cpu())
        reconstructions.append(to_uint8(vae.decode(z)).cpu())
    return torch.cat(latents), torch.cat(reconstructions)


def ensure_real(
    dataset: DatasetConfig, dataset_dir: Path, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The val split's images and labels, written on first use. Every factor is kept,
    whatever a model conditions on."""
    directory = real_dir(dataset_dir)
    if not is_complete(directory):
        split = build_dataset(
            dataset.name, train=False, size=(dataset.height, dataset.width)
        )
        loader = DataLoader(split, batch_size=batch_size, num_workers=LOADER_WORKERS)
        images, labels = [], []
        flat = False
        for batch_images, batch_labels in tqdm(loader, desc="real"):
            flat = batch_labels.dim() == 1
            images.append(to_uint8(batch_images))
            labels.append(batch_labels.long().reshape(batch_labels.shape[0], -1))
        manifest = RealManifest(
            dataset=dataset.name,
            split=REAL_SPLIT,
            n=sum(batch.shape[0] for batch in images),
            git_commit=current_git_commit(),
            flat_labels=flat,
        )
        write_dir(
            directory,
            manifest,
            {IMAGES: torch.cat(images), LABELS: torch.cat(labels)},
        )
    return load_tensor(directory, IMAGES), load_tensor(directory, LABELS)


def ensure_conditioning(cfg: PoolRunConfig) -> str:
    """Which labels the models are conditioned on, written on first use."""
    if cfg.generation.labels == REAL_LABELS:
        return REAL_LABELS
    key = stratified_key(cfg.generation.n_per_cell)
    directory = conditioning_dir(cfg.dataset_dir, key)
    if not is_complete(directory):
        labels = stratified_labels(cfg.generation.n_per_cell)
        manifest = ConditioningManifest(
            dataset=cfg.dataset.name,
            labels=key,
            n=labels.shape[0],
            git_commit=current_git_commit(),
        )
        write_dir(directory, manifest, {LABELS: labels})
    return key


def ensure_reconstructions(
    vae: AbstractAutoencoder,
    vae_ref: str,
    dataset: str,
    dataset_dir: Path,
    real_images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    rtpt: RTPT | None = None,
) -> None:
    """The val split round-tripped through `vae_ref`, written on first use."""
    directory = vae_dir(dataset_dir, vae_ref)
    if is_complete(directory):
        return
    latents, images = round_trip(vae, real_images, device, batch_size, rtpt)
    manifest = ReconstructionManifest(
        dataset=dataset,
        vae_checkpoint=vae_ref,
        n=images.shape[0],
        git_commit=current_git_commit(),
    )
    write_dir(directory, manifest, {LATENTS: latents, IMAGES: images})


def up_to_date(directory: Path, sha: str) -> bool:
    """Complete, and conditioned on these exact labels."""
    return is_complete(directory) and (
        load_manifest(directory, ModelManifest).labels_sha == sha
    )


@dataclass
class PoolReport:
    generated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def print(self) -> None:
        for title, entries in (
            ("generated", self.generated),
            ("up to date, skipped", self.skipped),
            ("MISSING on wandb", self.missing),
            ("FAILED", self.failed),
        ):
            print(f"\n{title} ({len(entries)})")
            for entry in entries:
                print(f"  {entry}")


def describe(entry: PoolModelConfig, ref: str, seed: int) -> str:
    return f"{ref}  std={entry.std_correction:g}  seed={seed}"


def generate_pools(cfg: PoolRunConfig, device: torch.device) -> PoolReport:
    """Fill `cfg.dataset_dir` with everything the config lists that is not there yet."""
    dataset_dir = cfg.dataset_dir
    batch_size = cfg.generation.batch_size
    report = PoolReport()

    real_images, _ = ensure_real(cfg.dataset, dataset_dir, batch_size)
    labels_key = ensure_conditioning(cfg)
    conditioning = load_conditioning(dataset_dir, labels_key)
    flat = (
        labels_key == REAL_LABELS
        and load_manifest(real_dir(dataset_dir), RealManifest).flat_labels
    )

    todo: list[tuple[PoolModelConfig, str, list[int]]] = []
    for entry in cfg.models:
        ref = resolve_artifact(entry.name, entry.version or "latest")
        if ref is None:
            wanted = f"{entry.name}:{entry.version or 'latest'}"
            print(f"WARNING: {wanted} ({entry.type}) is not on wandb; skipping it")
            report.missing.append(wanted)
            continue
        columns = None if entry.labels is None else label_columns(entry.labels)
        sha = labels_sha(conditioning if columns is None else conditioning[:, columns])
        seeds = []
        for seed in cfg.generation.seeds:
            directory = model_dir(
                dataset_dir, seed, entry.type, ref, entry.std_correction
            )
            if up_to_date(directory, sha):
                report.skipped.append(describe(entry, ref, seed))
            else:
                seeds.append(seed)
        if seeds:
            todo.append((entry, ref, seeds))

    n_samples = sum(len(seeds) for _, _, seeds in todo) * batch_count(
        conditioning.shape[0], batch_size
    )
    n_round_trips = len(todo) * batch_count(real_images.shape[0], batch_size)
    rtpt = start_rtpt(f"pools_{cfg.dataset.name}", n_samples + n_round_trips)

    vaes: dict[str, tuple[AbstractAutoencoder, str]] = {}
    for entry, ref, seeds in todo:
        model, ref, model_path = load_generative_model(entry.type, ref, device)
        if entry.type == GenerativeModelType.VAE_PRIOR:
            vae_name, vae_tag = ref, "latest"
        else:
            try:
                vae_name, vae_tag, _ = resolve_autoencoder(None, model_path, ref)
            except ValueError as error:
                print(f"WARNING: {error}")
                report.failed.append(f"{ref}: no autoencoder recorded")
                continue
        if vae_name not in vaes:
            vaes[vae_name] = load_vae(vae_name, vae_tag, False, cfg.dataset, device)
        vae, vae_ref = vaes[vae_name]
        ensure_reconstructions(
            vae,
            vae_ref,
            cfg.dataset.name,
            dataset_dir,
            real_images,
            device,
            batch_size,
            rtpt,
        )

        columns = None if entry.labels is None else label_columns(entry.labels)
        labels = conditioning if columns is None else conditioning[:, columns]
        model_labels = labels.squeeze(1) if flat else labels
        check_latent_dim(model, vae, device, ref, vae_ref, model_labels)
        for seed in seeds:
            seed_everything(seed)
            latents, images = sample_and_decode(
                model,
                vae,
                model_labels,
                device,
                std_correction=entry.std_correction,
                batch_size=batch_size,
                rtpt=rtpt,
            )
            manifest = ModelManifest(
                dataset=cfg.dataset.name,
                model_type=entry.type,
                model_checkpoint=ref,
                vae_checkpoint=vae_ref,
                seed=seed,
                std_correction=entry.std_correction,
                labels=labels_key,
                label_columns=columns,
                labels_sha=labels_sha(labels),
                n=images.shape[0],
                git_commit=current_git_commit(),
            )
            directory = model_dir(
                dataset_dir, seed, entry.type, ref, entry.std_correction
            )
            write_dir(directory, manifest, {LATENTS: latents, IMAGES: images})
            report.generated.append(describe(entry, ref, seed))
        del model

    report.print()
    return report
