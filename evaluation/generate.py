"""Stage 1: sample or encode latents and write them out as a run."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_loaders.colour_mnist import ColourMNIST
from evaluation.models import (
    StdCorrectedSampler,
    count_parameters,
    load_model,
    load_vae,
)
from evaluation.protocols import ConditionalSampler, LatentCodec
from evaluation.run import (
    RunManifest,
    current_git_commit,
    load_labels,
    run_dir_name,
    save_run,
)
from utils.reproducibility import seed_everything
from utils.wandb_utils import download_artifact


def colour_mnist_dataset_name(variant: str, split: str) -> str:
    return f"colour_mnist_{variant}_{split}"


@torch.no_grad()
def sample_latents(
    sampler: ConditionalSampler,
    labels: torch.Tensor,
    n_per_label: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`n_per_label` latents for every row of `labels`, with the label of each latent."""
    latents: list[torch.Tensor] = []
    sample_labels: list[torch.Tensor] = []
    for y in tqdm(labels.split(batch_size), desc="sampling"):
        z = sampler.sample(y.to(device), n_per_label)
        latents.append(z.flatten(0, 1).cpu())
        sample_labels.append(y.repeat_interleave(n_per_label, dim=0))
    return torch.cat(latents), torch.cat(sample_labels)


@torch.no_grad()
def encode_loader(
    vae: LatentCodec,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, batch_labels in tqdm(loader, desc="encoding"):
        latents.append(vae.encode(images.to(device)).cpu())
        labels.append(batch_labels.long())
    return torch.cat(latents), torch.cat(labels)


def generate_real_run(
    variant: str,
    split: str,
    vae_artifact: str,
    output_root: Path,
    device: torch.device,
    data_root: Path = Path("data"),
    batch_size: int = 256,
) -> Path:
    """Real colour-MNIST images encoded through the VAE, as the run `real`."""
    vae, resolved_vae = load_vae(vae_artifact, device)
    dataset = ColourMNIST(
        root=data_root, split=split, variant=variant, transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents, labels = encode_loader(vae, loader, device)

    manifest = RunManifest(
        model_name="real",
        dataset=colour_mnist_dataset_name(variant, split),
        seed=0,
        n_samples=latents.shape[0],
        checkpoint_path=None,
        vae_checkpoint=resolved_vae,
        git_commit=current_git_commit(),
        std_correction=None,
        sampling_seconds=None,
        n_parameters=None,
    )
    run_dir = output_root / run_dir_name(manifest.dataset, manifest.model_name, 0)
    save_run(run_dir, manifest, latents, labels)
    return run_dir


def generate_model_run(
    model_type: str,
    model_artifact: str,
    model_name: str,
    vae_artifact: str,
    reference_dir: Path,
    output_root: Path,
    seed: int,
    std_correction: float,
    device: torch.device,
    n_per_label: int = 1,
    batch_size: int = 256,
) -> Path:
    """Samples for every label in the reference run, so both share one label distribution."""
    reference = RunManifest.load(reference_dir)
    _, resolved_vae = download_artifact(vae_artifact)
    if reference.vae_checkpoint != resolved_vae:
        raise ValueError(
            f"reference {reference_dir} was encoded with {reference.vae_checkpoint}, "
            f"not {resolved_vae}"
        )

    model, resolved_model = load_model(model_type, model_artifact, device)
    sampler = StdCorrectedSampler(model, std_correction)
    labels = load_labels(reference_dir)

    seed_everything(seed)
    start = time.perf_counter()
    latents, sample_labels = sample_latents(
        sampler, labels, n_per_label, device, batch_size=batch_size
    )
    sampling_seconds = time.perf_counter() - start

    manifest = RunManifest(
        model_name=model_name,
        dataset=reference.dataset,
        seed=seed,
        n_samples=latents.shape[0],
        checkpoint_path=resolved_model,
        vae_checkpoint=resolved_vae,
        git_commit=current_git_commit(),
        std_correction=std_correction,
        sampling_seconds=sampling_seconds,
        n_parameters=count_parameters(model),
    )
    run_dir = output_root / run_dir_name(manifest.dataset, model_name, seed)
    save_run(run_dir, manifest, latents, sample_labels)
    return run_dir
