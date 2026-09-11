"""Turning a latent-space model into pictures.

The metrics say whether a model is right; these say what it actually drew. Sampling
mirrors `run_sample_metrics` exactly -- same `model.sample(labels, std_correction)` then
`ae.decode` -- so a figure and a number always describe the same thing.
"""

from collections.abc import Sequence

import torch

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from evaluation.harness import all_combinations
from models.autoencoder import AbstractAutoencoder


@torch.no_grad()
def decode_samples(
    model,
    ae: AbstractAutoencoder,
    labels: torch.Tensor,
    device: torch.device,
    std_correction: float = 1.0,
) -> torch.Tensor:
    """Images for one batch of labels, on the CPU and ready to plot."""
    model.eval()
    ae.eval()
    latents = model.sample(labels.to(device).long(), std_correction=std_correction)
    return ae.decode(latents).cpu()


@torch.no_grad()
def sample_combination_grid(
    model,
    ae: AbstractAutoencoder,
    device: torch.device,
    samples_per_combination: int = 1,
    std_correction: float = 1.0,
    combinations_per_chunk: int = 32,
) -> torch.Tensor:
    """Every (digit, fg, bg) combination, as `(10, 6, 3, n, C, H, W)`.

    The first four axes are the canonical table order, so this indexes the same way an
    `EvalReport` table does -- `grid[digit, fg, bg]` and `table[digit, fg, bg]` describe
    the same cell.
    """
    combinations = all_combinations()
    chunks = []
    for start in range(0, combinations.shape[0], combinations_per_chunk):
        chunk = combinations[start : start + combinations_per_chunk]
        labels = chunk.repeat_interleave(samples_per_combination, dim=0)
        chunks.append(decode_samples(model, ae, labels, device, std_correction))

    images = torch.cat(chunks)
    return images.view(
        NUM_DIGITS, NUM_FG, NUM_BG, samples_per_combination, *images.shape[1:]
    )


@torch.no_grad()
def sample_for_label(
    model,
    ae: AbstractAutoencoder,
    label: tuple[int, int, int],
    count: int,
    device: torch.device,
    std_correction: float = 1.0,
) -> torch.Tensor:
    """`count` samples of one fixed combination -- the strip that shows whether a model
    draws the same image every time."""
    labels = torch.tensor([label], dtype=torch.long).repeat(count, 1)
    return decode_samples(model, ae, labels, device, std_correction)


@torch.no_grad()
def reconstruct(
    ae: AbstractAutoencoder, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Posterior-mean reconstruction: `encode` is the mean, so this is deterministic and
    two autoencoders can be compared on the same image without sampling noise."""
    ae.eval()
    return ae.decode(ae.encode(images.to(device))).cpu()


@torch.no_grad()
def latent_traversal(
    ae: AbstractAutoencoder,
    image: torch.Tensor,
    dims: Sequence[int],
    values: Sequence[float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Decode one image's latent with a single dimension swept, one row per dimension.

    This is the picture behind a locality number: if the digit block is doing what its
    classifier head was trained to make it do, sweeping a dimension inside it changes the
    digit and nothing else.
    """
    ae.eval()
    latent = ae.encode(image[None].to(device))
    rows: dict[str, torch.Tensor] = {}
    for dim in dims:
        swept = latent.repeat(len(values), 1)
        swept[:, dim] = torch.tensor(values, dtype=swept.dtype, device=swept.device)
        rows[f"dim {dim}"] = ae.decode(swept).cpu()
    return rows
