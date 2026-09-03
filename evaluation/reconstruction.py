"""Does the autoencoder represent colour combinations it never saw in training?"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loaders.colour_mnist import TABLE_SHAPE
from evaluation.aggregate import combination_mean
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from models.autoencoder import AbstractAutoencoder


@dataclass
class CombinationProbe:
    """Per-(digit, fg, bg) reconstruction diagnostics, plus the raw latents behind them."""

    # Per-combination tables, shape (10, 6, 3), NaN where a combination has no images.
    counts: np.ndarray
    error: np.ndarray
    bg_accuracy: np.ndarray
    fg_accuracy: np.ndarray
    # Mean RGB distance from the intended colour. Keeps reporting after accuracy
    # saturates at 1.0, which it does as soon as the palette is well separated.
    bg_drift_table: np.ndarray
    fg_drift_table: np.ndarray

    # Per-image, shape (N,) — or (N, latent_dim) for the latents.
    latents: np.ndarray
    targets: np.ndarray
    per_image_error: np.ndarray
    bg_hit: np.ndarray
    fg_hit: np.ndarray
    bg_drift: np.ndarray
    fg_drift: np.ndarray


@torch.no_grad()
def run_combination_probe(
    ae: AbstractAutoencoder, loader: DataLoader, device: torch.device
) -> CombinationProbe:
    """Encode/decode every image in `loader`, aggregating results per combination.

    Uses the deterministic `encode` (the posterior mean) rather than `forward`'s
    reparameterized sample, so repeated runs agree and the latents match what the CSPN is
    trained on.
    """
    ae.eval()

    errors: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    bg_predicted: list[np.ndarray] = []
    fg_predicted: list[np.ndarray] = []
    bg_colours: list[np.ndarray] = []
    fg_colours: list[np.ndarray] = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        latent = ae.encode(images)
        recon = ae.decode(latent)

        errors.append((recon - images).pow(2).mean(dim=(1, 2, 3)).cpu().numpy())
        latents.append(latent.cpu().numpy())
        targets.append(target.numpy())
        recon_bg = border_colour(recon)
        recon_fg = foreground_colour(images, recon)
        bg_predicted.append(nearest_palette_index(recon_bg, BG_PALETTE))
        fg_predicted.append(nearest_palette_index(recon_fg, FG_PALETTE))
        bg_colours.append(recon_bg.cpu().numpy())
        fg_colours.append(recon_fg.cpu().numpy())

    per_image_error = np.concatenate(errors)
    all_targets = np.concatenate(targets)
    bg_hit = (np.concatenate(bg_predicted) == all_targets[:, 2]).astype(np.float64)
    fg_hit = (np.concatenate(fg_predicted) == all_targets[:, 1]).astype(np.float64)

    # Distance to the colour that was actually intended, which keeps reporting once the
    # nearest-palette accuracy has saturated at 1.0.
    bg_drift = np.linalg.norm(
        np.concatenate(bg_colours) - BG_PALETTE[all_targets[:, 2]], axis=1
    )
    fg_drift = np.linalg.norm(
        np.concatenate(fg_colours) - FG_PALETTE[all_targets[:, 1]], axis=1
    )

    index = (all_targets[:, 0], all_targets[:, 1], all_targets[:, 2])
    counts = np.zeros(TABLE_SHAPE)
    np.add.at(counts, index, 1.0)

    def _mean(values: np.ndarray) -> np.ndarray:
        return combination_mean(values, index, counts)

    return CombinationProbe(
        counts=counts,
        error=_mean(per_image_error),
        bg_accuracy=_mean(bg_hit),
        fg_accuracy=_mean(fg_hit),
        bg_drift_table=_mean(bg_drift),
        fg_drift_table=_mean(fg_drift),
        latents=np.concatenate(latents),
        targets=all_targets,
        per_image_error=per_image_error,
        bg_hit=bg_hit,
        fg_hit=fg_hit,
        bg_drift=bg_drift,
        fg_drift=fg_drift,
    )
