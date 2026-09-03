"""Colour fidelity of images the CSPN *generates*, not reconstructs."""

from dataclasses import dataclass

import numpy as np
import torch

from dataset_loaders.colour_mnist import (
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    TABLE_SHAPE,
)
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
class GenerationProbe:
    # All shape (10, 6, 3).
    bg_accuracy: np.ndarray
    fg_accuracy: np.ndarray
    bg_drift_table: np.ndarray
    fg_drift_table: np.ndarray
    # Distance between the generated foreground and background colours. Collapses toward
    # zero when the model emits a flat image rather than a wrongly-coloured digit — a
    # different failure from getting the colour wrong, and worth telling apart.
    contrast_table: np.ndarray
    samples_per_combination: int


@torch.no_grad()
def run_generation_probe(
    cspn,
    ae: AbstractAutoencoder,
    device: torch.device,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    chunk_size: int = 2048,
) -> GenerationProbe:
    """Sample every (digit, fg, bg) combination from the CSPN and check the colours."""
    cspn.eval()
    ae.eval()

    combinations = [
        (digit, fg, bg)
        for digit in range(NUM_DIGITS)
        for fg in range(NUM_FG)
        for bg in range(NUM_BG)
    ]
    labels = torch.tensor(
        [list(combo) for combo in combinations for _ in range(samples_per_combination)],
        dtype=torch.long,
        device=device,
    )

    bg_hits: list[np.ndarray] = []
    fg_hits: list[np.ndarray] = []
    bg_drifts: list[np.ndarray] = []
    fg_drifts: list[np.ndarray] = []
    contrasts: list[np.ndarray] = []

    for start in range(0, labels.shape[0], chunk_size):
        batch = labels[start : start + chunk_size]
        images = ae.decode(cspn.sample(batch, std_correction=std_correction))

        generated_bg = border_colour(images)
        # No source image to locate the digit, so the generated image's own contrast
        # against its border colour has to do it. A flat image therefore reads its
        # "foreground" off noise, which is why contrast is reported alongside.
        generated_fg = foreground_colour(images, images)

        target_bg = torch.tensor(
            BG_PALETTE[batch[:, 2].cpu().numpy()], dtype=images.dtype, device=device
        )
        target_fg = torch.tensor(
            FG_PALETTE[batch[:, 1].cpu().numpy()], dtype=images.dtype, device=device
        )

        bg_hits.append(
            nearest_palette_index(generated_bg, BG_PALETTE) == batch[:, 2].cpu().numpy()
        )
        fg_hits.append(
            nearest_palette_index(generated_fg, FG_PALETTE) == batch[:, 1].cpu().numpy()
        )
        bg_drifts.append((generated_bg - target_bg).norm(dim=1).cpu().numpy())
        fg_drifts.append((generated_fg - target_fg).norm(dim=1).cpu().numpy())
        contrasts.append((generated_fg - generated_bg).norm(dim=1).cpu().numpy())

    index = tuple(labels.cpu().numpy().T)
    counts = np.zeros(TABLE_SHAPE)
    np.add.at(counts, index, 1.0)

    def _mean(values: np.ndarray) -> np.ndarray:
        return combination_mean(values, index, counts)

    return GenerationProbe(
        bg_accuracy=_mean(np.concatenate(bg_hits)),
        fg_accuracy=_mean(np.concatenate(fg_hits)),
        bg_drift_table=_mean(np.concatenate(bg_drifts)),
        fg_drift_table=_mean(np.concatenate(fg_drifts)),
        contrast_table=_mean(np.concatenate(contrasts)),
        samples_per_combination=samples_per_combination,
    )
