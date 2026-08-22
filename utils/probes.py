"""Diagnostics for what an autoencoder does to held-out colour combinations.

The question these answer: when a (digit, foreground, background) combination never appears
in training, does the autoencoder still represent it? Reconstruction error alone can't say
— digit 1 might simply be harder than digit 7, and white backgrounds harder than black — so
everything here is computed per combination, letting the caller marginalize over each axis
and separate "this digit is hard" from "this *combination* is unreachable".
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loaders.colour_mnist import (
    BG_COLOURS,
    FG_COLOURS,
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    TABLE_SHAPE,
)
from models.autoencoder import AbstractAutoencoder

# Palettes as float RGB in [0, 1], indexed the same way labels are.
_FG_PALETTE = np.array(list(FG_COLOURS.values()), dtype=np.float64) / 255.0
_BG_PALETTE = np.array(list(BG_COLOURS.values()), dtype=np.float64) / 255.0

# MNIST digits never reach the image border, so the outer ring is pure background.
_BORDER_MARGIN = 2
# Fraction of pixels, ranked by distance from the background colour, treated as foreground.
_FG_QUANTILE = 0.9


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


def _border_colour(images: torch.Tensor, margin: int = _BORDER_MARGIN) -> torch.Tensor:
    """Mean colour of the outer ring — the background, robustly, without a mask."""
    top = images[:, :, :margin, :].mean(dim=(2, 3))
    bottom = images[:, :, -margin:, :].mean(dim=(2, 3))
    left = images[:, :, :, :margin].mean(dim=(2, 3))
    right = images[:, :, :, -margin:].mean(dim=(2, 3))
    return (top + bottom + left + right) / 4.0


def _foreground_colour(
    originals: torch.Tensor, recons: torch.Tensor, quantile: float = _FG_QUANTILE
) -> torch.Tensor:
    """Colour the reconstruction puts where the *original* is most clearly foreground.

    Picking the pixels from the original rather than the reconstruction is deliberate: a
    reconstruction that dropped the digit entirely should score badly, and it would score
    well if we let it choose its own foreground pixels.
    """
    batch = originals.shape[0]
    background = _border_colour(originals).unsqueeze(-1).unsqueeze(-1)
    distance = (originals - background).pow(2).sum(dim=1).flatten(1)  # (B, H*W)

    num_pixels = distance.shape[1]
    k = max(1, round(num_pixels * (1.0 - quantile)))
    idx = distance.topk(k, dim=1).indices  # (B, k)

    flat_recon = recons.flatten(2)  # (B, C, H*W)
    gathered = flat_recon.gather(
        2, idx.unsqueeze(1).expand(-1, flat_recon.shape[1], -1)
    )
    return gathered.mean(dim=2) if batch else gathered


def _nearest(colours: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """Index of the palette entry closest to each colour, in RGB space."""
    reference = torch.tensor(palette, dtype=colours.dtype, device=colours.device)
    distance = (colours.unsqueeze(1) - reference.unsqueeze(0)).pow(2).sum(-1)
    return distance.argmin(dim=1).cpu().numpy()


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
        recon_bg = _border_colour(recon)
        recon_fg = _foreground_colour(images, recon)
        bg_predicted.append(_nearest(recon_bg, _BG_PALETTE))
        fg_predicted.append(_nearest(recon_fg, _FG_PALETTE))
        bg_colours.append(recon_bg.cpu().numpy())
        fg_colours.append(recon_fg.cpu().numpy())

    per_image_error = np.concatenate(errors)
    all_targets = np.concatenate(targets)
    bg_hit = (np.concatenate(bg_predicted) == all_targets[:, 2]).astype(np.float64)
    fg_hit = (np.concatenate(fg_predicted) == all_targets[:, 1]).astype(np.float64)

    # Distance to the colour that was actually intended, which keeps reporting once the
    # nearest-palette accuracy has saturated at 1.0.
    bg_drift = np.linalg.norm(
        np.concatenate(bg_colours) - _BG_PALETTE[all_targets[:, 2]], axis=1
    )
    fg_drift = np.linalg.norm(
        np.concatenate(fg_colours) - _FG_PALETTE[all_targets[:, 1]], axis=1
    )

    index = (all_targets[:, 0], all_targets[:, 1], all_targets[:, 2])
    counts = np.zeros(TABLE_SHAPE)
    np.add.at(counts, index, 1.0)

    def _mean(values: np.ndarray) -> np.ndarray:
        total = np.zeros(TABLE_SHAPE)
        np.add.at(total, index, values)
        with np.errstate(invalid="ignore"):
            return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

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


@dataclass
class GenerationProbe:
    """Per-combination colour fidelity of images the CSPN *generates*, not reconstructs.

    The autoencoder probe asks whether a held-out combination can be represented; this asks
    whether the conditioned circuit can actually produce it. A gap between the two localizes
    the compositional failure to the hypernetwork rather than the latent space.
    """

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

        generated_bg = _border_colour(images)
        # No source image to locate the digit, so the generated image's own contrast
        # against its border colour has to do it. A flat image therefore reads its
        # "foreground" off noise, which is why contrast is reported alongside.
        generated_fg = _foreground_colour(images, images)

        target_bg = torch.tensor(
            _BG_PALETTE[batch[:, 2].cpu().numpy()], dtype=images.dtype, device=device
        )
        target_fg = torch.tensor(
            _FG_PALETTE[batch[:, 1].cpu().numpy()], dtype=images.dtype, device=device
        )

        bg_hits.append(_nearest(generated_bg, _BG_PALETTE) == batch[:, 2].cpu().numpy())
        fg_hits.append(_nearest(generated_fg, _FG_PALETTE) == batch[:, 1].cpu().numpy())
        bg_drifts.append((generated_bg - target_bg).norm(dim=1).cpu().numpy())
        fg_drifts.append((generated_fg - target_fg).norm(dim=1).cpu().numpy())
        contrasts.append((generated_fg - generated_bg).norm(dim=1).cpu().numpy())

    index = tuple(labels.cpu().numpy().T)
    counts = np.zeros(TABLE_SHAPE)
    np.add.at(counts, index, 1.0)

    def _mean(values: np.ndarray) -> np.ndarray:
        total = np.zeros(TABLE_SHAPE)
        np.add.at(total, index, values.astype(np.float64))
        return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

    return GenerationProbe(
        bg_accuracy=_mean(np.concatenate(bg_hits)),
        fg_accuracy=_mean(np.concatenate(fg_hits)),
        bg_drift_table=_mean(np.concatenate(bg_drifts)),
        fg_drift_table=_mean(np.concatenate(fg_drifts)),
        contrast_table=_mean(np.concatenate(contrasts)),
        samples_per_combination=samples_per_combination,
    )


def stack_images(dataset, indices) -> torch.Tensor:
    """Stack a few dataset images as a batch, whatever the dataset's transform returns."""
    from torchvision.transforms import functional as visual

    items = [dataset[int(i)][0] for i in indices]
    return torch.stack(
        [
            item if isinstance(item, torch.Tensor) else visual.to_tensor(item)
            for item in items
        ]
    )


def latent_mahalanobis(latents: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of every latent from the distribution of `reference` ones.

    The reference set should be the combinations the CSPN was trained on: if held-out
    latents land far outside it, the autoencoder may represent them perfectly and the CSPN
    would still never have learned to reach that region.
    """
    sample = latents[reference]
    mean = sample.mean(axis=0)
    covariance = np.cov(sample, rowvar=False)
    precision = np.linalg.pinv(covariance)
    centered = latents - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, precision, centered))


def per_image_seen(targets: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Boolean per image: was its combination present in training?"""
    return seen[targets[:, 0], targets[:, 1], targets[:, 2]]


def weighted_mean(table: np.ndarray, mask: np.ndarray, counts: np.ndarray) -> float:
    """Count-weighted mean of a per-combination table over the cells `mask` selects."""
    weights = counts * mask
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((np.nan_to_num(table) * weights).sum() / total)


def marginals(values: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Count-weighted means of a (10, 6, 3) table along each axis separately.

    This is the control: it says whether a gap belongs to the digit, the foreground, or the
    background on its own, rather than to the specific combination.
    """
    weighted = np.nan_to_num(values) * counts

    def _along(axes: tuple[int, ...]) -> np.ndarray:
        total = counts.sum(axis=axes)
        return np.where(
            total > 0, weighted.sum(axis=axes) / np.maximum(total, 1), np.nan
        )

    return {
        "digit": _along((1, 2)),
        "fg": _along((0, 2)),
        "bg": _along((0, 1)),
    }


assert _FG_PALETTE.shape == (NUM_FG, 3)
assert _BG_PALETTE.shape == (NUM_BG, 3)
assert TABLE_SHAPE == (NUM_DIGITS, NUM_FG, NUM_BG)
