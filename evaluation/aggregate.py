"""Reducing per-combination tables to the numbers a result section quotes."""

import numpy as np

from dataset_loaders.colour_mnist import TABLE_SHAPE


def combination_mean(
    values: np.ndarray, index: tuple[np.ndarray, ...], counts: np.ndarray
) -> np.ndarray:
    """Mean of `values` per (digit, fg, bg) cell, NaN where a cell has no samples."""
    total = np.zeros(TABLE_SHAPE)
    np.add.at(total, index, values.astype(np.float64))
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


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
