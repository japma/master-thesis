"""Evaluation metrics as plain functions.

Like `torch.nn.functional` losses with `reduction="none"`: tensors in, one value per
image out. Callers run the models and read colours off the images; these only do the
math.
"""

import math

import numpy as np
import torch

from dataset_loaders.colour_mnist import TABLE_SHAPE, combination_index


def palette_accuracy(
    colours: torch.Tensor, palette: np.ndarray, targets: torch.Tensor
) -> torch.Tensor:
    """1.0 where the palette entry nearest to each colour `(B, 3)` is its target index."""
    reference = torch.as_tensor(palette, dtype=colours.dtype, device=colours.device)
    distance = (colours.unsqueeze(1) - reference.unsqueeze(0)).pow(2).sum(dim=2)
    return (distance.argmin(dim=1) == targets).float()


def colour_drift(
    colours: torch.Tensor, palette: np.ndarray, targets: torch.Tensor
) -> torch.Tensor:
    """RGB distance from the intended palette colour. Keeps moving once accuracy is 1.0."""
    reference = torch.as_tensor(palette, dtype=colours.dtype, device=colours.device)
    return (colours - reference[targets]).norm(dim=1)


def contrast(foreground: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """RGB distance between foreground and background. Near zero for a flat image."""
    return (foreground - background).norm(dim=1)


def digit_accuracy(logits: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == digits).float()


def confidence(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=1).max(dim=1).values


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the predicted distribution: near 0 on a clear digit, near log 10 on a blur."""
    probabilities = logits.softmax(dim=1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def predicted_class_entropy(predictions: torch.Tensor, num_classes: int) -> float:
    """Entropy of how often each class is predicted, normalised to [0, 1].

    0 for a model that draws the same digit for every label, however clean it looks.
    """
    counts = torch.bincount(predictions, minlength=num_classes).double()
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * probabilities.log()).sum() / math.log(num_classes))


def fit_gaussian(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and precision of `(N, D)` latents, for `mahalanobis`."""
    latents = latents.double()
    return latents.mean(dim=0), torch.linalg.pinv(torch.cov(latents.T))


def mahalanobis(
    latents: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    centered = latents.double() - mean
    return torch.einsum("ij,jk,ik->i", centered, precision, centered).sqrt()


def spread(samples: torch.Tensor) -> torch.Tensor:
    """Standard deviation across `(N, ...)` samples, averaged over features."""
    return samples.flatten(1).std(dim=0).mean()


def joint_label_accuracy(
    log_scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """1.0 where the true combination has the highest score.

    `log_scores` is `(B, 180)`: the log-density of each latent under every combination,
    in `all_combinations()` order.
    """
    return (log_scores.argmax(dim=1) == combination_index(labels)).float()


def factor_label_accuracy(
    log_scores: torch.Tensor, labels: torch.Tensor, factor: int
) -> torch.Tensor:
    """1.0 where the posterior over one label factor (0 digit, 1 fg, 2 bg) peaks at the truth.

    Assumes a uniform prior over combinations, so the posterior is the softmax of the
    scores, marginalised over the other two factors.
    """
    posterior = log_scores.softmax(dim=1).reshape(-1, *TABLE_SHAPE)
    other_axes = tuple(axis + 1 for axis in range(3) if axis != factor)
    predicted = posterior.sum(dim=other_axes).argmax(dim=1)
    return (predicted == labels[:, factor]).float()
