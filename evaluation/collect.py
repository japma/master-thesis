"""Running models over colour-MNIST and collecting metric values into DataFrames.

Every frame has one row per image (or per combination) with `digit`, `fg` and `bg`
columns, so a results table is a `groupby`:

    frame.groupby("source").mean()             overall, per source
    frame.groupby(["source", "seen"]).mean()   trained vs held-out
    frame.groupby("digit").mean()              along one label axis
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from dataset_loaders.colour_mnist import (
    NUM_DIGITS,
    TABLE_SHAPE,
    all_combinations,
    combination_index,
)
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
)
from evaluation.metrics import (
    colour_drift,
    confidence,
    contrast,
    digit_accuracy,
    entropy,
    factor_label_accuracy,
    fit_gaussian,
    joint_label_accuracy,
    mahalanobis,
    palette_accuracy,
    predicted_class_entropy,
    spread,
)
from models.autoencoder import AbstractAutoencoder
from models.classifier import DigitClassifier

# Anything that yields (images, labels) -- a DataLoader in practice.
LabelledBatches = Iterable[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ImageSet:
    """Images, the labels they belong to and the latents they decode from. All on CPU."""

    labels: torch.Tensor  # (B, 3): digit, fg, bg
    latents: torch.Tensor  # (B, latent_dim)
    images: torch.Tensor  # (B, C, H, W)


@dataclass
class ModelEvaluation:
    # One row per image; `source` is "sample", "reconstruction" or "real".
    images: pd.DataFrame
    # One row per combination; `source` is "sample" or "real".
    spread: pd.DataFrame
    # One row per real test image that was scored.
    density: pd.DataFrame
    # The samples `images` was computed on, for figures of exactly those samples.
    samples: ImageSet


@torch.no_grad()
def sample_images(
    model,
    ae: AbstractAutoencoder,
    device: torch.device,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    combinations_per_chunk: int = 32,
) -> ImageSet:
    """`samples_per_combination` samples of every combination, in `all_combinations()` order.

    Chunked because a conditional circuit materialises its parameters per sample.
    """
    model.eval()
    ae.eval()
    labels, latents, images = [], [], []
    for combinations in all_combinations().split(combinations_per_chunk):
        chunk_labels = combinations.repeat_interleave(samples_per_combination, dim=0)
        chunk_latents = model.sample(
            chunk_labels.to(device), std_correction=std_correction
        )
        labels.append(chunk_labels)
        latents.append(chunk_latents.cpu())
        images.append(ae.decode(chunk_latents).cpu())
    return ImageSet(torch.cat(labels), torch.cat(latents), torch.cat(images))


@torch.no_grad()
def encode_images(
    ae: AbstractAutoencoder, loader: LabelledBatches, device: torch.device
) -> ImageSet:
    """Real images with their posterior-mean latents."""
    ae.eval()
    labels, latents, images = [], [], []
    for batch_images, batch_labels in loader:
        labels.append(batch_labels.long())
        latents.append(ae.encode(batch_images.to(device)).cpu())
        images.append(batch_images)
    return ImageSet(torch.cat(labels), torch.cat(latents), torch.cat(images))


@torch.no_grad()
def reconstruct_images(
    ae: AbstractAutoencoder,
    real: ImageSet,
    device: torch.device,
    batch_size: int = 256,
) -> ImageSet:
    ae.eval()
    images = [
        ae.decode(latents.to(device)).cpu()
        for latents in real.latents.split(batch_size)
    ]
    return ImageSet(real.labels, real.latents, torch.cat(images))


@torch.no_grad()
def fit_latent_gaussian(
    ae: AbstractAutoencoder, loader: LabelledBatches, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and precision of the real latents in `loader`, the reference for `mahalanobis`."""
    ae.eval()
    latents = [ae.encode(images.to(device)).cpu() for images, _ in loader]
    return fit_gaussian(torch.cat(latents))


@torch.no_grad()
def score_images(
    image_set: ImageSet,
    judge: DigitClassifier,
    latent_gaussian: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    reference: torch.Tensor | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """One row per image: colour fidelity, what the digit judge reads, latent plausibility.

    `reference` holds the originals of reconstructions. The foreground is then located in
    the original, so a reconstruction that lost or moved the digit cannot pick its own
    foreground pixels and score well.
    """
    judge.eval()
    mean, precision = latent_gaussian
    frames = []
    for start in range(0, len(image_set.labels), batch_size):
        rows = slice(start, start + batch_size)
        labels = image_set.labels[rows].to(device)
        images = image_set.images[rows].to(device)
        locate_in = images if reference is None else reference[rows].to(device)
        # Stays on the CPU: mahalanobis works in float64, which MPS lacks.
        latents = image_set.latents[rows]

        bg = border_colour(images)
        fg = foreground_colour(locate_in, images)
        logits = judge(images)

        columns = {
            "bg_accuracy": palette_accuracy(bg, BG_PALETTE, labels[:, 2]),
            "fg_accuracy": palette_accuracy(fg, FG_PALETTE, labels[:, 1]),
            "bg_drift": colour_drift(bg, BG_PALETTE, labels[:, 2]),
            "fg_drift": colour_drift(fg, FG_PALETTE, labels[:, 1]),
            "contrast": contrast(fg, bg),
            "digit_accuracy": digit_accuracy(logits, labels[:, 0]),
            "digit_confidence": confidence(logits),
            "digit_entropy": entropy(logits),
            "predicted_digit": logits.argmax(dim=1),
            "mahalanobis": mahalanobis(latents, mean, precision),
        }
        frames.append(to_frame(labels, columns))
    return pd.concat(frames, ignore_index=True)


def spread_by_combination(image_set: ImageSet) -> pd.DataFrame:
    """One row per combination: how much its images vary, in pixels and in latents.

    Groups by label rather than by position, so samples and real images work alike.
    """
    rows = []
    for combination in all_combinations():
        members = (image_set.labels == combination).all(dim=1)
        if members.sum() < 2:
            continue
        digit, fg, bg = combination.tolist()
        rows.append(
            {
                "digit": digit,
                "fg": fg,
                "bg": bg,
                "pixel_std": float(spread(image_set.images[members])),
                "latent_std": float(spread(image_set.latents[members])),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def score_density(
    model,
    real: ImageSet,
    device: torch.device,
    max_images: int | None = 2048,
    batch_size: int = 256,
) -> pd.DataFrame:
    """One row per real image: its NLL, and whether the density picks out its label.

    Every latent is scored under all 180 combinations -- 180 model calls per batch -- so
    `max_images` is the cost knob.
    """
    model.eval()
    combinations = all_combinations().to(device)
    count = len(real.labels)
    if max_images is not None:
        count = min(count, max_images)
    frames = []
    for start in range(0, count, batch_size):
        rows = slice(start, min(start + batch_size, count))
        labels = real.labels[rows].to(device)
        latents = real.latents[rows].to(device)

        log_scores = torch.stack(
            [
                model(latents, combination.expand(len(latents), -1))
                for combination in combinations
            ],
            dim=1,
        )
        log_prob = log_scores.gather(1, combination_index(labels).unsqueeze(1))

        columns = {
            "nll": -log_prob.squeeze(1),
            "joint_label_accuracy": joint_label_accuracy(log_scores, labels),
            "digit_label_accuracy": factor_label_accuracy(log_scores, labels, 0),
            "fg_label_accuracy": factor_label_accuracy(log_scores, labels, 1),
            "bg_label_accuracy": factor_label_accuracy(log_scores, labels, 2),
        }
        frames.append(to_frame(labels, columns))
    return pd.concat(frames, ignore_index=True)


def evaluate_model(
    model,
    ae: AbstractAutoencoder,
    judge: DigitClassifier,
    train_loader: LabelledBatches,
    test_loader: LabelledBatches,
    device: torch.device,
    seen: np.ndarray | None = None,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    density_images: int | None = 2048,
) -> ModelEvaluation:
    """Scores the model's samples next to the real test images and their reconstructions.

    Real images give the reference for every sample metric, reconstructions the ceiling
    the autoencoder allows. `seen` is the train split's combination mask; with it every
    frame gets a `seen` column.
    """
    latent_gaussian = fit_latent_gaussian(ae, train_loader, device)
    samples = sample_images(
        model,
        ae,
        device,
        samples_per_combination=samples_per_combination,
        std_correction=std_correction,
    )
    real = encode_images(ae, test_loader, device)
    reconstructions = reconstruct_images(ae, real, device)

    images = pd.concat(
        [
            score_images(samples, judge, latent_gaussian, device).assign(
                source="sample"
            ),
            score_images(
                reconstructions, judge, latent_gaussian, device, reference=real.images
            ).assign(source="reconstruction"),
            score_images(real, judge, latent_gaussian, device).assign(source="real"),
        ],
        ignore_index=True,
    )
    spreads = pd.concat(
        [
            spread_by_combination(samples).assign(source="sample"),
            spread_by_combination(real).assign(source="real"),
        ],
        ignore_index=True,
    )
    density = score_density(model, real, device, max_images=density_images)

    if seen is not None:
        images = mark_seen(images, seen)
        spreads = mark_seen(spreads, seen)
        density = mark_seen(density, seen)
    return ModelEvaluation(
        images=images, spread=spreads, density=density, samples=samples
    )


def to_frame(labels: torch.Tensor, columns: dict[str, torch.Tensor]) -> pd.DataFrame:
    labels = labels.cpu().numpy()
    frame = pd.DataFrame(
        {"digit": labels[:, 0], "fg": labels[:, 1], "bg": labels[:, 2]}
    )
    for name, values in columns.items():
        frame[name] = values.cpu().numpy()
    return frame


def mark_seen(frame: pd.DataFrame, seen: np.ndarray) -> pd.DataFrame:
    """Adds a `seen` column: was the row's combination in the train split?"""
    index = (frame["digit"].to_numpy(), frame["fg"].to_numpy(), frame["bg"].to_numpy())
    return frame.assign(seen=seen[index])


def combination_table(frame: pd.DataFrame, column: str) -> np.ndarray:
    """`column` averaged per combination, as a `(10, 6, 3)` array with NaN for empty cells."""
    table = np.full(TABLE_SHAPE, np.nan)
    means = frame.groupby(["digit", "fg", "bg"])[column].mean()
    for (digit, fg, bg), value in means.items():
        table[digit, fg, bg] = value
    return table


def predicted_digit_entropy(frame: pd.DataFrame) -> float:
    """`predicted_class_entropy` of the judge's readings in `frame`."""
    predictions = torch.from_numpy(frame["predicted_digit"].to_numpy(copy=True))
    return predicted_class_entropy(predictions, NUM_DIGITS)
