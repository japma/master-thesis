"""Frechet Inception Distance between two sets of images, via torchmetrics.

Not in `metrics/`: a metric module scores one source against its labels, while FID
compares a whole set against a reference set, so it has no place in that loop yet.
Call it directly until the wiring is decided.

The pool stores decoded uint8 images, so nothing here touches the VAE.
"""

from collections.abc import Iterator

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from evaluation.samples import to_float

DEFAULT_BATCH_SIZE = 256
DEFAULT_FEATURE_DIM = 2048


def batches(
    images: torch.Tensor, batch_size: int = DEFAULT_BATCH_SIZE, desc: str = "features"
) -> Iterator[torch.Tensor]:
    """`(N, C, H, W)` uint8 images as float batches in [0, 1]."""
    for batch in tqdm(images.split(batch_size), desc=desc):
        yield to_float(batch)


@torch.no_grad()
def frechet_inception_distance(
    images: torch.Tensor,
    reference_images: torch.Tensor,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    feature: int = DEFAULT_FEATURE_DIM,
) -> float:
    """FID of `images` against `reference_images`, both `(N, C, H, W)` uint8.

    Downloads the Inception weights on first use.
    """
    # The accumulated feature statistics are float64, which MPS does not support.
    metric_device = torch.device("cpu") if device.type == "mps" else device
    metric = FrechetInceptionDistance(feature=feature, normalize=True).to(metric_device)
    for batch in batches(reference_images, batch_size, desc="reference features"):
        metric.update(batch.to(metric_device), real=True)
    for batch in batches(images, batch_size, desc="sample features"):
        metric.update(batch.to(metric_device), real=False)
    return float(metric.compute())
