"""Frechet Inception Distance between two sets of images, via torchmetrics.

Not in `metrics/`: a metric module scores one source against its labels, while FID
compares a whole set against a reference set, so it has no place in that loop yet.
Call it directly until the wiring is decided.

The pool stores decoded uint8 images, so nothing here touches the VAE.
"""

import copy
from collections.abc import Iterator

import torch
from rtpt import RTPT
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from evaluation.evaluate import GENERATED, REAL, RECONSTRUCTION
from evaluation.halvings import halvings, load_set_pool, write_halvings
from evaluation.samples import to_float
from utils.config import EvaluationRunConfig
from utils.progress import batch_count, start_rtpt
from utils.reproducibility import float64_device

DEFAULT_BATCH_SIZE = 256
DEFAULT_FEATURE_DIM = 2048


def batches(
    images: torch.Tensor, batch_size: int = DEFAULT_BATCH_SIZE, desc: str = "features"
) -> Iterator[torch.Tensor]:
    """`(N, C, H, W)` uint8 images as float batches in [0, 1]."""
    for batch in tqdm(images.split(batch_size), desc=desc):
        yield to_float(batch)


@torch.no_grad()
def fid_against(
    reference_images: torch.Tensor,
    image_sets: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    feature: int = DEFAULT_FEATURE_DIM,
    rtpt: RTPT | None = None,
) -> dict[str, float]:
    """FID of each set in `image_sets` against `reference_images`.

    The reference features are computed once and shared. They are the bulk of the work
    -- CelebA's reference pool is twice the size of a sample pool -- and they do not
    change between sources, so scoring two sources costs one reference pass, not two.

    Downloads the Inception weights on first use.
    """
    metric_device = float64_device(device)
    base = FrechetInceptionDistance(feature=feature, normalize=True).to(metric_device)
    for batch in batches(reference_images, batch_size, desc="reference features"):
        base.update(batch.to(metric_device), real=True)
        if rtpt is not None:
            rtpt.step(subtitle="reference")

    scores: dict[str, float] = {}
    for name, images in image_sets.items():
        scorer = copy.deepcopy(base)
        for batch in batches(images, batch_size, desc=f"{name} features"):
            scorer.update(batch.to(metric_device), real=False)
            if rtpt is not None:
                rtpt.step(subtitle=name)
        scores[name] = float(scorer.compute())
    return scores


def frechet_inception_distance(
    images: torch.Tensor,
    reference_images: torch.Tensor,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    feature: int = DEFAULT_FEATURE_DIM,
) -> float:
    """FID of `images` against `reference_images`, both `(N, C, H, W)` uint8."""
    return fid_against(reference_images, {"fid": images}, device, batch_size, feature)[
        "fid"
    ]


FILENAME = "fid.csv"


def run_fid(cfg: EvaluationRunConfig, device: torch.device) -> None:
    """FID of a pool's samples against the real images, averaged over random halvings
    of the val split (see `evaluation.halvings`). No judge and no labels, which is why
    this is its own entrypoint.

    Inception features depend on the reference half, so each split recomputes them.
    """
    manifest, originals, reconstructions, generated = load_set_pool(cfg)
    batch_size = cfg.evaluation.batch_size
    generator = torch.Generator().manual_seed(manifest.seed)
    n, splits = halvings(
        originals.shape[0], generated.shape[0], cfg.evaluation.halvings, generator
    )

    total = batch_count(n, batch_size) * 4 * len(splits)
    rtpt = start_rtpt(f"fid_{manifest.dataset}", total)

    scores = []
    for reference, held_out, sampled in splits:
        sources = {
            REAL: originals[held_out],
            RECONSTRUCTION: reconstructions[held_out],
            GENERATED: generated[sampled],
        }
        scores.append(
            fid_against(originals[reference], sources, device, batch_size, rtpt=rtpt)
        )
    write_halvings(cfg, manifest, FILENAME, "fid", scores, n)
