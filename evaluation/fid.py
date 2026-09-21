"""Frechet Inception Distance between two sets of images, via torchmetrics.

Not in `metrics/`: a metric module scores one source against its labels, while FID
compares a whole set against a reference set, so it has no place in that loop yet.
Call it directly until the wiring is decided.

The pool stores decoded uint8 images, so nothing here touches the VAE.
"""

import copy
from collections.abc import Iterator

import pandas as pd
import torch
from rtpt import RTPT
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from evaluation.evaluate import (
    GENERATED,
    RECONSTRUCTION,
    RUN_KEYS,
    run_columns,
    tag,
    write_metric,
)
from evaluation.samples import (
    load_images,
    load_originals,
    load_reference_manifest,
    load_sample_manifest,
    reference_dir,
    to_float,
)
from utils.config import EvaluationRunConfig
from utils.progress import batch_count, start_rtpt

DEFAULT_BATCH_SIZE = 256
DEFAULT_FEATURE_DIM = 2048


def subsample(images: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """`n` images drawn without replacement, or all of them if there are fewer."""
    if images.shape[0] <= n:
        return images
    idx = torch.randperm(images.shape[0], generator=generator)[:n]
    return images[idx]


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
    # The accumulated feature statistics are float64, which MPS does not support.
    metric_device = torch.device("cpu") if device.type == "mps" else device
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
    """FID of a pool's samples against the real images, with the VAE round trip as the
    ceiling. No judge and no labels, which is why this is its own entrypoint."""
    pool_dir = cfg.pool_dir
    if not pool_dir.is_dir():
        raise FileNotFoundError(
            f"No sample pool at {pool_dir}. Run `uv run generate_samples` with this "
            "config first."
        )
    reference_pool = reference_dir(pool_dir)
    manifest = load_sample_manifest(pool_dir)
    load_reference_manifest(reference_pool)

    originals = load_originals(reference_pool)
    columns = run_columns(manifest)
    keys = {key: columns[key] for key in RUN_KEYS}
    batch_size = cfg.evaluation.batch_size

    sources = {
        GENERATED: load_images(pool_dir),
        RECONSTRUCTION: load_images(reference_pool),
    }

    # FID is biased upward at small sample counts, so a set scored against fewer images
    # looks worse for that reason alone. Every set is cut to the same n, or the
    # generated row would be penalised against a reconstruction ceiling measured on the
    # whole val split.
    common = min(
        int(originals.shape[0]), *(int(images.shape[0]) for images in sources.values())
    )
    generator = torch.Generator().manual_seed(manifest.seed)
    originals = subsample(originals, common, generator)
    sources = {
        source: subsample(images, common, generator)
        for source, images in sources.items()
    }

    total = batch_count(common, batch_size) * (1 + len(sources))
    rtpt = start_rtpt(f"fid_{manifest.dataset}", total)

    scores = fid_against(originals, sources, device, batch_size, rtpt=rtpt)
    rows = [
        tag(pd.DataFrame([{"value": score, "n": common}]), columns, source)
        for source, score in scores.items()
    ]
    table = pd.concat(rows, ignore_index=True)
    write_metric(cfg.evaluation.results_root / FILENAME, table, keys)
    print(
        f"\nfid (lower is better; reconstruction is the ceiling this model can reach)"
        f"\n     every set cut to n={common} so the rows are comparable"
    )
    for row in table.itertuples(index=False):
        print(f"  {row.source:<16} {row.value:.3f}  (n={row.n})")
