"""Frechet Inception Distance between two sets of images, via torchmetrics.

Not in `metrics/`: a metric module scores one source against its labels, while FID
compares a whole set against a reference set, so it has no place in that loop yet.
Call it directly until the wiring is decided.

The pool stores decoded uint8 images, so nothing here touches the VAE.
"""

from collections.abc import Iterator

import pandas as pd
import torch
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
    rows = []
    for source, images in sources.items():
        score = frechet_inception_distance(
            images, originals, device, batch_size=batch_size
        )
        frame = pd.DataFrame([{"value": score, "n": int(images.shape[0])}])
        rows.append(tag(frame, columns, source))

    table = pd.concat(rows, ignore_index=True)
    write_metric(cfg.evaluation.results_root / FILENAME, table, keys)
    print("\nfid (lower is better; reconstruction is the ceiling this model can reach)")
    for row in table.itertuples(index=False):
        print(f"  {row.source:<16} {row.value:.3f}  (n={row.n})")
