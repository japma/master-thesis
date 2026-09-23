"""Random halvings of the val split, the protocol FID and CMMD both average over.

Each halving cuts the val set in two: one half is the reference, the other is scored
against it as the `real` floor -- the distance of the data to itself at this n. The
reconstruction ceiling is the VAE round trip of that same held-out half, so the gap
between the two rows is the VAE alone. Every set in every halving is cut to the same n,
since both distances depend on it.
"""

import pandas as pd
import torch

from evaluation.evaluate import RUN_KEYS, run_columns, tag, write_metric
from evaluation.samples import (
    SampleManifest,
    load_images,
    load_originals,
    load_reference_manifest,
    load_sample_manifest,
    reference_dir,
)
from utils.config import EvaluationRunConfig


def load_set_pool(
    cfg: EvaluationRunConfig,
) -> tuple[SampleManifest, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A pool's manifest, then its originals, reconstructions and generated images."""
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
    reconstructions = load_images(reference_pool)
    if reconstructions.shape[0] != originals.shape[0]:
        raise ValueError(
            f"{reference_pool} holds {reconstructions.shape[0]} reconstructions for "
            f"{originals.shape[0]} originals; they must be paired."
        )
    return manifest, originals, reconstructions, load_images(pool_dir)


def halvings(
    n_originals: int, n_generated: int, n_splits: int, generator: torch.Generator
) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """The common n, and per split the reference, held-out and generated indices.

    Half the val split is the most the real floor can have.
    """
    n = min(n_originals // 2, n_generated)
    splits = []
    for _ in range(n_splits):
        order = torch.randperm(n_originals, generator=generator)
        if n_generated <= n:
            generated = torch.arange(n_generated)
        else:
            generated = torch.randperm(n_generated, generator=generator)[:n]
        splits.append((order[:n], order[n : 2 * n], generated))
    return n, splits


def write_halvings(
    cfg: EvaluationRunConfig,
    manifest: SampleManifest,
    filename: str,
    label: str,
    scores: list[dict[str, float]],
    n: int,
) -> None:
    """One row per split and source in `filename`, and the mean ± std per source."""
    columns = run_columns(manifest)
    keys = {key: columns[key] for key in RUN_KEYS}
    rows = [
        tag(pd.DataFrame([{"split": split, "value": score, "n": n}]), columns, source)
        for split, split_scores in enumerate(scores)
        for source, score in split_scores.items()
    ]
    table = pd.concat(rows, ignore_index=True)
    write_metric(cfg.evaluation.results_root / filename, table, keys)

    summary = table.groupby("source", sort=False)["value"].agg(["mean", "std"])
    print(
        f"\n{label} (lower is better; real is the floor, reconstruction the VAE ceiling)"
        f"\n     {len(scores)} splits of the val set, every set cut to n={n}"
    )
    for source, row in summary.iterrows():
        print(f"  {source:<16} {row['mean']:.3f} ± {row['std']:.3f}")
