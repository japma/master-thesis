"""Set metrics: a pool's images against the real ones, as whole sets rather than per
sample, averaged over random halvings of the val set"""

from collections.abc import Callable

import pandas as pd
import torch

from evaluation.distances import (
    cmmd,
    frechet_distance,
    kernel_inception_distance,
    precision,
    recall,
)
from evaluation.evaluate import (
    GENERATED,
    REAL,
    RECONSTRUCTION,
    RUN_KEYS,
    run_columns,
    tag,
    write_metric,
)
from evaluation.features import NETWORKS, extract
from evaluation.samples import (
    SampleManifest,
    load_images,
    load_originals,
    load_reference_manifest,
    load_sample_manifest,
    reference_dir,
)
from utils.config import EvaluationRunConfig
from utils.progress import batch_count, start_rtpt
from utils.reproducibility import float64_device

# Each metric: the network whose features it compares, and its score of a sample set
# against the reference set. Written to `<name>.csv`.
SET_METRICS: dict[str, tuple[str, Callable[[torch.Tensor, torch.Tensor], float]]] = {
    "fid": ("inception", frechet_distance),
    "kid": ("inception", kernel_inception_distance),
    "precision": ("vgg16", precision),
    "recall": ("vgg16", recall),
    "cmmd": ("clip", cmmd),
    "fd_dinov2": ("dinov2", frechet_distance),
}


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


def write_set_metric(
    cfg: EvaluationRunConfig,
    manifest: SampleManifest,
    name: str,
    scores: list[dict[str, float]],
    n: int,
) -> None:
    """One row per split and source in `<name>.csv`, and the mean ± std per source."""
    columns = run_columns(manifest)
    keys = {key: columns[key] for key in RUN_KEYS}
    rows = [
        tag(pd.DataFrame([{"split": split, "value": score, "n": n}]), columns, source)
        for split, split_scores in enumerate(scores)
        for source, score in split_scores.items()
    ]
    table = pd.concat(rows, ignore_index=True)
    write_metric(cfg.evaluation.results_root / f"{name}.csv", table, keys)

    summary = table.groupby("source", sort=False)["value"].agg(["mean", "std"])
    print(f"\n{name}")
    for source, row in summary.iterrows():
        print(f"  {source:<16} {row['mean']:.4f} ± {row['std']:.4f}")


def run_sets(cfg: EvaluationRunConfig, device: torch.device) -> None:
    """Every metric in `cfg.evaluation.set_metrics` for one pool, one CSV each."""
    manifest, originals, reconstructions, generated = load_set_pool(cfg)
    names = cfg.evaluation.set_metrics
    networks = list(dict.fromkeys(SET_METRICS[name][0] for name in names))
    batch_size = cfg.evaluation.batch_size

    sets = {REAL: originals, RECONSTRUCTION: reconstructions, GENERATED: generated}
    per_network = sum(batch_count(x.shape[0], batch_size) for x in sets.values())
    rtpt = start_rtpt(f"sets_{manifest.dataset}", per_network * len(networks))
    metric_device = float64_device(device)

    features: dict[str, dict[str, torch.Tensor]] = {}
    for network_name in networks:
        network = NETWORKS[network_name](device)
        features[network_name] = {
            source: extract(
                network,
                images,
                device,
                metric_device,
                batch_size,
                f"{network_name} {source}",
                rtpt,
            )
            for source, images in sets.items()
        }
        del network

    generator = torch.Generator().manual_seed(manifest.seed)
    n, splits = halvings(
        originals.shape[0], generated.shape[0], cfg.evaluation.halvings, generator
    )
    print(f"\n{len(splits)} halvings of the val set, every set cut to n={n}")
    for name in names:
        network_name, score = SET_METRICS[name]
        f = features[network_name]
        scores = [
            {
                REAL: score(f[REAL][held_out], f[REAL][reference]),
                RECONSTRUCTION: score(f[RECONSTRUCTION][held_out], f[REAL][reference]),
                GENERATED: score(f[GENERATED][sampled], f[REAL][reference]),
            }
            for reference, held_out, sampled in splits
        ]
        write_set_metric(cfg, manifest, name, scores, n)
