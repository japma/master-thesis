"""Set metrics: each model's images against the real ones, as whole sets rather than per
sample, averaged over random halvings of the val split. No judge and no labels, which
is why this is its own entrypoint and not a module in `metrics/`.

Each halving cuts the val set in two: one half is the reference, the other is scored
against it as the `real` floor -- the distance of the data to itself at this n. The
reconstruction ceiling is the VAE round trip of that same held-out half, so the gap
between the two rows is the VAE alone. When a model was conditioned on the real labels
(sample i on image i), its generated set is the samples of that same held-out half, so
all three sets carry identical labels; otherwise it is a random subset of the same n.

The halvings depend only on the seed, so every model of a seed is scored on the same
splits, and the real and reconstruction scores are computed once and shared. Features
do not depend on the reference either: each network embeds every image once.
"""

from collections.abc import Callable
from functools import partial

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
    find_models,
    run_columns,
    tag,
    write_metric,
)
from evaluation.features import NETWORKS, extract
from evaluation.pools import (
    IMAGES,
    REAL_LABELS,
    ModelManifest,
    load_tensor,
    real_dir,
    vae_dir,
)
from utils.config import PoolRunConfig
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


Split = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def halvings(
    n_real: int, n_generated: int, n_splits: int, seed: int, paired: bool
) -> tuple[int, list[Split]]:
    """The common n, and per split the reference, held-out and generated indices.

    Half the val split is the most the real floor can have. `paired` generated sets
    line up with the real ones, so they take the held-out indices themselves.
    """
    if paired and n_generated != n_real:
        raise ValueError(
            f"paired sets must match: {n_generated} generated, {n_real} real"
        )
    n = min(n_real // 2, n_generated)
    real_order = torch.Generator().manual_seed(seed)
    generated_order = torch.Generator().manual_seed(seed + 1)
    splits = []
    for _ in range(n_splits):
        order = torch.randperm(n_real, generator=real_order)
        reference, held_out = order[:n], order[n : 2 * n]
        if paired:
            generated = held_out
        elif n_generated <= n:
            generated = torch.arange(n_generated)
        else:
            generated = torch.randperm(n_generated, generator=generated_order)[:n]
        splits.append((reference, held_out, generated))
    return n, splits


def write_set_metric(
    cfg: PoolRunConfig,
    manifest: ModelManifest,
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


def run_sets(cfg: PoolRunConfig, device: torch.device) -> None:
    """Every metric in `cfg.evaluation.set_metrics` for every generated model."""
    dataset_dir = cfg.dataset_dir
    models = find_models(cfg)
    names = cfg.evaluation.set_metrics
    networks = list(dict.fromkeys(SET_METRICS[name][0] for name in names))
    batch_size = cfg.evaluation.batch_size
    metric_device = float64_device(device)
    halving_count = cfg.evaluation.halvings

    real_images = load_tensor(real_dir(dataset_dir), IMAGES)
    vae_refs = sorted({manifest.vae_checkpoint for _, manifest in models})
    n_real = real_images.shape[0]
    per_network = batch_count(n_real, batch_size) * (1 + len(vae_refs)) + sum(
        batch_count(manifest.n, batch_size) for _, manifest in models
    )
    rtpt = start_rtpt(f"sets_{cfg.dataset.name}", per_network * len(networks))

    plans = [
        halvings(
            n_real,
            manifest.n,
            halving_count,
            manifest.seed,
            paired=manifest.labels == REAL_LABELS and manifest.n == n_real,
        )
        for _, manifest in models
    ]
    scores: dict[tuple[int, str], list[dict[str, float]]] = {}
    for network_name in networks:
        network = NETWORKS[network_name](device)
        embed = partial(
            extract,
            network,
            device=device,
            out_device=metric_device,
            batch_size=batch_size,
            rtpt=rtpt,
        )

        real = embed(real_images, desc=f"{network_name} real")
        reconstructions = {
            vae_ref: embed(
                load_tensor(vae_dir(dataset_dir, vae_ref), IMAGES),
                desc=f"{network_name} {vae_ref}",
            )
            for vae_ref in vae_refs
        }
        shared: dict[tuple, float] = {}
        for index, ((directory, manifest), (n, splits)) in enumerate(
            zip(models, plans, strict=True)
        ):
            generated = embed(
                load_tensor(directory, IMAGES),
                desc=f"{network_name} {manifest.model_checkpoint}",
            )
            recon = reconstructions[manifest.vae_checkpoint]
            for name in names:
                metric_network, score = SET_METRICS[name]
                if metric_network != network_name:
                    continue
                per_split = []
                for split, (reference, held_out, sampled) in enumerate(splits):
                    floor_key = (name, manifest.seed, n, split)
                    if floor_key not in shared:
                        shared[floor_key] = score(real[held_out], real[reference])
                    ceiling_key = (*floor_key, manifest.vae_checkpoint)
                    if ceiling_key not in shared:
                        shared[ceiling_key] = score(recon[held_out], real[reference])
                    per_split.append(
                        {
                            REAL: shared[floor_key],
                            RECONSTRUCTION: shared[ceiling_key],
                            GENERATED: score(generated[sampled], real[reference]),
                        }
                    )
                scores[(index, name)] = per_split
        del network, embed

    for index, ((_, manifest), (n, splits)) in enumerate(
        zip(models, plans, strict=True)
    ):
        print(
            f"\n=== {manifest.model_checkpoint} (std={manifest.std_correction:g}, "
            f"seed={manifest.seed}): {len(splits)} halvings, every set cut to n={n}"
        )
        for name in names:
            write_set_metric(cfg, manifest, name, scores[(index, name)], n)
