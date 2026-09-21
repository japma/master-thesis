"""Marginalized queries: "a green 5, background unspecified".

A query is a label row with `UNSPECIFIED` where a factor is free. Nothing here scores a
single sample: with a factor free there is no single right answer, only a right *mix*,
so the unit of measurement is the histogram over one query's samples.

Two pieces:
  `sample_labels`  the mixture reference -- free factors drawn from the training
                   conditional, then the model is asked a fully specified question.
                   Correct by construction, so its calibration is the floor.
  `calibration`    how far a set of images is from the distribution it should have.
"""

from collections.abc import Sequence

import pandas as pd
import torch

from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.conditionals import (
    CARDINALITIES,
    FACTOR_NAMES,
    UNSPECIFIED,
    conditional,
    matching,
    total_variation,
    training_labels,
)
from evaluation.evaluate import RUN_KEYS, tag, write_metric
from evaluation.generate import (
    check_latent_dim,
    load_generative_model,
    load_vae,
    resolve_autoencoder,
    sample_and_decode,
)
from evaluation.samples import BG, DIGIT, FG, to_float, to_uint8
from models.cspn.joint_pc import JointPC
from utils.config import EvaluationRunConfig
from utils.progress import start_rtpt
from utils.reproducibility import seed_everything

# The judge reads digits, the palette reads colours; only the latter is available here,
# so a query has to say which digit it wants.
READABLE = (FG, BG)


def as_query(query: Sequence[int]) -> torch.Tensor:
    """Validate a `[digit, fg, bg]` spec and return it as a tensor."""
    row = torch.tensor(list(query), dtype=torch.long)
    if row.shape != (3,):
        raise ValueError(f"a query is [digit, fg, bg], got {list(query)}")
    if int(row[DIGIT]) == UNSPECIFIED:
        raise ValueError(
            "digit must be specified: reading a digit back needs the judge, which "
            "this path does not load"
        )
    free = free_factors(row)
    if not free:
        raise ValueError(
            f"query {list(query)} specifies every factor -- nothing is marginalized"
        )
    for factor in free:
        if factor not in READABLE:
            raise ValueError(f"factor {FACTOR_NAMES[factor]} cannot be read off pixels")
    return row


def free_factors(query: torch.Tensor) -> list[int]:
    """Which factors the query leaves to the model."""
    return [f for f, value in enumerate(query.tolist()) if value == UNSPECIFIED]


def sample_labels(
    query: torch.Tensor,
    n: int,
    labels: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`n` fully specified labels for `query`, the free factors drawn from training.

    Drawn by picking whole training rows, so several free factors come from their joint
    conditional rather than from independent marginals.
    """
    rows = matching(labels, query)
    if not bool(rows.any()):
        raise ValueError(f"no training rows match query {query.tolist()}")
    candidates = labels[rows]
    idx = torch.randint(candidates.shape[0], (n,), generator=generator)
    return candidates[idx].clone()


def read_factor(images: torch.Tensor, factor: int) -> torch.Tensor:
    """The factor each image actually shows, read off the pixels."""
    if factor == FG:
        return nearest_palette_index(foreground_colour(images), FG_PALETTE)
    if factor == BG:
        return nearest_palette_index(border_colour(images), BG_PALETTE)
    raise ValueError(f"factor {factor} cannot be read off pixels")


def histogram(values: torch.Tensor, factor: int) -> torch.Tensor:
    """Frequency of each value of `factor`, over a query's samples."""
    counts = torch.bincount(values, minlength=CARDINALITIES[factor])
    return counts / counts.sum()


def _query_columns(query: torch.Tensor) -> dict[str, int]:
    return {name: int(query[f]) for f, name in enumerate(FACTOR_NAMES)}


def calibration(
    images: torch.Tensor, query: torch.Tensor, labels: torch.Tensor
) -> pd.DataFrame:
    """One row per free factor: how far the generated mix is from the training one."""
    rows = []
    for factor in free_factors(query):
        generated = histogram(read_factor(images, factor), factor)
        truth = conditional(labels, query, factor)
        rows.append(
            {
                **_query_columns(query),
                "factor": FACTOR_NAMES[factor],
                "value": total_variation(generated, truth),
                "n": int(images.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def calibration_histogram(
    images: torch.Tensor, query: torch.Tensor, labels: torch.Tensor
) -> pd.DataFrame:
    """The two distributions behind each `calibration` row, for the diagnostic plot."""
    rows = []
    for factor in free_factors(query):
        generated = histogram(read_factor(images, factor), factor)
        truth = conditional(labels, query, factor)
        for value in range(CARDINALITIES[factor]):
            rows.append(
                {
                    **_query_columns(query),
                    "factor": FACTOR_NAMES[factor],
                    "colour": value,
                    "generated": float(generated[value]),
                    "truth": float(truth[value]),
                    "n": int(images.shape[0]),
                }
            )
    return pd.DataFrame(rows)


# Which sampler produced a set of images, the way `source` names it for a pool.
MIXTURE = "mixture"
MARGINALIZED = "marginalized"

CALIBRATION_FILENAME = "colour_calibration.csv"
HISTOGRAM_FILENAME = "colour_calibration_histogram.csv"


@torch.no_grad()
def mixture_samples(
    model: object,
    vae: object,
    query: torch.Tensor,
    n: int,
    labels: torch.Tensor,
    device: torch.device,
    std_correction: float = 1.0,
    batch_size: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The reference arm: draw the free factors from training, then ask for a full label.

    p(z | given) = sum_free p(free | given) p(z | given, free), sampled ancestrally. The
    model answers only fully specified questions, so this is correct whenever the model
    can render each combination -- which is what makes it the floor.
    """
    drawn = sample_labels(query, n, labels, generator)
    _, images = sample_and_decode(
        model, vae, drawn, device, std_correction=std_correction, batch_size=batch_size
    )
    return images


@torch.no_grad()
def marginalized_samples(
    model: object,
    vae: object,
    query: torch.Tensor,
    n: int,
    device: torch.device,
    std_correction: float = 1.0,
    batch_size: int = 256,
) -> torch.Tensor:
    """The model marginalizing the free factors itself, for a PC that has a `p(y)`."""
    known = {
        factor: int(value)
        for factor, value in enumerate(query.tolist())
        if value != UNSPECIFIED
    }
    images = []
    remaining = n
    while remaining > 0:
        size = min(batch_size, remaining)
        latents, _ = model.sample_partial_labels(
            known, size, device=device, std_correction=std_correction
        )
        images.append(to_uint8(vae.decode(latents)).cpu())
        remaining -= size
    return torch.cat(images)


def score(
    images: torch.Tensor, query: torch.Tensor, labels: torch.Tensor
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both tables for one arm of one query: the distance, and the two histograms."""
    pixels = to_float(images)
    return (
        calibration(pixels, query, labels),
        calibration_histogram(pixels, query, labels),
    )


def run_columns(cfg: EvaluationRunConfig, model: str, vae: str, seed: int) -> dict:
    """The identity columns, as a pool's metrics carry them. No `classifier`: nothing
    here runs the judge, the colours are read straight off the pixels."""
    return {
        "model": cfg.model.model_type,
        "dataset": cfg.dataset.name,
        "checkpoint": model,
        "seed": seed,
        "std_correction": cfg.generation.std_correction,
        "vae": vae,
    }


def run_marginal(cfg: EvaluationRunConfig, device: torch.device) -> None:
    """Every query, every arm the model supports, into two CSVs under `results_root`."""
    if cfg.marginal is None:
        raise ValueError(
            "this config has no `marginal:` block, so there are no queries to run"
        )

    model, resolved_model, model_path = load_generative_model(
        cfg.model.model_type, cfg.model.name, device, cfg.model.tag
    )
    name, tag_, external = resolve_autoencoder(
        cfg.autoencoder, model_path, resolved_model
    )
    vae, resolved_vae = load_vae(name, tag_, external, cfg.dataset, device)
    seed = seed_everything(cfg.generation.seed)
    generator = torch.Generator().manual_seed(seed)
    labels = training_labels(cfg.dataset.name)
    check_latent_dim(model, vae, device, resolved_model, resolved_vae, labels)
    columns = run_columns(cfg, resolved_model, resolved_vae, seed)
    keys = {key: columns[key] for key in RUN_KEYS}

    n = cfg.marginal.n_per_query
    batch_size = cfg.generation.batch_size
    std_correction = cfg.generation.std_correction

    arms_per_query = 2 if isinstance(model, JointPC) else 1
    rtpt = start_rtpt(
        f"marginal_{cfg.dataset.name}", len(cfg.marginal.queries) * arms_per_query
    )

    distances: list[pd.DataFrame] = []
    histograms: list[pd.DataFrame] = []
    for spec in cfg.marginal.queries:
        query = as_query(spec)
        arms = {
            MIXTURE: mixture_samples(
                model,
                vae,
                query,
                n,
                labels,
                device,
                std_correction=std_correction,
                batch_size=batch_size,
                generator=generator,
            )
        }
        if isinstance(model, JointPC):
            arms[MARGINALIZED] = marginalized_samples(
                model,
                vae,
                query,
                n,
                device,
                std_correction=std_correction,
                batch_size=batch_size,
            )
        for arm, images in arms.items():
            rtpt.step(subtitle=f"{arm} {spec}")
            distance, histogram_table = score(images, query, labels)
            distances.append(tag(distance, columns, arm))
            histograms.append(tag(histogram_table, columns, arm))

    results_root = cfg.evaluation.results_root
    distance_table = pd.concat(distances, ignore_index=True)
    write_metric(results_root / CALIBRATION_FILENAME, distance_table, keys)
    write_metric(
        results_root / HISTOGRAM_FILENAME,
        pd.concat(histograms, ignore_index=True),
        keys,
    )
    _print_summary(distance_table)


def _print_summary(distances: pd.DataFrame) -> None:
    print("\ncolour calibration (total variation, lower is better)")
    for row in distances.itertuples(index=False):
        query = f"[{row.digit}, {row.fg}, {row.bg}]"
        print(f"  {query:<14} {row.source:<13} {row.factor:<3} {row.value:.4f}")
