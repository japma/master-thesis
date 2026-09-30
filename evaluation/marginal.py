"""Marginalized queries: "a green 5, background unspecified".

A query is a label row with `UNSPECIFIED` where a factor is free. Nothing here scores a
single sample: with a factor free there is no single right answer, only a right *mix*,
so the unit of measurement is the histogram over one query's samples, read by the judge.

Two pieces:
  `sample_labels`  the mixture reference -- free factors drawn from the training
                   conditional, then the model is asked a fully specified question.
                   Correct by construction, so its calibration is the floor.
  `calibration`    how far a set of judged images is from the distribution it should
                   have: each free factor alone, and all of them jointly.
"""

from collections.abc import Sequence

import pandas as pd
import torch

from dataset_loaders.colour_mnist import FACTOR_NAMES
from evaluation.conditionals import (
    UNSPECIFIED,
    cell_index,
    conditional,
    matching,
    num_cells,
    total_variation,
    training_labels,
)
from evaluation.evaluate import RUN_KEYS, load_judge, predict, tag, write_metric
from evaluation.generate import (
    check_latent_dim,
    load_generative_model,
    load_vae,
    resolve_autoencoder,
    sample_and_decode,
)
from evaluation.samples import to_uint8
from models.cspn.joint_pc import JointPC
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.config import EvaluationRunConfig
from utils.progress import start_rtpt
from utils.reproducibility import seed_everything


def as_query(query: Sequence[int]) -> torch.Tensor:
    """Validate a `[digit, fg, bg]` spec and return it as a tensor."""
    row = torch.tensor(list(query), dtype=torch.long)
    if row.shape != (len(FACTOR_NAMES),):
        raise ValueError(f"a query is [digit, fg, bg], got {list(query)}")
    if not free_factors(row):
        raise ValueError(
            f"query {list(query)} specifies every factor -- nothing is marginalized"
        )
    return row


def free_factors(query: torch.Tensor) -> list[int]:
    """Which factors the query leaves to the model."""
    return [f for f, value in enumerate(query.tolist()) if value == UNSPECIFIED]


def scored_groups(query: torch.Tensor) -> list[list[int]]:
    """Each free factor alone, then all of them jointly when there are several.

    The joint row is what catches a model that gets every marginal right but samples
    the free factors independently of each other.
    """
    free = free_factors(query)
    return [[f] for f in free] + ([free] if len(free) > 1 else [])


def group_name(factors: Sequence[int]) -> str:
    return "+".join(FACTOR_NAMES[f] for f in factors)


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


def histogram(predictions: torch.Tensor, factors: Sequence[int]) -> torch.Tensor:
    """Frequency of each cell over `factors`, over a query's judged samples."""
    cells = cell_index(predictions[:, list(factors)], factors)
    counts = torch.bincount(cells, minlength=num_cells(factors))
    return counts / counts.sum()


def _query_columns(query: torch.Tensor) -> dict[str, int]:
    return {name: int(query[f]) for f, name in enumerate(FACTOR_NAMES)}


def calibration(
    predictions: torch.Tensor, query: torch.Tensor, labels: torch.Tensor
) -> pd.DataFrame:
    """One row per scored group: how far the generated mix is from the training one.

    :param predictions: `(N, factors)` the judge's reading of the query's samples.
    """
    rows = []
    for factors in scored_groups(query):
        generated = histogram(predictions, factors)
        truth = conditional(labels, query, factors)
        rows.append(
            {
                **_query_columns(query),
                "factor": group_name(factors),
                "value": total_variation(generated, truth),
                "n": int(predictions.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def calibration_histogram(
    predictions: torch.Tensor, query: torch.Tensor, labels: torch.Tensor
) -> pd.DataFrame:
    """The two distributions behind each `calibration` row, for the diagnostic plot.

    `class` is the cell index over the group's factors, row-major in `factor`'s order.
    """
    rows = []
    for factors in scored_groups(query):
        generated = histogram(predictions, factors)
        truth = conditional(labels, query, factors)
        for cell in range(num_cells(factors)):
            rows.append(
                {
                    **_query_columns(query),
                    "factor": group_name(factors),
                    "class": cell,
                    "generated": float(generated[cell]),
                    "truth": float(truth[cell]),
                    "n": int(predictions.shape[0]),
                }
            )
    return pd.DataFrame(rows)


# Which sampler produced a set of images, the way `source` names it for a pool.
MIXTURE = "mixture"
MARGINALIZED = "marginalized"
DONT_CARE = "dont_care"

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


@torch.no_grad()
def dont_care_samples(
    model: PsiNetCSPN,
    vae: object,
    query: torch.Tensor,
    n: int,
    device: torch.device,
    std_correction: float = 1.0,
    batch_size: int = 256,
) -> torch.Tensor:
    """The learned stand-in: free factors carry the encoder's "unspecified" index.

    Not a marginal. The hypernetwork saw that index during training and learned *a*
    parameter set for it; whether that set matches the true conditional is exactly what
    the calibration number measures.
    """
    if not model.supports_unknown:
        raise ValueError(
            "this CSPN was trained without label dropout, so it has no unknown index; "
            "set encoder_config.label_dropout_prob above 0 and retrain"
        )
    labels = query.clone()
    for factor in free_factors(query):
        labels[factor] = model.unknown_indices[factor]
    rows = labels.unsqueeze(0).repeat(n, 1)
    _, images = sample_and_decode(
        model, vae, rows, device, std_correction=std_correction, batch_size=batch_size
    )
    return images


def run_columns(
    cfg: EvaluationRunConfig, model: str, vae: str, classifier: str, seed: int
) -> dict:
    """The identity columns, as a pool's metrics carry them."""
    return {
        "model": cfg.model.model_type,
        "dataset": cfg.dataset.name,
        "checkpoint": model,
        "seed": seed,
        "std_correction": cfg.generation.std_correction,
        "vae": vae,
        "classifier": classifier,
    }


def run_marginal(cfg: EvaluationRunConfig, device: torch.device) -> None:
    """Every query, every arm the model supports, into two CSVs under `results_root`."""
    if cfg.marginal is None:
        raise ValueError(
            "this config has no `marginal:` block, so there are no queries to run"
        )

    if cfg.classifier is None:
        raise ValueError(
            "this config has no `classifier:`, and every factor is read by the judge"
        )
    if cfg.dataset.labels is not None:
        raise ValueError(
            f"dataset.labels is {list(cfg.dataset.labels)}, but marginal queries "
            "condition on every factor"
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
    judge, classifier = load_judge(cfg.classifier, device)
    labels = training_labels(cfg.dataset.name)
    check_latent_dim(model, vae, device, resolved_model, resolved_vae, labels)
    columns = run_columns(cfg, resolved_model, resolved_vae, classifier, seed)
    keys = {key: columns[key] for key in RUN_KEYS}

    n = cfg.marginal.n_per_query
    batch_size = cfg.generation.batch_size
    std_correction = cfg.generation.std_correction

    arms_per_query = 1 + int(
        isinstance(model, JointPC)
        or (isinstance(model, PsiNetCSPN) and model.supports_unknown)
    )
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
        if isinstance(model, PsiNetCSPN) and model.supports_unknown:
            arms[DONT_CARE] = dont_care_samples(
                model,
                vae,
                query,
                n,
                device,
                std_correction=std_correction,
                batch_size=batch_size,
            )
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
            predictions = predict(
                judge, images, device, cfg.evaluation.batch_size, f"judging {arm}"
            )
            distances.append(tag(calibration(predictions, query, labels), columns, arm))
            histograms.append(
                tag(calibration_histogram(predictions, query, labels), columns, arm)
            )

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
        print(f"  {query:<14} {row.source:<13} {row.factor:<13} {row.value:.4f}")
