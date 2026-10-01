"""Marginalized queries: "a green 5, background unspecified".

A query is a label row with `UNSPECIFIED` where a factor is free. Nothing here scores a
single sample: with a factor free there is no single right answer, only a right *mix*,
so the unit of measurement is the histogram over one query's samples, read by the judge.

Two stages around the pool, like the stratified sets (see `evaluation.pools`):
  `generate_marginal`  every listed model's answer to every query, once per arm it
                       supports, plus the training images matching the query.
  `evaluate_marginal`  the judge reads each set, and `calibration` scores how far its
                       mix is from the training conditional: each free factor alone,
                       and all of them jointly.

The arms, i.e. how a model is made to answer a query with factors left free:
  real          training images matching the query. Not a model: the floor, judge
                error plus sampling noise.
  mixture       free factors drawn from the training labels, then a fully specified
                question. Any conditional model can do it, and it is correct whenever
                every conditional is -- so it checks the conditionals, not a marginal.
  dont_care     a CSPN trained with label dropout, given its "unspecified" index.
  marginalized  the joint PC, marginalizing the free factors exactly.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from dataset_loaders import build_dataset
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
from evaluation.evaluate import (
    SET_KEYS,
    load_judge,
    pinned_version,
    predict,
    tag,
    write_metric,
)
from evaluation.generate import (
    ConditionalSampler,
    VAECache,
    check_latent_dim,
    load_pool_model,
    sample_and_decode,
)
from evaluation.pools import (
    IMAGES,
    LABELS,
    LATENTS,
    MarginalManifest,
    is_complete,
    load_manifest,
    load_tensor,
    marginal_dir,
    model_marginal_dir,
    model_root,
    real_marginal_dir,
    version_number,
    write_dir,
)
from evaluation.samples import current_git_commit, to_uint8
from models.autoencoder import AbstractAutoencoder
from models.cspn.joint_pc import JointPC
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.config import GenerativeModelType, PoolModelConfig, PoolRunConfig
from utils.progress import start_rtpt
from utils.reproducibility import seed_everything
from utils.wandb_utils import resolve_artifact


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


REAL = "real"
MIXTURE = "mixture"
DONT_CARE = "dont_care"
MARGINALIZED = "marginalized"
ARMS = (REAL, MIXTURE, DONT_CARE, MARGINALIZED)

# The conditional models; the others ignore their labels, so no query applies to them.
QUERYABLE = (
    GenerativeModelType.CSPN,
    GenerativeModelType.JOINT_PC,
    GenerativeModelType.NN_BASELINE,
)

# The real set is the same for every model, so it gets one fixed draw.
REAL_SEED = 0

CALIBRATION_FILENAME = "colour_calibration.csv"
HISTOGRAM_FILENAME = "colour_calibration_histogram.csv"


# --- stage 1: sampling ---
def model_arms(model: object) -> list[str]:
    """The arms a model can answer a query by."""
    arms = [MIXTURE]
    if isinstance(model, PsiNetCSPN) and model.supports_unknown:
        arms.append(DONT_CARE)
    if isinstance(model, JointPC):
        arms.append(MARGINALIZED)
    return arms


def real_images(
    dataset: str,
    size: tuple[int, int],
    query: torch.Tensor,
    n: int,
    labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`n` training images matching `query`, drawn with replacement, and their labels."""
    rows = matching(labels, query).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        raise ValueError(f"no training rows match query {query.tolist()}")
    picked = rows[torch.randint(rows.numel(), (n,), generator=generator)]
    split = build_dataset(dataset, train=True, size=size)
    images = torch.stack([split[int(i)][0] for i in picked])
    return to_uint8(images), labels[picked].clone()


@torch.no_grad()
def answer(
    arm: str,
    model: ConditionalSampler,
    vae: AbstractAutoencoder,
    query: torch.Tensor,
    n: int,
    labels: torch.Tensor,
    device: torch.device,
    std_correction: float,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One model's `n` samples for a query by one arm: latents, images, and the labels
    behind them (`UNSPECIFIED` where the arm left a factor to the model)."""
    if arm == MIXTURE:
        drawn = sample_labels(query, n, labels, generator)
        latents, images = sample_and_decode(
            model, vae, drawn, device, std_correction, batch_size
        )
        return latents, images, drawn
    if arm == DONT_CARE:
        assert isinstance(model, PsiNetCSPN)
        asked = query.clone()
        for factor in free_factors(query):
            asked[factor] = model.unknown_indices[factor]
        latents, images = sample_and_decode(
            model, vae, asked.repeat(n, 1), device, std_correction, batch_size
        )
        return latents, images, query.repeat(n, 1)
    if arm == MARGINALIZED:
        assert isinstance(model, JointPC)
        known = {
            factor: int(value)
            for factor, value in enumerate(query.tolist())
            if value != UNSPECIFIED
        }
        latents, images, completed = [], [], []
        for size in torch.arange(n).split(batch_size):
            z, y = model.sample_partial_labels(
                known, len(size), device=device, std_correction=std_correction
            )
            latents.append(z.float().cpu())
            images.append(to_uint8(vae.decode(z)).cpu())
            completed.append(y.cpu())
        return torch.cat(latents), torch.cat(images), torch.cat(completed)
    raise ValueError(f"unknown arm {arm!r}")


def up_to_date(directory: Path, n: int) -> bool:
    return is_complete(directory) and load_manifest(directory, MarginalManifest).n == n


def generate_marginal(cfg: PoolRunConfig, device: torch.device) -> None:
    """Fill `<pool>/marginal/` with every query's real set and every listed model's
    answers, skipping whatever is already there."""
    if cfg.marginal is None:
        raise ValueError("this pool config has no `marginal:` block")
    dataset_dir = cfg.dataset_dir
    n = cfg.marginal.n_per_query
    batch_size = cfg.generation.batch_size
    queries = [as_query(spec) for spec in cfg.marginal.queries]
    labels = training_labels(cfg.dataset.name)
    size = (cfg.dataset.height, cfg.dataset.width)

    generator = torch.Generator().manual_seed(REAL_SEED)
    for query in queries:
        directory = real_marginal_dir(dataset_dir, query.tolist())
        if up_to_date(directory, n):
            continue
        images, real_labels = real_images(
            cfg.dataset.name, size, query, n, labels, generator
        )
        manifest = MarginalManifest(
            dataset=cfg.dataset.name,
            query=query.tolist(),
            arm=REAL,
            n=n,
            git_commit=current_git_commit(),
        )
        write_dir(directory, manifest, {IMAGES: images, LABELS: real_labels})

    entries = [entry for entry in cfg.models if queryable(entry)]
    rtpt = start_rtpt(
        f"marginal_{cfg.dataset.name}",
        len(entries) * len(cfg.generation.seeds) * len(queries),
    )
    vaes: VAECache = {}
    for entry in entries:
        ref = resolve_artifact(entry.name, entry.version or "latest")
        if ref is None:
            print(f"WARNING: {entry.name} is not on wandb; skipping its queries")
            continue
        try:
            model, ref, vae, vae_ref = load_pool_model(
                entry, ref, cfg.dataset, device, vaes
            )
        except ValueError as error:
            print(f"WARNING: {error}")
            continue
        check_latent_dim(model, vae, device, ref, vae_ref, labels)
        arms = model_arms(model)
        for seed in cfg.generation.seeds:
            seed_everything(seed)
            generator = torch.Generator().manual_seed(seed)
            for query in queries:
                rtpt.step(subtitle=f"{entry.name} {query.tolist()}")
                for arm in arms:
                    directory = model_marginal_dir(
                        dataset_dir,
                        seed,
                        entry.type,
                        ref,
                        entry.std_correction,
                        arm,
                        query.tolist(),
                    )
                    if up_to_date(directory, n):
                        continue
                    latents, images, asked = answer(
                        arm,
                        model,
                        vae,
                        query,
                        n,
                        labels,
                        device,
                        entry.std_correction,
                        batch_size,
                        generator,
                    )
                    manifest = MarginalManifest(
                        dataset=cfg.dataset.name,
                        query=query.tolist(),
                        arm=arm,
                        n=n,
                        git_commit=current_git_commit(),
                        model_type=entry.type,
                        model_checkpoint=ref,
                        vae_checkpoint=vae_ref,
                        seed=seed,
                        std_correction=entry.std_correction,
                    )
                    write_dir(
                        directory,
                        manifest,
                        {LATENTS: latents, IMAGES: images, LABELS: asked},
                    )
        del model


def queryable(entry: PoolModelConfig) -> bool:
    """Conditioned on every factor; a query needs all three to leave some free."""
    if entry.type not in QUERYABLE or entry.labels is not None:
        print(f"{entry.name}: not conditioned on every factor; no marginal queries")
        return False
    return True


# --- stage 2: scoring ---
def find_marginal(cfg: PoolRunConfig) -> list[Path]:
    """Every complete set for the config's queries: the real ones, then each listed
    model's pinned version or newest one with anything sampled, per seed and arm."""
    assert cfg.marginal is not None
    dataset_dir = cfg.dataset_dir
    queries = [as_query(spec).tolist() for spec in cfg.marginal.queries]
    found = [real_marginal_dir(dataset_dir, query) for query in queries]
    for seed in cfg.generation.seeds:
        for entry in cfg.models:
            if entry.type not in QUERYABLE or entry.labels is not None:
                continue
            root = model_root(marginal_dir(dataset_dir), seed, entry.type, entry.name)
            version = pinned_version(entry)
            candidates = (
                [root / version]
                if version is not None
                else sorted(root.glob("v*"), key=version_number, reverse=True)
            )
            std = f"std{entry.std_correction:g}"
            sampled = [v for v in candidates if (v / std).is_dir()]
            if not sampled:
                print(f"WARNING: no marginal sets for {entry.name} (seed={seed})")
                continue
            ref = f"{entry.name}:{sampled[0].name}"
            found += [
                model_marginal_dir(
                    dataset_dir, seed, entry.type, ref, entry.std_correction, arm, q
                )
                for arm in ARMS
                for q in queries
            ]
    complete = [directory for directory in found if is_complete(directory)]
    if not complete:
        raise FileNotFoundError(
            f"No marginal sets under {marginal_dir(dataset_dir)}. Run "
            "`uv run generate_pools` with this config first."
        )
    return complete


def identity(manifest: MarginalManifest, classifier: str) -> dict:
    """The columns every pool metric starts with; the real set leaves the model's
    columns empty."""
    return {
        "model": manifest.model_type,
        "dataset": manifest.dataset,
        "checkpoint": manifest.model_checkpoint,
        "seed": manifest.seed,
        "std_correction": manifest.std_correction,
        "vae": manifest.vae_checkpoint,
        "classifier": classifier,
    }


def evaluate_marginal(cfg: PoolRunConfig, device: torch.device) -> None:
    """Judge every marginal set in the pool, and write both calibration CSVs."""
    if cfg.marginal is None:
        raise ValueError("this pool config has no `marginal:` block")
    if cfg.classifier is None:
        raise ValueError("every factor is read by the judge; the config has none")
    directories = find_marginal(cfg)
    labels = training_labels(cfg.dataset.name)
    judge, classifier = load_judge(cfg.classifier, device)
    batch_size = cfg.evaluation.batch_size
    results = cfg.evaluation.results_root
    rtpt = start_rtpt(f"evaluate_marginal_{cfg.dataset.name}", len(directories))

    distances = []
    for directory in directories:
        manifest = load_manifest(directory, MarginalManifest)
        query = torch.tensor(manifest.query)
        rtpt.step(subtitle=f"{manifest.arm} {manifest.query}")
        desc = f"{manifest.model_checkpoint or 'real'} {manifest.arm} {manifest.query}"
        predictions = predict(
            judge, load_tensor(directory, IMAGES), device, batch_size, desc
        )
        columns = identity(manifest, classifier)
        keys = {key: columns[key] for key in SET_KEYS} | {"source": manifest.arm}
        keys |= {name: int(query[f]) for f, name in enumerate(FACTOR_NAMES)}
        distance = tag(calibration(predictions, query, labels), columns, manifest.arm)
        write_metric(results / CALIBRATION_FILENAME, distance, keys)
        write_metric(
            results / HISTOGRAM_FILENAME,
            tag(
                calibration_histogram(predictions, query, labels), columns, manifest.arm
            ),
            keys,
        )
        distances.append(distance)
    _print_summary(pd.concat(distances, ignore_index=True))


def _print_summary(distances: pd.DataFrame) -> None:
    print("\ncolour calibration (total variation, lower is better)")
    for row in distances.itertuples(index=False):
        query = f"[{row.digit}, {row.fg}, {row.bg}]"
        name = row.checkpoint if isinstance(row.checkpoint, str) else "real"
        print(
            f"  {name:<55} {query:<14} {row.source:<13} {row.factor:<12} {row.value:.4f}"
        )
