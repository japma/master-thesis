"""What a latent actually encodes, and where.

Three questions, three different probes -- an autoencoder can pass one and fail the
others, and only the three together support "this factor lives in these dimensions":

  sufficiency   is the factor in the latent at all           probe every dimension
  locality      is it in the block it was assigned           probe that block alone
  exclusivity   is it *only* there                           probe every other dimension

Linear probes throughout, because "linearly decodable from its own block" is the property
a supervised block is trained for and the one a circuit downstream can exploit; the MLP
probe is reported beside it to separate "not represented" from "represented nonlinearly".
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models.autoencoder import (
    AbstractAutoencoder,
    SupervisedVAE,
    VariationalAutoencoder,
)

# Per-dimension search and MI estimation are the slow parts; they are also the least
# sensitive to sample size, so they run on a subsample of the probe's training split.
SEARCH_SUBSAMPLE: int = 5000


@dataclass
class FactorProbe:
    """One label factor's answers to the three questions above."""

    name: str
    block: tuple[int, int]
    majority: float
    sufficiency_linear: float
    sufficiency_mlp: float
    locality_linear: float
    exclusivity_linear: float
    exclusivity_mlp: float
    best_dim: int
    best_dim_linear: float
    mig: float


@dataclass
class LatentReport:
    factors: list[FactorProbe]
    kl_per_dim: np.ndarray
    reconstruction: dict[str, float] = field(default_factory=dict)

    @property
    def active_units(self) -> int:
        """Dimensions carrying more than 0.01 nats -- the usual posterior-collapse count."""
        return int((self.kl_per_dim > 0.01).sum())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "factor": probe.name,
                    "block": f"{probe.block[0]}:{probe.block[1]}",
                    "majority": probe.majority,
                    "sufficiency (linear, all dims)": probe.sufficiency_linear,
                    "sufficiency (MLP, all dims)": probe.sufficiency_mlp,
                    "locality (linear, own block)": probe.locality_linear,
                    "leakage (linear, other dims)": probe.exclusivity_linear,
                    "leakage (MLP, other dims)": probe.exclusivity_mlp,
                    "best single dim": probe.best_dim,
                    "best single dim acc": probe.best_dim_linear,
                    "MIG": probe.mig,
                }
                for probe in self.factors
            ]
        ).set_index("factor")


def blocks_for(
    ae: AbstractAutoencoder,
    fallback: Sequence[slice],
    fallback_names: Sequence[str],
) -> tuple[list[slice], list[str]]:
    """A supervised model names its own blocks; anything else is probed on `fallback`.

    Probing an unsupervised latent on the *same* dimension ranges is the point: the
    supervised model's blocks only mean something relative to what those dimensions
    happened to hold before.
    """
    if isinstance(ae, SupervisedVAE):
        return ae.supervision.slices(), ae.supervision.factor_names
    return list(fallback), list(fallback_names)


@torch.no_grad()
def encode_dataset(
    ae: AbstractAutoencoder,
    loader: DataLoader,
    device: torch.device,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior means, per-dimension KL, and targets for a whole split."""
    ae.eval()
    means, kls, targets = [], [], []
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        if isinstance(ae, VariationalAutoencoder):
            mu, log_var = ae.encode_distribution(images)
            kls.append(
                (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).cpu().numpy()
            )
        else:
            mu = ae.encode(images)
        means.append(mu.cpu().numpy())
        targets.append(labels.numpy())
        seen += images.shape[0]
        if limit is not None and seen >= limit:
            break

    mu = np.concatenate(means)[:limit]
    labels = np.concatenate(targets)[:limit]
    kl = np.concatenate(kls)[:limit].mean(axis=0) if kls else np.zeros(mu.shape[1])
    return mu, kl, labels


def _accuracy(estimator, train_x, train_y, test_x, test_y) -> float:
    if train_x.shape[1] == 0:
        return float("nan")
    estimator.fit(train_x, train_y)
    return float(estimator.score(test_x, test_y))


def _linear(seed: int):
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    )


def _mlp(seed: int):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        ),
    )


def _mig(latents: np.ndarray, targets: np.ndarray, seed: int) -> float:
    """Gap between the two most informative dimensions, over the factor's entropy.

    1.0 means one dimension carries everything the latent knows about the factor; 0.0
    means the top two are equally informative, i.e. the factor is spread across axes.
    """
    information = mutual_info_classif(latents, targets, random_state=seed)
    top = np.sort(information)[::-1]
    frequencies = np.bincount(targets) / targets.shape[0]
    frequencies = frequencies[frequencies > 0]
    entropy = float(-(frequencies * np.log(frequencies)).sum())
    if entropy <= 0 or top.shape[0] < 2:
        return float("nan")
    return float((top[0] - top[1]) / entropy)


def probe_factor(
    name: str,
    block: slice,
    train_latents: np.ndarray,
    train_targets: np.ndarray,
    test_latents: np.ndarray,
    test_targets: np.ndarray,
    seed: int = 0,
) -> FactorProbe:
    dimensions = train_latents.shape[1]
    inside = np.zeros(dimensions, dtype=bool)
    inside[block] = True

    subsample = min(SEARCH_SUBSAMPLE, train_latents.shape[0])
    per_dim = [
        _accuracy(
            _linear(seed),
            train_latents[:subsample, [dim]],
            train_targets[:subsample],
            test_latents[:, [dim]],
            test_targets,
        )
        for dim in range(dimensions)
    ]
    best_dim = int(np.argmax(per_dim))

    frequencies = np.bincount(test_targets) / test_targets.shape[0]

    return FactorProbe(
        name=name,
        block=(int(np.flatnonzero(inside)[0]), int(np.flatnonzero(inside)[-1]) + 1),
        majority=float(frequencies.max()),
        sufficiency_linear=_accuracy(
            _linear(seed), train_latents, train_targets, test_latents, test_targets
        ),
        sufficiency_mlp=_accuracy(
            _mlp(seed), train_latents, train_targets, test_latents, test_targets
        ),
        locality_linear=_accuracy(
            _linear(seed),
            train_latents[:, inside],
            train_targets,
            test_latents[:, inside],
            test_targets,
        ),
        exclusivity_linear=_accuracy(
            _linear(seed),
            train_latents[:, ~inside],
            train_targets,
            test_latents[:, ~inside],
            test_targets,
        ),
        exclusivity_mlp=_accuracy(
            _mlp(seed),
            train_latents[:, ~inside],
            train_targets,
            test_latents[:, ~inside],
            test_targets,
        ),
        best_dim=best_dim,
        best_dim_linear=float(per_dim[best_dim]),
        mig=_mig(
            train_latents[:subsample], train_targets[:subsample], seed
        ),
    )


def probe_latents(
    train_latents: np.ndarray,
    train_targets: np.ndarray,
    test_latents: np.ndarray,
    test_targets: np.ndarray,
    blocks: Sequence[slice],
    names: Sequence[str],
    kl_per_dim: np.ndarray | None = None,
    seed: int = 0,
) -> LatentReport:
    """One `FactorProbe` per label factor, in target-vector order."""
    if train_targets.ndim == 1:
        train_targets = train_targets[:, None]
        test_targets = test_targets[:, None]
    if train_targets.shape[1] != len(blocks):
        raise ValueError(
            f"{len(blocks)} blocks named but the targets carry "
            f"{train_targets.shape[1]} factors per sample"
        )

    factors = [
        probe_factor(
            name,
            block,
            train_latents,
            train_targets[:, i],
            test_latents,
            test_targets[:, i],
            seed=seed,
        )
        for i, (name, block) in enumerate(zip(names, blocks, strict=True))
    ]
    return LatentReport(
        factors=factors,
        kl_per_dim=(
            kl_per_dim
            if kl_per_dim is not None
            else np.zeros(train_latents.shape[1])
        ),
    )
