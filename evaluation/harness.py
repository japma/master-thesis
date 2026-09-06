"""Sample once, score many times.

Every generation-side number wants the same expensive thing: draw the model's samples
for each (digit, fg, bg) combination and decode them. This module does that pass once
and hands each chunk to every registered metric, so adding a metric costs a class
rather than another 11,520-image sampling run.

Metrics come in two families, because they need different evidence:

  SampleMetric   scores what the model *generates* -- colour, digit, diversity.
  DensityMetric  scores the density the model assigns to *real* held-out data, and can
                 re-score it under other labels (see `DensityBatch.score`).

Both are model-agnostic: anything exposing `sample(labels, std_correction)` and
`forward(z, labels)` works, which is every CSPN, the JointPC, and both neural baselines.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loaders.colour_mnist import (
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    TABLE_SHAPE,
)
from evaluation.aggregate import combination_mean, marginals, weighted_mean
from models.autoencoder import AbstractAutoencoder


def all_combinations() -> torch.Tensor:
    """Every (digit, fg, bg) triple, in the canonical table order."""
    return torch.tensor(
        [
            [digit, fg, bg]
            for digit in range(NUM_DIGITS)
            for fg in range(NUM_FG)
            for bg in range(NUM_BG)
        ],
        dtype=torch.long,
    )


@dataclass
class SampleBatch:
    """Generated samples for whole combinations -- never a partial one, so metrics that
    compare samples within a combination (diversity) can work chunk by chunk."""

    images: torch.Tensor
    latents: torch.Tensor
    labels: torch.Tensor

    @property
    def index(self) -> tuple[np.ndarray, ...]:
        labels = self.labels.cpu().numpy()
        return (labels[:, 0], labels[:, 1], labels[:, 2])


@dataclass
class DensityBatch:
    """Real encoded data plus the model's density on it.

    `score(z, labels)` re-evaluates the model's log-density under arbitrary labels,
    which is what a metric needs to ask "does this model's density prefer the right
    label?" without knowing anything about the model.
    """

    latents: torch.Tensor
    labels: torch.Tensor
    log_prob: torch.Tensor
    score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @property
    def index(self) -> tuple[np.ndarray, ...]:
        labels = self.labels.cpu().numpy()
        return (labels[:, 0], labels[:, 1], labels[:, 2])


@dataclass
class MetricResult:
    name: str
    # Per-(digit, fg, bg) tables, shape (10, 6, 3), NaN where a cell has no samples.
    tables: dict[str, np.ndarray] = field(default_factory=dict)
    # Numbers that do not decompose per combination.
    scalars: dict[str, float] = field(default_factory=dict)


class SampleMetric(ABC):
    """Scores generated samples. `update` sees whole combinations at a time."""

    name: str

    @abstractmethod
    def update(self, batch: SampleBatch) -> None: ...

    @abstractmethod
    def compute(self) -> MetricResult: ...


class DensityMetric(ABC):
    """Scores the density the model assigns to real data."""

    name: str

    @abstractmethod
    def update(self, batch: DensityBatch) -> None: ...

    @abstractmethod
    def compute(self) -> MetricResult: ...


class PerCombination:
    """Accumulates per-image values into a per-combination table across chunks."""

    def __init__(self) -> None:
        self.values: dict[str, list[np.ndarray]] = {}
        self.index: list[np.ndarray] = []

    def add(self, index: tuple[np.ndarray, ...], **values: np.ndarray) -> None:
        if not self.index:
            self.index = [np.array([]) for _ in index]
        self.index = [
            np.concatenate([existing, new]) for existing, new in zip(
                self.index, index, strict=True
            )
        ]
        for key, value in values.items():
            self.values.setdefault(key, []).append(np.asarray(value, dtype=np.float64))

    def counts(self) -> np.ndarray:
        counts = np.zeros(TABLE_SHAPE)
        if self.index:
            np.add.at(counts, tuple(i.astype(np.int64) for i in self.index), 1.0)
        return counts

    def tables(self) -> dict[str, np.ndarray]:
        counts = self.counts()
        index = tuple(i.astype(np.int64) for i in self.index)
        return {
            key: combination_mean(np.concatenate(value), index, counts)
            for key, value in self.values.items()
        }


@dataclass
class EvalReport:
    """Everything the suite produced, plus the reductions a results section quotes."""

    tables: dict[str, np.ndarray]
    scalars: dict[str, float]
    counts: np.ndarray
    seen: np.ndarray | None = None

    def overall(self, name: str) -> float:
        return weighted_mean(self.tables[name], np.ones(TABLE_SHAPE), self.counts)

    def split(self, name: str) -> tuple[float, float]:
        """(trained, held-out) means for a table, using the train-split combination mask.

        The held-out half is NaN when the variant has no held-out combinations.
        """
        if self.seen is None:
            raise ValueError("no seen mask: this report cannot be split")
        table = self.tables[name]
        return (
            weighted_mean(table, self.seen, self.counts),
            weighted_mean(table, ~self.seen, self.counts),
        )

    def marginals(self, name: str) -> dict[str, np.ndarray]:
        return marginals(self.tables[name], self.counts)

    def summary(self) -> dict[str, float]:
        """Flat scalar view, suitable for wandb."""
        out = dict(self.scalars)
        for name in self.tables:
            out[name] = self.overall(name)
            if self.seen is not None:
                trained, held_out = self.split(name)
                out[f"{name}/trained"] = trained
                out[f"{name}/held_out"] = held_out
        return out


@torch.no_grad()
def run_sample_metrics(
    model,
    ae: AbstractAutoencoder,
    metrics: Sequence[SampleMetric],
    device: torch.device,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    combinations_per_chunk: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    """One sampling pass over every combination, fanned out to each metric."""
    model.eval()
    ae.eval()

    combinations = all_combinations()
    counts = np.zeros(TABLE_SHAPE)

    for start in range(0, combinations.shape[0], combinations_per_chunk):
        chunk = combinations[start : start + combinations_per_chunk]
        labels = chunk.repeat_interleave(samples_per_combination, dim=0).to(device)

        latents = model.sample(labels, std_correction=std_correction)
        images = ae.decode(latents)

        batch = SampleBatch(images=images, latents=latents, labels=labels)
        np.add.at(counts, batch.index, 1.0)
        for metric in metrics:
            metric.update(batch)

    return (*_collect(metrics), counts)


@torch.no_grad()
def run_density_metrics(
    model,
    ae: AbstractAutoencoder,
    metrics: Sequence[DensityMetric],
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    """One pass over real data, scoring the model's density on it."""
    model.eval()
    ae.eval()

    def score(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return model(z, labels)

    counts = np.zeros(TABLE_SHAPE)

    for step, (images, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        latents = ae.encode(images)

        batch = DensityBatch(
            latents=latents,
            labels=labels,
            log_prob=score(latents, labels),
            score=score,
        )
        np.add.at(counts, batch.index, 1.0)
        for metric in metrics:
            metric.update(batch)

    return (*_collect(metrics), counts)


def _collect(
    metrics: Iterable[SampleMetric | DensityMetric],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    tables: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}
    for metric in metrics:
        result = metric.compute()
        for key, table in result.tables.items():
            tables[f"{result.name}/{key}"] = table
        for key, value in result.scalars.items():
            scalars[f"{result.name}/{key}"] = value
    return tables, scalars


def run_eval_suite(
    model,
    ae: AbstractAutoencoder,
    device: torch.device,
    sample_metrics: Sequence[SampleMetric] = (),
    density_metrics: Sequence[DensityMetric] = (),
    loader: DataLoader | None = None,
    seen: np.ndarray | None = None,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    max_density_batches: int | None = None,
) -> EvalReport:
    """Run both families and merge them into one report.

    Counts come from the sampling pass when there is one -- that is what the generated
    tables are averages over -- and from the data pass otherwise.
    """
    tables: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}
    counts: np.ndarray | None = None

    if sample_metrics:
        sample_tables, sample_scalars, counts = run_sample_metrics(
            model,
            ae,
            sample_metrics,
            device,
            samples_per_combination=samples_per_combination,
            std_correction=std_correction,
        )
        tables |= sample_tables
        scalars |= sample_scalars

    if density_metrics:
        if loader is None:
            raise ValueError("density metrics need a `loader` of real data")
        density_tables, density_scalars, density_counts = run_density_metrics(
            model, ae, density_metrics, loader, device, max_batches=max_density_batches
        )
        tables |= density_tables
        scalars |= density_scalars
        if counts is None:
            counts = density_counts

    if counts is None:
        raise ValueError("no metrics given")

    return EvalReport(tables=tables, scalars=scalars, counts=counts, seen=seen)


assert TABLE_SHAPE == (NUM_DIGITS, NUM_FG, NUM_BG)
