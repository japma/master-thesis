"""Sample once, score many times.

Every generation-side number wants the same expensive thing: draw the model's samples
for each (digit, fg, bg) combination and decode them. A `Source` does that pass once and
hands each chunk to every metric paired with it, so adding a metric costs a class rather
than another 11,520-image sampling run.

A run is a list of `Pass`es -- one source and the metrics that read it. Metrics declare
what evidence they need and sources declare what they produce, so a mismatched pairing
is rejected before any sampling happens rather than failing partway through.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from dataset_loaders.colour_mnist import TABLE_SHAPE
from evaluation.aggregate import combination_mean, marginals, weighted_mean
from evaluation.batch import EvalBatch
from evaluation.sources import (
    DensitySource,
    LabelledBatches,
    SampleSource,
    Source,
    all_combinations,
)
from models.autoencoder import AbstractAutoencoder

__all__ = [
    "EvalBatch",
    "EvalReport",
    "Metric",
    "MetricResult",
    "Pass",
    "PerCombination",
    "all_combinations",
    "run_eval_suite",
    "run_suite",
]


@dataclass
class MetricResult:
    name: str
    # Per-(digit, fg, bg) tables, shape (10, 6, 3), NaN where a cell has no samples.
    tables: dict[str, np.ndarray] = field(default_factory=dict)
    # Numbers that do not decompose per combination.
    scalars: dict[str, float] = field(default_factory=dict)


class Metric(ABC):
    """Scores one kind of evidence. `requires` names the `EvalBatch` fields it reads."""

    name: str
    requires: frozenset[str] = frozenset()

    @abstractmethod
    def update(self, batch: EvalBatch) -> None: ...

    @abstractmethod
    def compute(self) -> MetricResult: ...


@dataclass
class Pass:
    """One source and the metrics that read it.

    `prefix` distinguishes the same metric run over two sources -- the reconstruction
    ceiling and the model's samples produce the same table names otherwise.
    """

    source: Source
    metrics: Sequence[Metric]
    prefix: str = ""

    def validate(self) -> None:
        for metric in self.metrics:
            missing = metric.requires - self.source.provides
            if missing:
                raise ValueError(
                    f"metric {metric.name!r} needs {sorted(missing)}, which the "
                    f"{self.source.name!r} source does not provide"
                )


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


def run_suite(passes: Sequence[Pass], seen: np.ndarray | None = None) -> EvalReport:
    """Run every pass and merge the results into one report."""
    if not passes or all(not p.metrics for p in passes):
        raise ValueError("no metrics given")

    for one in passes:
        one.validate()

    tables: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}
    counts: np.ndarray | None = None

    for one in passes:
        if not one.metrics:
            continue
        for batch in one.source.batches():
            for metric in one.metrics:
                metric.update(batch)
        for metric in one.metrics:
            result = metric.compute()
            for key, table in result.tables.items():
                tables[f"{one.prefix}{result.name}/{key}"] = table
            for key, value in result.scalars.items():
                scalars[f"{one.prefix}{result.name}/{key}"] = value
        if counts is None:
            counts = one.source.counts

    assert counts is not None
    return EvalReport(tables=tables, scalars=scalars, counts=counts, seen=seen)


def run_eval_suite(
    model,
    ae: AbstractAutoencoder,
    device: torch.device,
    sample_metrics: Sequence[Metric] = (),
    density_metrics: Sequence[Metric] = (),
    loader: LabelledBatches | None = None,
    seen: np.ndarray | None = None,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    max_density_batches: int | None = None,
) -> EvalReport:
    """The two-pass run a latent-space model is scored with.

    Counts come from the sampling pass when there is one -- that is what the generated
    tables are averages over -- and from the data pass otherwise.
    """
    passes: list[Pass] = []

    if sample_metrics:
        passes.append(
            Pass(
                source=SampleSource(
                    model,
                    ae,
                    device,
                    samples_per_combination=samples_per_combination,
                    std_correction=std_correction,
                ),
                metrics=sample_metrics,
            )
        )

    if density_metrics:
        if loader is None:
            raise ValueError("density metrics need a `loader` of real data")
        passes.append(
            Pass(
                source=DensitySource(
                    model, ae, loader, device, max_batches=max_density_batches
                ),
                metrics=density_metrics,
            )
        )

    return run_suite(passes, seen=seen)
