"""Where a batch comes from."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from dataset_loaders.colour_mnist import (
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    TABLE_SHAPE,
)
from evaluation.batch import EvalBatch
from models.autoencoder import AbstractAutoencoder

# Anything that yields (images, labels) -- a DataLoader in practice.
LabelledBatches = Iterable[tuple[torch.Tensor, torch.Tensor]]


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


class Source(ABC):
    """Yields `EvalBatch`es and counts what it yielded."""

    name: str
    provides: frozenset[str]

    def __init__(self) -> None:
        self.counts = np.zeros(TABLE_SHAPE)

    @abstractmethod
    def _batches(self) -> Iterator[EvalBatch]: ...

    def batches(self) -> Iterator[EvalBatch]:
        self.counts = np.zeros(TABLE_SHAPE)
        for batch in self._batches():
            np.add.at(self.counts, batch.index, 1.0)
            yield batch


class SampleSource(Source):
    name = "sample"
    provides = frozenset({"images", "latents"})

    def __init__(
        self,
        model,
        ae: AbstractAutoencoder,
        device: torch.device,
        samples_per_combination: int = 64,
        std_correction: float = 1.0,
        combinations_per_chunk: int = 32,
    ) -> None:
        super().__init__()
        self.model = model
        self.ae = ae
        self.device = device
        self.samples_per_combination = samples_per_combination
        self.std_correction = std_correction
        self.combinations_per_chunk = combinations_per_chunk

    def _batches(self) -> Iterator[EvalBatch]:
        self.model.eval()
        self.ae.eval()
        combinations = all_combinations()
        with torch.no_grad():
            for start in range(
                0, combinations.shape[0], self.combinations_per_chunk
            ):
                chunk = combinations[start : start + self.combinations_per_chunk]
                labels = chunk.repeat_interleave(
                    self.samples_per_combination, dim=0
                ).to(self.device)
                latents = self.model.sample(
                    labels, std_correction=self.std_correction
                )
                yield EvalBatch(
                    labels=labels,
                    images=self.ae.decode(latents),
                    latents=latents,
                )


class RealSource(Source):
    name = "real"
    provides = frozenset({"images", "latents"})

    def __init__(
        self,
        ae: AbstractAutoencoder,
        loader: LabelledBatches,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.ae = ae
        self.loader = loader
        self.device = device
        self.max_batches = max_batches

    def _batches(self) -> Iterator[EvalBatch]:
        self.ae.eval()
        with torch.no_grad():
            for step, (images, labels) in enumerate(self.loader):
                if self.max_batches is not None and step >= self.max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()
                yield EvalBatch(
                    labels=labels,
                    images=images,
                    latents=self.ae.encode(images),
                )


class ReconstructionSource(Source):

    name = "reconstruction"
    provides = frozenset({"images", "reference", "latents"})

    def __init__(
        self,
        ae: AbstractAutoencoder,
        loader: LabelledBatches,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.ae = ae
        self.loader = loader
        self.device = device
        self.max_batches = max_batches

    def _batches(self) -> Iterator[EvalBatch]:
        self.ae.eval()
        with torch.no_grad():
            for step, (images, labels) in enumerate(self.loader):
                if self.max_batches is not None and step >= self.max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()
                latents = self.ae.encode(images)
                yield EvalBatch(
                    labels=labels,
                    images=self.ae.decode(latents),
                    reference=images,
                    latents=latents,
                )


class DensitySource(Source):
    name = "density"
    provides = frozenset({"latents", "log_prob", "score"})

    def __init__(
        self,
        model,
        ae: AbstractAutoencoder,
        loader: LabelledBatches,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.ae = ae
        self.loader = loader
        self.device = device
        self.max_batches = max_batches

    def _batches(self) -> Iterator[EvalBatch]:
        self.model.eval()
        self.ae.eval()

        def score(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return self.model(z, labels)

        with torch.no_grad():
            for step, (images, labels) in enumerate(self.loader):
                if self.max_batches is not None and step >= self.max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()
                latents = self.ae.encode(images)
                yield EvalBatch(
                    labels=labels,
                    latents=latents,
                    log_prob=score(latents, labels),
                    score=score,
                )


assert TABLE_SHAPE == (NUM_DIGITS, NUM_FG, NUM_BG)
