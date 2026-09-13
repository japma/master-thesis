"""How the samples are spread, rather than whether any one of them is right."""

import numpy as np

from evaluation.batch import EvalBatch
from evaluation.harness import Metric, MetricResult, PerCombination


class SampleDiversity(Metric):
    name = "diversity"
    requires = frozenset({"images", "latents"})

    def __init__(self, samples_per_combination: int) -> None:
        if samples_per_combination < 2:
            raise ValueError("diversity needs at least 2 samples per combination")
        self.samples_per_combination = samples_per_combination
        self.accumulator = PerCombination()

    def update(self, batch: EvalBatch) -> None:
        assert batch.images is not None and batch.latents is not None
        per_combination = self.samples_per_combination
        total = batch.images.shape[0]
        if total % per_combination != 0:
            raise ValueError(
                f"batch of {total} is not whole combinations of "
                f"{per_combination} samples"
            )
        groups = total // per_combination

        images = batch.images.reshape(groups, per_combination, -1)
        latents = batch.latents.reshape(groups, per_combination, -1)

        # Spread within a combination, averaged over pixels / latent dims.
        pixel_std = images.std(dim=1).mean(dim=1)
        latent_std = latents.std(dim=1).mean(dim=1)

        self.accumulator.add(
            tuple(i[::per_combination] for i in batch.index),
            pixel_std=pixel_std.cpu().numpy(),
            latent_std=latent_std.cpu().numpy(),
        )

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())


class LatentPlausibility(Metric):
    name = "latent"
    requires = frozenset({"latents"})

    def __init__(self, reference_latents: np.ndarray) -> None:
        self.mean = reference_latents.mean(axis=0)
        covariance = np.cov(reference_latents, rowvar=False)
        self.precision = np.linalg.pinv(covariance)
        self.accumulator = PerCombination()

    def update(self, batch: EvalBatch) -> None:
        assert batch.latents is not None
        centered = batch.latents.cpu().numpy().astype(np.float64) - self.mean
        distance = np.sqrt(
            np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        )
        self.accumulator.add(batch.index, mahalanobis=distance)

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())
