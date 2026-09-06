"""The metrics the harness fans samples and densities out to.

Each is independent of every other and of any particular model -- they see tensors,
not circuits -- so a new one is a class here and a line in the suite.
"""

import numpy as np
import torch

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from evaluation.classifier import DigitClassifier
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.harness import (
    DensityBatch,
    DensityMetric,
    MetricResult,
    PerCombination,
    SampleBatch,
    SampleMetric,
    all_combinations,
)


class ColourFidelity(SampleMetric):
    """Do generated images carry the foreground and background colour they were asked
    for? `contrast` separates a wrongly-coloured digit from a flat image with no digit
    at all, which would otherwise both read as a colour miss."""

    name = "colour"

    def __init__(self) -> None:
        self.accumulator = PerCombination()

    def update(self, batch: SampleBatch) -> None:
        images = batch.images
        labels = batch.labels.cpu().numpy()

        generated_bg = border_colour(images)
        # No source image to locate the digit, so the generated image's own contrast
        # against its border has to do it -- which is why contrast is reported too.
        generated_fg = foreground_colour(images, images)

        target_bg = torch.tensor(
            BG_PALETTE[labels[:, 2]], dtype=images.dtype, device=images.device
        )
        target_fg = torch.tensor(
            FG_PALETTE[labels[:, 1]], dtype=images.dtype, device=images.device
        )

        self.accumulator.add(
            batch.index,
            bg_accuracy=nearest_palette_index(generated_bg, BG_PALETTE) == labels[:, 2],
            fg_accuracy=nearest_palette_index(generated_fg, FG_PALETTE) == labels[:, 1],
            bg_drift=(generated_bg - target_bg).norm(dim=1).cpu().numpy(),
            fg_drift=(generated_fg - target_fg).norm(dim=1).cpu().numpy(),
            contrast=(generated_fg - generated_bg).norm(dim=1).cpu().numpy(),
        )

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())


class DigitAccuracy(SampleMetric):
    """Is the generated digit the one that was asked for?

    The colour metrics cannot see this, and it is where a model that only predicts a
    conditional mean is expected to fail: it lands on the centroid of a combination's
    latent cluster, which decodes to the right colours around a digit-shaped blur.
    """

    name = "digit"

    def __init__(self, classifier: DigitClassifier) -> None:
        self.classifier = classifier
        self.accumulator = PerCombination()
        self.confusion = np.zeros((NUM_DIGITS, NUM_DIGITS))

    @torch.no_grad()
    def update(self, batch: SampleBatch) -> None:
        self.classifier.eval()
        logits = self.classifier(batch.images)
        predicted = logits.argmax(dim=1)
        labels = batch.labels.cpu().numpy()

        probabilities = logits.softmax(dim=1)
        # A confident judge on a real-looking digit; near log 10 on a blur.
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)

        np.add.at(self.confusion, (labels[:, 0], predicted.cpu().numpy()), 1.0)
        self.accumulator.add(
            batch.index,
            accuracy=(predicted.cpu().numpy() == labels[:, 0]),
            confidence=probabilities.max(dim=1).values.cpu().numpy(),
            entropy=entropy.cpu().numpy(),
        )

    def compute(self) -> MetricResult:
        return MetricResult(
            name=self.name,
            tables=self.accumulator.tables(),
            scalars={
                # How much of the digit signal survives at all, ignoring which digit:
                # a model emitting one digit for everything scores ~0.1 above.
                "predicted_digit_entropy": float(
                    _entropy(self.confusion.sum(axis=0)) / np.log(NUM_DIGITS)
                ),
            },
        )


class SampleDiversity(SampleMetric):
    """How much do repeated samples for the *same* combination differ?

    Scores exactly zero for a point predictor, which is the honest way to show what a
    deterministic baseline gives up: it can score perfectly on colour and digit while
    emitting one image per combination.
    """

    name = "diversity"

    def __init__(self, samples_per_combination: int) -> None:
        if samples_per_combination < 2:
            raise ValueError("diversity needs at least 2 samples per combination")
        self.samples_per_combination = samples_per_combination
        self.accumulator = PerCombination()

    def update(self, batch: SampleBatch) -> None:
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


class LatentPlausibility(SampleMetric):
    """Do generated latents land where the autoencoder's real latents live?

    A model can satisfy every colour check while drifting into a latent region the
    decoder was never trained on; this separates "wrong sample" from "off-manifold".
    """

    name = "latent"

    def __init__(self, reference_latents: np.ndarray) -> None:
        self.mean = reference_latents.mean(axis=0)
        covariance = np.cov(reference_latents, rowvar=False)
        self.precision = np.linalg.pinv(covariance)
        self.accumulator = PerCombination()

    def update(self, batch: SampleBatch) -> None:
        centered = batch.latents.cpu().numpy().astype(np.float64) - self.mean
        distance = np.sqrt(
            np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        )
        self.accumulator.add(batch.index, mahalanobis=distance)

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())


class NegativeLogLikelihood(DensityMetric):
    """Mean log p(z | y) on real data -- the one number every model reports on the same
    scale, since the CSPN's normalization Jacobian puts it in raw-latent space too."""

    name = "nll"

    def __init__(self) -> None:
        self.accumulator = PerCombination()

    def update(self, batch: DensityBatch) -> None:
        self.accumulator.add(
            batch.index, value=-batch.log_prob.double().cpu().numpy()
        )

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())


class LabelDiscrimination(DensityMetric):
    """Given a real latent, does the model's own density prefer the right label?

    Scores every one of the 180 combinations and reads the argmax, then the per-factor
    argmax of the induced posterior. This needs no classifier and no samples: it asks
    whether the conditional density is actually conditioned on anything. A model whose
    p(z | y) barely moves with y scores at chance here however good its samples look.
    """

    name = "discrimination"

    def __init__(self) -> None:
        self.accumulator = PerCombination()

    @torch.no_grad()
    def update(self, batch: DensityBatch) -> None:
        combinations = all_combinations().to(batch.latents.device)
        scores = torch.stack(
            [
                batch.score(batch.latents, combo.expand(batch.latents.shape[0], -1))
                for combo in combinations
            ],
            dim=1,
        )

        labels = batch.labels.cpu().numpy()
        best = combinations[scores.argmax(dim=1)].cpu().numpy()

        # Uniform prior over combinations, so the posterior is the softmax of the
        # scores; marginalize it to ask about one factor at a time.
        posterior = scores.softmax(dim=1).reshape(-1, NUM_DIGITS, NUM_FG, NUM_BG)
        digit = posterior.sum(dim=(2, 3)).argmax(dim=1).cpu().numpy()
        fg = posterior.sum(dim=(1, 3)).argmax(dim=1).cpu().numpy()
        bg = posterior.sum(dim=(1, 2)).argmax(dim=1).cpu().numpy()

        self.accumulator.add(
            batch.index,
            joint_accuracy=(best == labels).all(axis=1),
            digit_accuracy=(digit == labels[:, 0]),
            fg_accuracy=(fg == labels[:, 1]),
            bg_accuracy=(bg == labels[:, 2]),
        )

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts / total
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log(nonzero)).sum())
