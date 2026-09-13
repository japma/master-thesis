"""Does the model obey the label it was given?"""

import numpy as np
import torch

from dataset_loaders.colour_mnist import NUM_BG, NUM_DIGITS, NUM_FG
from evaluation.batch import EvalBatch
from evaluation.classifier import DigitClassifier
from evaluation.harness import Metric, MetricResult, PerCombination
from evaluation.sources import all_combinations


class DigitAccuracy(Metric):
    name = "digit"
    requires = frozenset({"images"})

    def __init__(self, classifier: DigitClassifier) -> None:
        self.classifier = classifier
        self.accumulator = PerCombination()
        self.confusion = np.zeros((NUM_DIGITS, NUM_DIGITS))

    @torch.no_grad()
    def update(self, batch: EvalBatch) -> None:
        assert batch.images is not None
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


class LabelDiscrimination(Metric):
    name = "discrimination"
    requires = frozenset({"latents", "score"})

    def __init__(self) -> None:
        self.accumulator = PerCombination()

    @torch.no_grad()
    def update(self, batch: EvalBatch) -> None:
        assert batch.latents is not None and batch.score is not None
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
