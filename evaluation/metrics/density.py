"""What the model says about real data, rather than what it draws."""

from evaluation.batch import EvalBatch
from evaluation.harness import Metric, MetricResult, PerCombination


class NegativeLogLikelihood(Metric):
    name = "nll"
    requires = frozenset({"log_prob"})

    def __init__(self) -> None:
        self.accumulator = PerCombination()

    def update(self, batch: EvalBatch) -> None:
        assert batch.log_prob is not None
        self.accumulator.add(
            batch.index, value=-batch.log_prob.double().cpu().numpy()
        )

    def compute(self) -> MetricResult:
        return MetricResult(name=self.name, tables=self.accumulator.tables())
