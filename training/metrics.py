from collections import defaultdict

from training.objectives.base import StepOutput


class MetricsCollector:
    def __init__(self) -> None:
        self.step_outputs: list[StepOutput] = []

    def update(self, step_output: StepOutput) -> None:
        self.step_outputs.append(step_output)

    def compute_average_metrics(self) -> dict[str, float]:
        if not self.step_outputs:
            return {}

        weighted_sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)

        for step_output in self.step_outputs:
            bs = step_output.batch_size
            for key, value in step_output.metrics.items():
                weighted_sums[key] += value * bs
                counts[key] += bs

        return {key: weighted_sums[key] / counts[key] for key in weighted_sums}

    def reset(self) -> None:
        self.step_outputs = []
