import torch

from training.objectives.base import StepOutput


class MetricsCollector:
    def __init__(self) -> None:
        self._weighted_sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def update(self, step_output: StepOutput) -> None:
        bs = float(step_output.batch_size)
        for key, value in step_output.metrics.items():
            if key not in self._weighted_sums:
                self._weighted_sums[key] = torch.zeros_like(value)
            self._weighted_sums[key] = self._weighted_sums[key] + value * bs
            self._counts[key] = self._counts.get(key, 0) + int(bs)

    def compute_average_metrics(self) -> dict[str, float]:
        if not self._counts:
            return {}

        return {
            key: (self._weighted_sums[key] / self._counts[key]).item()
            for key in self._counts
        }

    def reset(self) -> None:
        self._weighted_sums = {}
        self._counts = {}
