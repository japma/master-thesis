from training.objectives.base import StepOutput
import torch


class MetricsCollector:
    def __init__(self) -> None:
        self._weighted_sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, step_output: StepOutput | torch.Tensor) -> None:
        """
        Accept either a StepOutput or a torch.Tensor (interpreted as 'loss').
        If a Tensor is provided it's recorded under 'loss' with batch_size=1.
        """
        # Normalize input to a metrics dict and a batch size
        if isinstance(step_output, torch.Tensor):
            bs = 1.0
            metrics = {"loss": step_output}
        else:
            bs = float(step_output.batch_size)
            metrics = step_output.metrics

        for key, value in metrics.items():
            try:
                metric_value = float(value.detach().cpu().item())
            except Exception:
                try:
                    metric_value = float(value.detach().mean().cpu().item())
                except Exception:
                    metric_value = float(value.detach().item())

            self._weighted_sums[key] = self._weighted_sums.get(key, 0.0) + (
                metric_value * bs
            )
            self._counts[key] = self._counts.get(key, 0) + int(bs)

    def compute_average_metrics(self) -> dict[str, float]:
        if not self._counts:
            return {}

        return {
            key: self._weighted_sums[key] / self._counts[key] for key in self._counts
        }

    def reset(self) -> None:
        self._weighted_sums = {}
        self._counts = {}
