import torch

from training.objectives.base import StepOutput


class MetricsCollector:
    def __init__(self) -> None:
        self._weighted_sums: dict[str, float | torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def update(self, step_output: StepOutput | torch.Tensor) -> None:
        if isinstance(step_output, torch.Tensor):
            bs = 1.0
            metrics = {"loss": step_output}
        else:
            bs = float(step_output.batch_size)
            metrics = step_output.metrics

        for key, value in metrics.items():
            detached = value.detach()
            if detached.numel() == 1:
                update_value: float | torch.Tensor = float(detached.cpu().item())
            else:
                update_value = detached.cpu()

            prev = self._weighted_sums.get(key)
            if prev is None:
                self._weighted_sums[key] = update_value * bs
            else:
                self._weighted_sums[key] = prev + update_value * bs
            self._counts[key] = self._counts.get(key, 0) + int(bs)

    def compute_average_metrics(self) -> dict[str, float | torch.Tensor]:
        if not self._counts:
            return {}
        return {
            key: self._weighted_sums[key] / self._counts[key] for key in self._counts
        }

    def reset(self) -> None:
        self._weighted_sums = {}
        self._counts = {}
