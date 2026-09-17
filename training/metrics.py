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


class PerClassAccuracy:
    """Correct/seen counts per class, so a judge that is blind to one digit cannot
    hide behind a good overall number."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        hits = (predictions == targets).float().cpu()
        targets = targets.cpu()
        self._correct.index_add_(0, targets, hits)
        self._seen.index_add_(0, targets, torch.ones_like(hits))

    def reset(self) -> None:
        self._correct = torch.zeros(self.num_classes)
        self._seen = torch.zeros(self.num_classes)

    @property
    def overall(self) -> float:
        return float(self._correct.sum() / self._seen.sum().clamp(min=1.0))

    @property
    def per_class(self) -> torch.Tensor:
        """NaN for a class that never appeared, rather than a misleading zero."""
        return torch.where(
            self._seen > 0, self._correct / self._seen.clamp(min=1.0), torch.nan
        )
