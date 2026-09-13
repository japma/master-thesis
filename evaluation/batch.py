"""The one batch type every metric consumes."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EvalBatch:
    labels: torch.Tensor
    images: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    log_prob: torch.Tensor | None = None
    # Re-evaluates the model's log-density under arbitrary labels, so a metric can ask
    # "does this density prefer the right label?" without knowing the model.
    score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    FIELDS = ("images", "reference", "latents", "log_prob", "score")

    @property
    def index(self) -> tuple[np.ndarray, ...]:
        labels = self.labels.cpu().numpy()
        return (labels[:, 0], labels[:, 1], labels[:, 2])

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    def provides(self) -> frozenset[str]:
        return frozenset(
            name for name in self.FIELDS if getattr(self, name) is not None
        )
