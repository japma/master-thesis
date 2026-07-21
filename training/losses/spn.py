import torch

from training.losses.base import LossOutput


def negative_log_likelihood_loss(outputs: torch.Tensor) -> LossOutput:
    """Negative log-likelihood loss for SPN outputs."""
    nll = -outputs.mean()
    return LossOutput(total=nll)
