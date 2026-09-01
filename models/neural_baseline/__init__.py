"""Neural baseline package."""

from utils.config import NeuralBaselineConfig, NeuralBaselineType

from .abstract_nn import AbstractNeuralBaseline
from .deterministic import DeterministicBaseline
from .mixture_density import MixtureDensityBaseline

__all__ = [
    "AbstractNeuralBaseline",
    "DeterministicBaseline",
    "MixtureDensityBaseline",
    "build_neural_baseline",
]


def build_neural_baseline(config: NeuralBaselineConfig) -> AbstractNeuralBaseline:
    """Builds the baseline named by `config.model_type`."""
    match config.model_type:
        case NeuralBaselineType.DETERMINISTIC:
            return DeterministicBaseline(config)
        case NeuralBaselineType.MIXTURE:
            return MixtureDensityBaseline(config)
        case _:
            raise ValueError(f"Unknown neural baseline variant {config.model_type!r}")
