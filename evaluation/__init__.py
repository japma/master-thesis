"""Everything that produces a number for the results section."""

from evaluation.aggregate import (
    combination_mean,
    latent_mahalanobis,
    marginals,
    per_image_seen,
    weighted_mean,
)
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.generation import GenerationProbe, run_generation_probe
from evaluation.reconstruction import CombinationProbe, run_combination_probe

__all__ = [
    "BG_PALETTE",
    "FG_PALETTE",
    "CombinationProbe",
    "GenerationProbe",
    "border_colour",
    "combination_mean",
    "foreground_colour",
    "latent_mahalanobis",
    "marginals",
    "nearest_palette_index",
    "per_image_seen",
    "run_combination_probe",
    "run_generation_probe",
    "weighted_mean",
]
