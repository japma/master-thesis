"""Colour fidelity of images a model *generates*, not reconstructs.

Kept as a named probe because `probe_cspn` and the notebooks read these fields, but the
work happens in the harness now -- see `evaluation.metrics.ColourFidelity`.
"""

from dataclasses import dataclass

import numpy as np
import torch

from evaluation.harness import run_sample_metrics
from evaluation.metrics import ColourFidelity
from models.autoencoder import AbstractAutoencoder


@dataclass
class GenerationProbe:
    # All shape (10, 6, 3).
    bg_accuracy: np.ndarray
    fg_accuracy: np.ndarray
    bg_drift_table: np.ndarray
    fg_drift_table: np.ndarray
    # Distance between the generated foreground and background colours. Collapses toward
    # zero when the model emits a flat image rather than a wrongly-coloured digit — a
    # different failure from getting the colour wrong, and worth telling apart.
    contrast_table: np.ndarray
    samples_per_combination: int


@torch.no_grad()
def run_generation_probe(
    model,
    ae: AbstractAutoencoder,
    device: torch.device,
    samples_per_combination: int = 64,
    std_correction: float = 1.0,
    combinations_per_chunk: int = 32,
) -> GenerationProbe:
    """Sample every (digit, fg, bg) combination and check the colours."""
    tables, _, _ = run_sample_metrics(
        model,
        ae,
        [ColourFidelity()],
        device,
        samples_per_combination=samples_per_combination,
        std_correction=std_correction,
        combinations_per_chunk=combinations_per_chunk,
    )

    return GenerationProbe(
        bg_accuracy=tables["colour/bg_accuracy"],
        fg_accuracy=tables["colour/fg_accuracy"],
        bg_drift_table=tables["colour/bg_drift"],
        fg_drift_table=tables["colour/fg_drift"],
        contrast_table=tables["colour/contrast"],
        samples_per_combination=samples_per_combination,
    )
