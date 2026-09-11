"""Everything that produces a number for the results section."""

from evaluation.aggregate import (
    combination_mean,
    latent_mahalanobis,
    marginals,
    per_image_seen,
    weighted_mean,
)
from evaluation.classifier import (
    DigitClassifier,
    load_digit_classifier,
    train_digit_classifier,
)
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.generation import GenerationProbe, run_generation_probe
from evaluation.harness import (
    DensityBatch,
    DensityMetric,
    EvalReport,
    MetricResult,
    SampleBatch,
    SampleMetric,
    all_combinations,
    run_density_metrics,
    run_eval_suite,
    run_sample_metrics,
)
from evaluation.latent_probe import (
    FactorProbe,
    LatentReport,
    blocks_for,
    encode_dataset,
    probe_factor,
    probe_latents,
)
from evaluation.metrics import (
    ColourFidelity,
    DigitAccuracy,
    LabelDiscrimination,
    LatentPlausibility,
    NegativeLogLikelihood,
    SampleDiversity,
)
from evaluation.reconstruction import (
    CombinationProbe,
    reconstruction_summary,
    run_combination_probe,
)
from evaluation.samples import (
    decode_samples,
    latent_traversal,
    reconstruct,
    sample_combination_grid,
    sample_for_label,
)

__all__ = [
    "BG_PALETTE",
    "FG_PALETTE",
    "ColourFidelity",
    "CombinationProbe",
    "DensityBatch",
    "DensityMetric",
    "DigitAccuracy",
    "DigitClassifier",
    "EvalReport",
    "FactorProbe",
    "GenerationProbe",
    "LabelDiscrimination",
    "LatentPlausibility",
    "LatentReport",
    "MetricResult",
    "NegativeLogLikelihood",
    "SampleBatch",
    "SampleDiversity",
    "SampleMetric",
    "all_combinations",
    "blocks_for",
    "border_colour",
    "combination_mean",
    "decode_samples",
    "encode_dataset",
    "foreground_colour",
    "latent_mahalanobis",
    "latent_traversal",
    "load_digit_classifier",
    "marginals",
    "nearest_palette_index",
    "per_image_seen",
    "probe_factor",
    "probe_latents",
    "reconstruct",
    "reconstruction_summary",
    "run_combination_probe",
    "run_density_metrics",
    "run_eval_suite",
    "run_generation_probe",
    "run_sample_metrics",
    "sample_combination_grid",
    "sample_for_label",
    "train_digit_classifier",
    "weighted_mean",
]
