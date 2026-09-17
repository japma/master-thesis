"""Everything that produces a number for the results section.

    metrics.py   the math: tensors in, one value per image out
    collect.py   the loops: run models, call metrics, return DataFrames
"""

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
from evaluation.collect import (
    ImageSet,
    ModelEvaluation,
    combination_table,
    encode_images,
    evaluate_model,
    fit_latent_gaussian,
    mark_seen,
    predicted_digit_entropy,
    reconstruct_images,
    sample_images,
    score_density,
    score_images,
    spread_by_combination,
)
from evaluation.colour import (
    BG_PALETTE,
    FG_PALETTE,
    border_colour,
    foreground_colour,
    nearest_palette_index,
)
from evaluation.latent_probe import (
    FactorProbe,
    LatentReport,
    blocks_for,
    encode_dataset,
    probe_factor,
    probe_latents,
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
    "CombinationProbe",
    "DigitClassifier",
    "FactorProbe",
    "ImageSet",
    "LatentReport",
    "ModelEvaluation",
    "blocks_for",
    "border_colour",
    "combination_mean",
    "combination_table",
    "decode_samples",
    "encode_dataset",
    "encode_images",
    "evaluate_model",
    "fit_latent_gaussian",
    "foreground_colour",
    "latent_mahalanobis",
    "latent_traversal",
    "load_digit_classifier",
    "marginals",
    "mark_seen",
    "nearest_palette_index",
    "per_image_seen",
    "predicted_digit_entropy",
    "probe_factor",
    "probe_latents",
    "reconstruct",
    "reconstruct_images",
    "reconstruction_summary",
    "run_combination_probe",
    "sample_combination_grid",
    "sample_for_label",
    "sample_images",
    "score_density",
    "score_images",
    "spread_by_combination",
    "train_digit_classifier",
    "weighted_mean",
]
