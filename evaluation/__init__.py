"""Generation and classifier-based evaluation, as two separate stages.

    samples.py    the on-disk contract: what a sample pool is
    generate.py   stage 1 -- sample p(z | y), decode, write a pool
    metrics.py    the numbers: predictions and labels in, a value or a table out
    evaluate.py   stage 2 -- run the judge, write one CSV per metric

Stage 1 computes no metric and stage 2 loads no VAE; the pool directory is the only
thing they share.
"""

from evaluation.evaluate import evaluate_pool, load_judge, predict
from evaluation.generate import (
    MODEL_TYPES,
    generate_pool,
    load_generative_model,
    load_vae,
    sample_and_decode,
    stratified_labels,
)
from evaluation.metrics import accuracy, accuracy_by_combination, confusion
from evaluation.samples import (
    ReferenceManifest,
    SampleManifest,
    load_images,
    load_labels,
    load_latents,
    load_originals,
    load_reference_manifest,
    load_sample_manifest,
    reference_dir,
    save_pool,
)

__all__ = [
    "MODEL_TYPES",
    "ReferenceManifest",
    "SampleManifest",
    "accuracy",
    "accuracy_by_combination",
    "confusion",
    "evaluate_pool",
    "generate_pool",
    "load_generative_model",
    "load_images",
    "load_judge",
    "load_labels",
    "load_latents",
    "load_originals",
    "load_reference_manifest",
    "load_sample_manifest",
    "load_vae",
    "predict",
    "reference_dir",
    "sample_and_decode",
    "save_pool",
    "stratified_labels",
]
