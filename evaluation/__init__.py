"""Generation and classifier-based evaluation, as two separate stages.

    samples.py    the on-disk contract: what a sample pool is
    generate.py   stage 1 -- sample p(z | y), decode, write a pool
    colour.py     how a colour is read off an image, against the fixed palettes
    conditionals.py  what the training split says a factor's distribution is
    marginal.py   queries with a factor left free: the mixture reference and
                  the calibration that scores it (its own entrypoint)
    metrics/      one module per metric: a FILENAME and a compute(), listed in METRICS
    features.py   the pretrained networks set metrics compare images in
    distances.py  FID, KID, precision/recall, CMMD between two feature sets
    sets.py       set metrics over val-split halvings (its own entrypoint)
    evaluate.py   stage 2 -- run the judge, write one CSV per metric

Stage 1 computes no metric and stage 2 loads no VAE; the pool directory is the only
thing they share.
"""

from evaluation.conditionals import conditional, total_variation, training_labels
from evaluation.evaluate import evaluate_pool, load_judge, predict
from evaluation.generate import (
    MODEL_TYPES,
    generate_pool,
    load_generative_model,
    load_vae,
    sample_and_decode,
    stratified_labels,
)
from evaluation.marginal import calibration, run_marginal, sample_labels
from evaluation.metrics import METRICS
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
    "METRICS",
    "MODEL_TYPES",
    "ReferenceManifest",
    "SampleManifest",
    "calibration",
    "conditional",
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
    "run_marginal",
    "sample_and_decode",
    "sample_labels",
    "save_pool",
    "stratified_labels",
    "total_variation",
    "training_labels",
]
