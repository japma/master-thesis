"""Generation and evaluation, as two separate stages around one pool per dataset.

    pools.py      the on-disk contract: a dataset's pool and what each directory holds
    samples.py    label columns and the uint8 pixel format every stage shares
    generate.py   stage 1 -- real images, VAE round trips, every listed model's samples
    colour.py     how a colour is read off an image, against the fixed palettes
    conditionals.py  what the training split says a factor's distribution is
    marginal.py   queries with a factor left free: the mixture reference and
                  the calibration that scores it (its own entrypoint)
    metrics.py    the judged metrics, one function each, named in METRICS
    evaluate.py   stage 2 -- judge each set once, write one CSV per metric
    features.py   the pretrained networks set metrics compare images in
    distances.py  FID, KID, precision/recall, CMMD between two feature sets
    sets.py       stage 2 -- set metrics over val-split halvings

Stage 1 computes no metric and stage 2 loads no VAE; the pool is the only thing they
share.
"""

from evaluation.conditionals import conditional, total_variation, training_labels
from evaluation.evaluate import evaluate_pools, find_models, load_judge, predict
from evaluation.generate import (
    MODEL_TYPES,
    generate_pools,
    load_generative_model,
    load_vae,
    sample_and_decode,
    stratified_labels,
)
from evaluation.marginal import calibration, run_marginal, sample_labels
from evaluation.metrics import METRICS
from evaluation.sets import run_sets

__all__ = [
    "METRICS",
    "MODEL_TYPES",
    "calibration",
    "conditional",
    "evaluate_pools",
    "find_models",
    "generate_pools",
    "load_generative_model",
    "load_judge",
    "load_vae",
    "predict",
    "run_marginal",
    "run_sets",
    "sample_and_decode",
    "sample_labels",
    "stratified_labels",
    "total_variation",
    "training_labels",
]
