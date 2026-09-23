"""Checkpoint and wandb artifact names, derived from config so none can be forgotten.

    vae_<dataset>_<vae kind>
    <family>_<dataset>_<vae kind>[_<extra>...]      cspn, joint_pc, nn_baseline
    label_pc_<dataset>
    digit_classifier_<dataset>

Underscores only. The dataset part includes a label selection
(`colour_mnist_skewed_labels_digit`). The VAE kind names the latent space a model lives
in -- `variational`, `anchored`, `anchored_digit`, `variational_tcvae`,
`variational_beta4` -- so the same CSPN on two autoencoders is two collections, never two
versions of one.
"""

import re
from collections.abc import Mapping
from enum import StrEnum

from utils.config import (
    AutoencoderConfig,
    AutoencoderTrainingConfig,
    ConditioningType,
    CSPNConfig,
    DatasetConfig,
    NeuralBaselineConfig,
    PretrainedAutoencoderConfig,
    VAETrainingType,
)

# The key under which a VAE artifact records its kind, for the models trained on it.
VAE_KIND_KEY = "vae_kind"


class ModelFamily(StrEnum):
    VAE = "vae"
    CSPN = "cspn"
    JOINT_PC = "joint_pc"
    NN_BASELINE = "nn_baseline"
    LABEL_PC = "label_pc"
    DIGIT_CLASSIFIER = "digit_classifier"


def _number(value: float) -> str:
    """`4.0` -> `4`, `0.5` -> `0p5`: a number as a name can hold it."""
    return f"{value:g}".replace(".", "p").replace("-", "m")


def artifact_name(family: ModelFamily, dataset: DatasetConfig, *parts: str) -> str:
    return "_".join([family, dataset.artifact_name, *parts])


def vae_kind(
    model: AutoencoderConfig, training: AutoencoderTrainingConfig | None = None
) -> str:
    """Model type and variant, then the objective and beta where they differ from the
    plain beta=1 VAE. Without `training`, only what a checkpoint's model config holds."""
    parts = [str(model.model_type)]
    if model.variant is not None:
        parts.append(model.variant)
    if training is not None:
        if training.vae_type in (VAETrainingType.FACTOR, VAETrainingType.TCVAE):
            parts.append(training.vae_type)
        if training.beta != 1.0:
            parts.append(f"beta{_number(training.beta)}")
    return "_".join(parts)


def pretrained_vae_kind(
    cfg: PretrainedAutoencoderConfig,
    metadata: Mapping[str, object],
    model_config: AutoencoderConfig | None,
) -> str:
    """The kind of the VAE a latent-space model is trained on.

    Its artifact records it; autoencoders logged before that fall back to the model
    config inside the checkpoint, and Hugging Face ones to their repo name.
    """
    if cfg.external:
        return re.sub(r"[^a-z0-9]+", "_", cfg.name.lower()).strip("_")
    recorded = metadata.get(VAE_KIND_KEY)
    if isinstance(recorded, str):
        return recorded
    if model_config is None:
        raise ValueError(f"cannot tell what kind of VAE {cfg.name} is")
    return vae_kind(model_config)


def cspn_extras(model: CSPNConfig) -> list[str]:
    extras = []
    if model.conditioning_type is not ConditioningType.JOINT:
        extras.append(str(model.conditioning_type))
    if model.encoder_config.label_dropout_prob > 0:
        extras.append("dontcare")
    return extras


def nn_baseline_extras(model: NeuralBaselineConfig) -> list[str]:
    return [str(model.model_type)]


def intermediate_name(name: str) -> str:
    return f"intermediate_{name}"
