"""Pool and evaluation configs.

`PoolRunConfig` (`type: pools`) describes one dataset: every generative model to sample
into its pool, and how the pool is scored. `generate_pools`, `evaluate_samples` and
`evaluate_sets` all read it.

`EvaluationRunConfig` (`type: evaluation`) is one model at a time, and only
`evaluate_marginal` still reads it.
"""

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from utils.config.common import (
    DatasetConfig,
    LabelFactor,
    PretrainedAutoencoderConfig,
)


class GenerativeModelType(StrEnum):
    """The models over a VAE's latent space; all expose `sample(labels)`, and the
    unconditional ones use the labels only for the batch size."""

    CSPN = "cspn"
    JOINT_PC = "joint_pc"
    NN_BASELINE = "nn_baseline"
    # N(0, I), decoded by the VAE named as the model itself.
    VAE_PRIOR = "vae_prior"
    GMM = "gmm"
    SPN = "spn"


class CheckpointConfig(BaseModel):
    """A wandb artifact: `name` alone takes `tag`, `name:v3` pins itself."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tag: str = "latest"


class GeneratedModelConfig(CheckpointConfig):
    model_type: GenerativeModelType


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_per_cell: int = Field(default=100, ge=1)
    seed: int = 0
    # Scales the sampled standard deviation; 1.0 samples the model as trained.
    std_correction: float = 1.0
    batch_size: int = Field(default=256, ge=1)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=512, ge=1)
    # One CSV per metric lands here, accumulating across runs.
    results_root: Path = Path("results")
    # Which metrics to write, by name in `evaluation.metrics.METRICS`. Omit for all of
    # them, so a new metric applies to every existing config.
    metrics: list[str] | None = None
    # Which metrics `evaluate_sets` computes, by name in `evaluation.sets.SET_METRICS`.
    set_metrics: list[str] = Field(
        default_factory=lambda: ["fid", "kid", "precision", "recall", "cmmd"]
    )
    # Random halvings of the val split that `evaluate_sets` averages over.
    halvings: int = Field(default=5, ge=1)

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, names: list[str] | None) -> list[str] | None:
        if names is None:
            return names
        # Imported here, not at module level: `evaluation` imports this module back.
        from evaluation.metrics import METRICS

        if not names:
            raise ValueError("metrics is empty; omit the key to run all of them")
        if len(set(names)) != len(names):
            raise ValueError(f"metrics lists the same metric twice: {names}")
        unknown = [name for name in names if name not in METRICS]
        if unknown:
            raise ValueError(
                f"unknown metrics {unknown}; known metrics are {sorted(METRICS)}"
            )
        return names

    @field_validator("set_metrics")
    @classmethod
    def _known_set_metrics(cls, names: list[str]) -> list[str]:
        from evaluation.sets import SET_METRICS

        if not names:
            raise ValueError("set_metrics is empty")
        if len(set(names)) != len(names):
            raise ValueError(f"set_metrics lists the same metric twice: {names}")
        unknown = [name for name in names if name not in SET_METRICS]
        if unknown:
            raise ValueError(
                f"unknown set metrics {unknown}; known set metrics are "
                f"{sorted(SET_METRICS)}"
            )
        return names


class MarginalConfig(BaseModel):
    """Marginalized queries: which factors to leave to the model, and how many samples."""

    model_config = ConfigDict(extra="forbid")

    # Each query is [digit, fg, bg]; -1 leaves that factor free. The digit must be
    # given: reading a digit back needs the judge, the colours are read off pixels.
    queries: list[list[int]]
    n_per_query: int = Field(default=1000, ge=1)

    @field_validator("queries")
    @classmethod
    def _valid_queries(cls, queries: list[list[int]]) -> list[list[int]]:
        # Imported here, not at module level: `evaluation` imports this module back.
        from evaluation.marginal import as_query

        if not queries:
            raise ValueError("queries is empty; drop the `marginal` block instead")
        for query in queries:
            as_query(query)
        return queries


class EvaluationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["evaluation"]
    dataset: DatasetConfig
    model: GeneratedModelConfig
    # Omit to decode with the autoencoder the model checkpoint recorded being trained
    # against; pin it only to override that. Guessing the name is how you end up
    # decoding 20-dim latents with a 16-dim decoder.
    autoencoder: PretrainedAutoencoderConfig | None = None
    classifier: CheckpointConfig | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    marginal: MarginalConfig | None = None


class PoolModelConfig(BaseModel):
    """One generative model in a dataset's pool."""

    model_config = ConfigDict(extra="forbid")

    type: GenerativeModelType
    # The wandb artifact collection.
    name: str
    # `v3` pins a version, an alias such as `best` follows it; omit for the latest.
    # Either is resolved to its `vN` and recorded, so no pool or result names an alias.
    version: str | None = None
    # The label columns the model conditions on, for models trained on a subset, e.g.
    # `[digit]`. Omit for all of them.
    labels: tuple[LabelFactor, ...] | None = None
    std_correction: float = 1.0

    @field_validator("version")
    @classmethod
    def _is_a_version(cls, version: str | None) -> str | None:
        if version == "latest":
            raise ValueError("omit version for the latest instead of naming it")
        if version is not None and not re.fullmatch(r"v\d+|[A-Za-z][\w-]*", version):
            raise ValueError(f"version must look like v3 or an alias, got {version!r}")
        return version


class PoolGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # What the models are conditioned on. `real`: the val labels, one sample per real
    # image, in the same order. `stratified`: every colour-MNIST (digit, fg, bg) cell
    # `n_per_cell` times, held-out cells included.
    labels: Literal["real", "stratified"]
    n_per_cell: int = Field(default=100, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    batch_size: int = Field(default=256, ge=1)
    root: Path = Path("results/pools")


class PoolRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["pools"]
    dataset: DatasetConfig
    models: list[PoolModelConfig] = Field(min_length=1)
    # Omit when no judge applies: CelebA has no attribute classifier yet.
    classifier: CheckpointConfig | None = None
    generation: PoolGenerationConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _unique_models(self) -> Self:
        keys = [(m.name, m.std_correction) for m in self.models]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(
                f"models lists the same (name, std_correction) twice: {repeated}"
            )
        return self

    @property
    def dataset_dir(self) -> Path:
        return self.generation.root / self.dataset.name
