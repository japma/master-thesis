"""Generation and evaluation run config.

One file describes a whole evaluation: which model to sample, which VAE decodes it,
which classifier judges it, and how. Both stages read the same config, so the sample
pool's location is derived rather than pasted between them.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.config.common import (
    DatasetConfig,
    PretrainedAutoencoderConfig,
)

MODEL_TYPES = Literal["cspn", "joint_pc", "nn_baseline"]


class CheckpointConfig(BaseModel):
    """A wandb artifact: `name` alone takes `tag`, `name:v3` pins itself."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tag: str = "latest"


class GeneratedModelConfig(CheckpointConfig):
    model_type: MODEL_TYPES


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "stratified" enumerates every colour-MNIST (digit, fg, bg) cell. "empirical"
    # draws whole label rows from the training split, for a label space too large to
    # enumerate -- CelebA's 40 binary attributes are 2^40 combinations.
    schedule: Literal["stratified", "empirical"] = "stratified"
    # Samples per (digit, fg, bg) combination; 180 cells, so 100 is 18k samples.
    n_per_cell: int = Field(default=100, ge=1)
    # Total samples, read only by the empirical schedule.
    n_samples: int = Field(default=10000, ge=1)
    seed: int = 0
    # Scales the sampled standard deviation; 1.0 samples the model as trained.
    std_correction: float = 1.0
    batch_size: int = Field(default=256, ge=1)
    output_root: Path = Path("results/samples")


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=512, ge=1)
    # One CSV per metric lands here, accumulating across runs.
    results_root: Path = Path("results")
    # Which metrics to write, by module name in `evaluation/metrics/`. Omit for all of
    # them, so a new metric applies to every existing config.
    metrics: list[str] | None = None

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, names: list[str] | None) -> list[str] | None:
        if names is None:
            return names
        # Imported here, not at module level: `evaluation` imports this module back.
        from evaluation.metrics import METRICS_BY_NAME

        if not names:
            raise ValueError("metrics is empty; omit the key to run all of them")
        if len(set(names)) != len(names):
            raise ValueError(f"metrics lists the same metric twice: {names}")
        unknown = [name for name in names if name not in METRICS_BY_NAME]
        if unknown:
            raise ValueError(
                f"unknown metrics {unknown}; known metrics are "
                f"{sorted(METRICS_BY_NAME)}"
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
    # Omit when no judge applies: CelebA is scored with FID, not a digit classifier.
    classifier: CheckpointConfig | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    # Only `evaluate_marginal` reads this; omit it for the ordinary two stages.
    marginal: MarginalConfig | None = None

    @property
    def pool_dir(self) -> Path:
        """Where stage 1 writes and stage 2 reads. Determined by the config alone."""
        return self.generation.output_root / (
            f"{self.dataset.name}__{self.model.name}__seed{self.generation.seed}"
        )
