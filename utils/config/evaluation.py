"""Generation and evaluation run config.

One file describes a whole evaluation: which model to sample, which VAE decodes it,
which classifier judges it, and how. Both stages read the same config, so the sample
pool's location is derived rather than pasted between them.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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

    # Samples per (digit, fg, bg) combination; 180 cells, so 100 is 18k samples.
    n_per_cell: int = Field(default=100, ge=1)
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


class EvaluationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["evaluation"]
    dataset: DatasetConfig
    model: GeneratedModelConfig
    # Omit to decode with the autoencoder the model checkpoint recorded being trained
    # against; pin it only to override that. Guessing the name is how you end up
    # decoding 20-dim latents with a 16-dim decoder.
    autoencoder: PretrainedAutoencoderConfig | None = None
    classifier: CheckpointConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def pool_dir(self) -> Path:
        """Where stage 1 writes and stage 2 reads. Determined by the config alone."""
        return self.generation.output_root / (
            f"{self.dataset.name}__{self.model.name}__seed{self.generation.seed}"
        )
