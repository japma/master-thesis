"""Unconditional SPN model and run configs."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    DatasetConfig,
    PretrainedAutoencoderConfig,
    WandbConfig,
)
from utils.config.cspn import CSPNTrainingConfig


class SPNConfig(BaseModel):
    """The circuit fields of `CSPNConfig`, without the conditioning network."""

    model_config = ConfigDict(extra="forbid")

    num_vars: int = Field(gt=0)
    num_repetitions: int
    num_input_distributions: int
    num_sums: int
    min_var: float
    max_var: float
    normalize_latents: bool = False

    @model_validator(mode="after")
    def valid_var_range(self) -> Self:
        if self.min_var >= self.max_var:
            raise ValueError(
                f"min_var ({self.min_var}) must be less than max_var ({self.max_var})"
            )
        return self


class SPNRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["spn"]
    dataset: DatasetConfig
    model: SPNConfig
    autoencoder: PretrainedAutoencoderConfig
    training: CSPNTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
