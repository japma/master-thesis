"""Neural baseline model and run configs."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from utils.config.common import (
    DatasetConfig,
    PretrainedAutoencoderConfig,
    WandbConfig,
)
from utils.config.cspn import CSPNEncoderConfig, CSPNTrainingConfig


class NeuralBaselineType(StrEnum):
    DETERMINISTIC = "deterministic"
    MIXTURE = "mixture"


class NeuralBaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: NeuralBaselineType
    num_vars: int
    h_dims: list[int]
    encoder_config: CSPNEncoderConfig
    num_components: int = 1
    min_std: float = 0.001
    max_std: float = 2.0
    # DETERMINISTIC only: the fixed sigma that turns NLL into MSE.
    fixed_std: float = 1.0

    @model_validator(mode="after")
    def valid_std_range(self) -> Self:
        if self.min_std >= self.max_std:
            raise ValueError(
                f"min_std ({self.min_std}) must be less than max_std ({self.max_std})"
            )
        if not self.min_std > 0.0:
            raise ValueError(f"min_std ({self.min_std}) must be positive")
        return self

    @model_validator(mode="after")
    def components_match_variant(self) -> Self:
        if (
            self.model_type is NeuralBaselineType.DETERMINISTIC
            and self.num_components != 1
        ):
            raise ValueError(
                "model_type=deterministic has no mixture to weight, so num_components "
                f"must be 1 (got {self.num_components}). Use mixture for more."
            )
        if self.num_components < 1:
            raise ValueError(f"num_components must be >= 1 (got {self.num_components})")
        return self


class NeuralBaselineRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["nn_baseline"]
    dataset: DatasetConfig
    model: NeuralBaselineConfig
    autoencoder: PretrainedAutoencoderConfig
    training: CSPNTrainingConfig
    wandb: WandbConfig
