"""LabelPC model and run configs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from utils.config.common import (
    BaseTrainingConfig,
    DatasetConfig,
    WandbConfig,
)


class LabelPCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_attributes: int = Field(gt=0)
    num_input_distributions: int = 10
    num_sums: int = 10
    num_repetitions: int = 5


class LabelPCRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["label_pc"]
    dataset: DatasetConfig
    model: LabelPCConfig
    training: BaseTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
