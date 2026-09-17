"""Digit classifier model and run configs."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    BaseTrainingConfig,
    DatasetConfig,
    WandbConfig,
)


class ClassifierConfig(BaseModel):
    """A small CNN: two conv blocks, a max-pool after each, then a linear head."""

    model_config = ConfigDict(extra="forbid")

    num_classes: int = 10
    channels: int = 3
    image_size: int = 28
    conv_channels: list[int] = Field(default_factory=lambda: [32, 32, 64])
    hidden_dim: int = 128
    dropout: float = 0.25

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if len(self.conv_channels) != 3:
            raise ValueError(
                "conv_channels names the two convolutions before the first pool and "
                f"the one after it, so it needs exactly 3 entries (got "
                f"{self.conv_channels})"
            )
        if any(width <= 0 for width in self.conv_channels):
            raise ValueError(f"conv_channels must be positive: {self.conv_channels}")
        if self.image_size % 4 != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by 4: the two "
                "max-pools each halve it"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1) (got {self.dropout})")
        return self


class ClassifierTrainingConfig(BaseTrainingConfig):
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


class ClassifierRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["classifier"]
    dataset: DatasetConfig
    model: ClassifierConfig
    training: ClassifierTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
