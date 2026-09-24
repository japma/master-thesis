"""Colour-MNIST classifier model and run configs."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    BaseTrainingConfig,
    DatasetConfig,
    WandbConfig,
)


class ClassifierConfig(BaseModel):
    """A small CNN: two conv blocks, a max-pool after each, then one linear head per
    label factor on a shared hidden layer.

    `cardinalities` and `names` are parallel lists in label-column order; colour-MNIST
    labels are [digit, fg, bg].
    """

    model_config = ConfigDict(extra="forbid")

    cardinalities: list[int] = Field(default_factory=lambda: [10, 6, 3])
    names: list[str] = Field(default_factory=lambda: ["digit", "fg", "bg"])
    channels: int = 3
    image_size: int = 28
    conv_channels: list[int] = Field(default_factory=lambda: [32, 32, 64])
    hidden_dim: int = 128
    dropout: float = 0.25

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if len(self.cardinalities) != len(self.names):
            raise ValueError(
                f"cardinalities ({len(self.cardinalities)}) and names "
                f"({len(self.names)}) must name the same label factors"
            )
        if not self.cardinalities or any(c < 2 for c in self.cardinalities):
            raise ValueError(
                f"every label factor needs at least two classes: {self.cardinalities}"
            )
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


class ClassifierRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["classifier"]
    dataset: DatasetConfig
    model: ClassifierConfig
    training: BaseTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
