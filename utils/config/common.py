"""Config fragments shared by every run type."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    channels: int
    height: int
    width: int
    num_classes: int

    @model_validator(mode="after")
    def is_square(self) -> Self:
        assert self.height == self.width
        return self


class PretrainedAutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    external: bool
    tag: str = "best"


class BaseTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int
    learning_rate: float
    batch_size: int


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["online", "offline", "shared", "disabled"]
    project: str
    entity: str
