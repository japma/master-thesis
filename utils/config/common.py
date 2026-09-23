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
    # Colour-MNIST only: the label factors a batch carries, e.g. `[digit]`. Omit for
    # all of them.
    labels: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def is_square(self) -> Self:
        assert self.height == self.width
        return self

    @property
    def artifact_name(self) -> str:
        """The dataset as checkpoint and wandb names spell it.

        A label selection changes what a model is trained on, so it has to change the
        name too, or a digit-only run would overwrite the full-label one.
        """
        if self.labels is None:
            return self.name
        return f"{self.name}_labels-{'-'.join(self.labels)}"


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
    # torch.compile the trained model; `--compile` on the CLI forces this on.
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"


WANDB_ENTITY: str = "jmartini-tu-darmstadt"
WANDB_PROJECT: str = "master-thesis"


class WandbConfig(BaseModel):
    """Defaulted throughout, so a run config only mentions wandb to override it."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["online", "offline", "shared", "disabled"] = "online"
    project: str = WANDB_PROJECT
    entity: str = WANDB_ENTITY
