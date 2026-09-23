"""Joint latent+label PC model and run configs."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    DatasetConfig,
    PretrainedAutoencoderConfig,
    WandbConfig,
)
from utils.config.cspn import CSPNTrainingConfig


class JointPCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_latents: int = Field(gt=0)
    # One entry per label factor, in target-vector order. Colour-MNIST targets are
    # [digit, fg, bg], so [10, 6, 3].
    label_cardinalities: list[int]
    num_repetitions: int = 5
    num_input_distributions: int = 16
    num_sums: int = 16
    min_var: float = 1e-4
    max_var: float = 4.0
    normalize_latents: bool = False
    # Weight on the exact log p(y | z) term added to the joint NLL. The motivation is
    # that the labels are ~5 nats of a loss dominated by 16 continuous dims, so the
    # label-latent coupling gets little of the gradient. UNVERIFIED: a 400-step
    # synthetic run at lambda=5 was indistinguishable from lambda=0, which is too short
    # to conclude anything either way. 0.0 keeps the plain joint NLL.
    conditional_weight: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_variant(cls, data: object) -> object:
        """Checkpoints written before names were derived still carry a `variant`."""
        if isinstance(data, dict):
            data = {key: value for key, value in data.items() if key != "variant"}
        return data

    @model_validator(mode="after")
    def valid_label_cardinalities(self) -> Self:
        if not self.label_cardinalities or any(c < 2 for c in self.label_cardinalities):
            raise ValueError(
                "label_cardinalities must be a non-empty list of integers >= 2"
            )
        return self

    @model_validator(mode="after")
    def valid_var_range(self) -> Self:
        if self.min_var >= self.max_var:
            raise ValueError(
                f"min_var ({self.min_var}) must be less than max_var ({self.max_var})"
            )
        return self


class JointPCRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["joint_pc"]
    dataset: DatasetConfig
    model: JointPCConfig
    autoencoder: PretrainedAutoencoderConfig
    training: CSPNTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
