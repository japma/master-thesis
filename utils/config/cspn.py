"""CSPN model, label-encoder, LabelPC, training, and run configs."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    BaseTrainingConfig,
    DatasetConfig,
    PretrainedAutoencoderConfig,
    WandbConfig,
)


class CSPNEncoderType(StrEnum):
    CATEGORICAL = "categorical"
    MULTI_BINARY = "multi_binary"
    MULTI_CATEGORICAL = "multi_categorical"


class ConditioningType(StrEnum):
    JOINT = "joint"
    FACTORIZED = "factorized"


class CSPNType(StrEnum):
    PSINET = "psinet"
    SPFLOW = "spflow"
    CUSTOM = "custom"
    PSINET_DEPRECATED = "PsiNetCSPN"
    CUSTOM_DEPRECATED = "custom_cspn"


class CSPNEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_type: CSPNEncoderType
    num_classes: list[int] = []

    @model_validator(mode="after")
    def validate_encoder_config(self) -> Self:
        match self.encoder_type:
            case CSPNEncoderType.CATEGORICAL | CSPNEncoderType.MULTI_BINARY:
                if (
                    not self.num_classes
                    or len(self.num_classes) != 1
                    or self.num_classes[0] <= 0
                ):
                    raise ValueError(
                        "num_classes must be a one-element list of positive integers"
                    )
            case CSPNEncoderType.MULTI_CATEGORICAL:
                if not self.num_classes or any(c <= 0 for c in self.num_classes):
                    raise ValueError(
                        "num_classes must be a non-empty list of positive integers for multi-categorical encoder."
                    )
            case _:
                raise ValueError("Unknown encoder type")

        return self


class CSPNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: CSPNType
    num_vars: int
    num_repetitions: int
    num_input_distributions: int
    num_sums: int
    min_var: float
    max_var: float
    h_dims: list[int]
    encoder_config: CSPNEncoderConfig
    normalize_latents: bool = False
    conditioning_type: ConditioningType = ConditioningType.JOINT

    @model_validator(mode="after")
    def valid_var_range(self) -> Self:
        if self.min_var >= self.max_var:
            raise ValueError(
                f"min_var ({self.min_var}) must be less than max_var ({self.max_var})"
            )
        return self

    @model_validator(mode="after")
    def factorized_needs_multiple_factors(self) -> Self:
        if (
            self.conditioning_type is ConditioningType.FACTORIZED
            and self.encoder_config.encoder_type is CSPNEncoderType.CATEGORICAL
        ):
            raise ValueError(
                "conditioning_type=factorized requires a label encoder with more than "
                "one factor, but `categorical` encodes a single one — the factorized "
                "and joint networks would be identical. Use multi_binary or "
                "multi_categorical, or leave conditioning_type at its default."
            )
        return self


class CSPNTrainingConfig(BaseTrainingConfig):
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.01


class PretrainedLabelPCConfig(BaseModel):
    """Which trained LabelPC artifact a CSPN run should load, if any.

    Architecture lives in `LabelPCConfig`, which only a LabelPC run reads.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    tag: str = "latest"

    def resolve_name(self, dataset_name: str) -> str:
        return self.name or f"label_pc_{dataset_name}"


class CSPNRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cspn"]
    dataset: DatasetConfig
    model: CSPNConfig
    autoencoder: PretrainedAutoencoderConfig
    training: CSPNTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    label_pc: PretrainedLabelPCConfig = Field(
        default_factory=PretrainedLabelPCConfig
    )
