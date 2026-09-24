"""Gaussian mixture (ex-post density) run config."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from utils.config.common import DatasetConfig, PretrainedAutoencoderConfig, WandbConfig


class GMMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_components: int = Field(default=10, ge=1)
    # Added to every covariance diagonal, as in scikit-learn.
    reg_covar: float = Field(default=1e-6, gt=0)


class GMMTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # EM iterations; each is one pass over the encoded training set.
    epochs: int = Field(default=200, ge=1)
    # Only for encoding the dataset.
    batch_size: int = Field(default=256, ge=1)


class GMMRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gmm"]
    dataset: DatasetConfig
    autoencoder: PretrainedAutoencoderConfig
    model: GMMConfig = Field(default_factory=GMMConfig)
    training: GMMTrainingConfig = Field(default_factory=GMMTrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
