"""Autoencoder model, training, and run configs."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    BaseTrainingConfig,
    DatasetConfig,
    WandbConfig,
)


class AutoencoderType(StrEnum):
    VARIATIONAL = "variational"
    OTHER = "other"


class VAETrainingType(StrEnum):
    VANILLA = "vanilla"
    BETA = "beta"
    FACTOR = "factor"
    TCVAE = "tcvae"


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: AutoencoderType
    latent_dim: int
    num_blocks: int
    base_channels: int
    image_size: int = 0
    channels: int = 3 # rgb as default
    num_encoder_resblocks: int = 1
    num_decoder_resblocks: int = 1


class AutoencoderTrainingConfig(BaseTrainingConfig):
    beta: float
    beta_start: float
    beta_end: float
    kl_warmup_epochs: int
    vae_type: VAETrainingType

    # only required (and used) when vae_type == tcvae; see validate_tcvae_params
    tcvae_alpha: float | None = None
    tcvae_beta: float | None = None
    tcvae_gamma: float | None = None

    # TODO move into config files if needed
    free_bits: float = 0.5
    lambda_perceptual: float = 1.0
    lambda_adversarial: float = 0.1
    adversarial_warmup_steps: int = 1000

    @model_validator(mode="after")
    def validate_beta(self) -> Self:
        assert self.beta >= 0
        assert self.beta_start <= self.beta_end
        if self.vae_type == VAETrainingType.BETA:
            assert self.beta == self.beta_end
        return self

    @model_validator(mode="after")
    def validate_tcvae_params(self) -> Self:
        if self.vae_type == VAETrainingType.TCVAE:
            missing = [
                name
                for name, val in (
                    ("tcvae_alpha", self.tcvae_alpha),
                    ("tcvae_beta", self.tcvae_beta),
                    ("tcvae_gamma", self.tcvae_gamma),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"vae_type=tcvae requires {', '.join(missing)} to be set in training config"
                )
            assert self.tcvae_beta is not None
            assert self.beta_start <= self.tcvae_beta
        return self


class AERunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ae"]
    dataset: DatasetConfig
    model: AutoencoderConfig
    training: AutoencoderTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def inject_image_size(self) -> Self:
        self.model.image_size = self.dataset.height
        self.model.channels = self.dataset.channels
        return self
