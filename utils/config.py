from typing import Self
import argparse
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class AutoencoderType(StrEnum):
    VARIATIONAL = "variational"
    OTHER = "other"


class VAETrainingType(StrEnum):
    VANILLA = "vanilla"
    BETA = "beta"
    FACTOR = "factor"
    TCVAE = "tcvae"


class CSPNType(StrEnum):
    PSINET = "psinet"
    SPFLOW = "spflow"
    CUSTOM = "custom"
    PSINET_DEPRECATED = "PsiNetCSPN"
    CUSTOM_DEPRECATED = "custom_cspn"


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


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: AutoencoderType
    latent_dim: int
    num_blocks: int
    base_channels: int
    image_size: int = 0
    num_encoder_resblocks: int = 1
    num_decoder_resblocks: int = 1


class PretrainedAutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    external: bool
    latent_dim: int


class CSPNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: CSPNType
    num_vars: int = 0
    num_repetitions: int
    num_input_distributions: int
    num_sums: int
    min_var: float
    max_var: float
    h_dims: list[int]
    num_classes: int = 0

    @model_validator(mode="after")
    def valid_var_range(self) -> Self:
        if self.min_var >= self.max_var:
            raise ValueError(
                f"min_var ({self.min_var}) must be less than max_var ({self.max_var})"
            )
        return self


class BaseTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int
    learning_rate: float
    batch_size: int


class AutoencoderTrainingConfig(BaseTrainingConfig):
    beta: float
    beta_start: float
    beta_end: float
    kl_warmup_epochs: int
    vae_type: VAETrainingType

    # TODO move into config files if needed
    free_bits: float = 0.5
    lambda_perceptual: float = 1.0
    lambda_adversarial: float = 0.1
    adversarial_warmup_steps: int = 1000

    @model_validator(mode="after")
    def validate_beta(self) -> Self:
        assert self.beta >= 0
        assert self.beta_start <= self.beta_end
        assert self.beta == self.beta_end
        return self


class CSPNTrainingConfig(BaseTrainingConfig):
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.01


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["online", "offline", "shared", "disabled"]
    project: str
    entity: str


class AERunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ae"]
    dataset: DatasetConfig
    model: AutoencoderConfig
    training: AutoencoderTrainingConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def inject_image_size(self) -> Self:
        self.model.image_size = self.dataset.height
        return self


class CSPNRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cspn"]
    dataset: DatasetConfig
    model: CSPNConfig
    autoencoder: PretrainedAutoencoderConfig
    training: CSPNTrainingConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def inject_num_classes(self) -> Self:
        self.model.num_classes = self.dataset.num_classes
        return self

    @model_validator(mode="after")
    def inject_num_vars(self) -> Self:
        self.model.num_vars = self.autoencoder.latent_dim
        return self


def load_config() -> tuple[AERunConfig | CSPNRunConfig, int | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    seed = args.seed

    path = args.config_file
    if not path.exists():
        raise FileNotFoundError(f"No config found at {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    run_type = raw.get("type")
    if run_type == "ae":
        return AERunConfig.model_validate(raw), seed
    elif run_type == "cspn":
        return CSPNRunConfig.model_validate(raw), seed
    else:
        raise ValueError(f"Unknown or missing run type: {run_type!r}")
