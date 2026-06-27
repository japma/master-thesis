import argparse
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_RUNS_DIR = Path(__file__).parent.parent / "configv2" / "runs"


class AutoencoderType(StrEnum):
    VARIATIONAL = "variational"
    OTHER = "other"


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
    def is_square(self):
        assert self.height == self.width
        return self


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: AutoencoderType
    latent_dim: int
    num_blocks: int
    base_channels: int
    image_size: int = 0


class PretrainedAutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    external: bool


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
    num_classes: int = 0

    @model_validator(mode="after")
    def valid_var_range(self):
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
    kl_warmup_epochs: int


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
    def inject_image_size(self):
        self.model.image_size = self.dataset.height
        return self


class CSPNRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cspn"]
    dataset: DatasetConfig
    model: CSPNConfig
    autoencoder: PretrainedAutoencoderConfig
    training: BaseTrainingConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def inject_num_classes(self):
        self.model.num_classes = self.dataset.num_classes
        return self


def load_config() -> tuple[AERunConfig | CSPNRunConfig, int | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    seed = args.seed

    # path = _RUNS_DIR / args.config_file
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
