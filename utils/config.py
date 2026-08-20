import argparse
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

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


class CSPNEncoderType(StrEnum):
    CATEGORICAL = "categorical"
    MULTI_BINARY = "multi_binary"
    MULTI_CATEGORICAL = "multi_categorical"


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
    channels: int = 3 # rgb as default
    num_encoder_resblocks: int = 1
    num_decoder_resblocks: int = 1


class PretrainedAutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    external: bool
    tag: str = "best"


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
        self.model.channels = self.dataset.channels
        return self


class CSPNRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cspn"]
    dataset: DatasetConfig
    model: CSPNConfig
    autoencoder: PretrainedAutoencoderConfig
    training: CSPNTrainingConfig
    wandb: WandbConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """override's keys win; nested dicts are merged recursively rather than replaced
    wholesale, so e.g. a config only needs to state the dataset fields that diverge
    from configs/datasets/{name}.yaml's defaults."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dataset_defaults(raw: dict) -> dict:
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict) or "name" not in dataset:
        return raw

    fragment_path = Path("configs/datasets") / f"{dataset['name']}.yaml"
    if not fragment_path.exists():
        return raw

    with open(fragment_path) as f:
        defaults = yaml.safe_load(f) or {}

    raw["dataset"] = _deep_merge(defaults, dataset)
    return raw


def load_config() -> tuple[AERunConfig | CSPNRunConfig, int | None, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the intermediate checkpoint's saved training state "
            "(optimizer/scheduler/epoch/RNG), if one exists. No-ops back to a "
            "fresh run if no matching train-state sidecar is found."
        ),
    )
    args = parser.parse_args()

    seed = args.seed
    dry_run: bool = args.dry_run
    resume: bool = args.resume

    path = args.config_file
    if not path.exists():
        raise FileNotFoundError(f"No config found at {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _apply_dataset_defaults(raw)

    run_type = raw.get("type")
    if dry_run:
        raw["training"]["epochs"] = 1
        raw["wandb"]["mode"] = "disabled"

    if run_type == "ae":
        return AERunConfig.model_validate(raw), seed, resume
    elif run_type == "cspn":
        return CSPNRunConfig.model_validate(raw), seed, resume
    else:
        raise ValueError(f"Unknown or missing run type: {run_type!r}")
