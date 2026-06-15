from dataclasses import dataclass, field
from pathlib import Path

import argparse
from typing import Literal, Union

import yaml


@dataclass
class DatasetConfig:
    name: str = "mnist"
    channels: int = 1
    height: int = 28
    width: int = 28
    num_classes: int = 10
    num_workers: int = 4


@dataclass
class VariationalAutoencoderConfig:
    latent_dim: int = 32
    base_channels: int = 32
    num_blocks: int = 2
    res_blocks: int = 1


@dataclass
class PretrainedAutoencoderConfig:
    model_name: str = "madebyollin/taesd"


AutoencoderConfig = Union[VariationalAutoencoderConfig, PretrainedAutoencoderConfig]


@dataclass
class CSPNConfig:
    model_type: str = "custom"
    num_leaves: int = 10
    num_sums: int = 10
    depth: int = 3
    num_repetitions: int = 1
    nn_hidden_dim: int = 64
    nn_num_hidden_layers: int = 2


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 32
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_anneal_epochs: int = 25


@dataclass
class WandbConfig:
    mode: Literal["online", "offline", "shared", "disabled"] = "online"
    project: str = "master-thesis"
    entity: str = "jmartini-tu-darmstadt"


@dataclass
class PathConfig:
    autoencoder_path: Path = Path("checkpoints/autoencoder/default.pt")
    cspn_path: Path = Path("checkpoints/cspn/default.pt")


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    autoencoder: AutoencoderConfig | None = None
    cspn: CSPNConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: int | None = None


def _parse_autoencoder_config(raw: dict) -> AutoencoderConfig:
    model_type = raw.pop("model_type", "variational")
    if model_type == "variational":
        return VariationalAutoencoderConfig(**raw)
    elif model_type == "pretrained":
        return PretrainedAutoencoderConfig(**raw)
    else:
        raise ValueError(f"Unknown autoencoder model_type: {model_type!r}")


def load_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path, help="Path to config file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        dataset=DatasetConfig(**raw["dataset"])
        if "dataset" in raw
        else DatasetConfig(),
        autoencoder=_parse_autoencoder_config(raw["autoencoder"])
        if "autoencoder" in raw
        else VariationalAutoencoderConfig(),
        cspn=CSPNConfig(**raw["cspn"]) if "cspn" in raw else None,
        training=TrainingConfig(**raw["training"])
        if "training" in raw
        else TrainingConfig(),
        wandb=WandbConfig(**raw["wandb"]) if "wandb" in raw else WandbConfig(),
        paths=_parse_paths(raw["paths"]) if "paths" in raw else PathConfig(),
        seed=raw.get("seed"),
    )

    print(f"Loaded configuration from {args.config_file}")
    print(cfg)
    return cfg


def _parse_paths(raw: dict) -> PathConfig:
    return PathConfig(
        autoencoder_path=Path(raw["autoencoder"])
        if "autoencoder" in raw
        else PathConfig.autoencoder_path,
        cspn_path=Path(raw["cspn"]) if "cspn" in raw else PathConfig.cspn_path,
    )
