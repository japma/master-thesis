from dataclasses import dataclass, field
from pathlib import Path

import argparse
from typing import Literal

import yaml

from models.autoencoder import AutoencoderType


@dataclass
class DatasetConfig:
    name: str = "mnist"
    channels: int = 1
    height: int = 28
    width: int = 28
    num_classes: int = 10
    latent_size: int = 8


@dataclass
class AutoencoderConfig:
    model_type: AutoencoderType = AutoencoderType.VARIATIONAL
    base_channels: int = 32
    num_blocks: int = 2
    res_blocks: bool = False
    loss: str = "mse"


@dataclass
class CSPNConfig:
    num_leaves: int = 10
    num_nodes: int = 8
    nn_hidden_dim: int = 64
    nn_num_hidden_layers: int = 2


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_anneal_epochs: int = 10


@dataclass
class Config:
    seed: int | None = None
    wandb: Literal["disabled", "offline", "online", "shared"] | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    cspn: CSPNConfig = field(default_factory=CSPNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--autoencoder", type=str, default="default")
    parser.add_argument("--cspn", type=str, default="default")
    parser.add_argument("--training", type=str, default="default")
    parser.add_argument("--wandb", type=str, default="online")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    dataset_config = Path(f"./configs/dataset/{args.dataset}.yaml")
    autoencoder_config = Path(f"./configs/autoencoder/{args.autoencoder}.yaml")
    cspn_config = Path(f"./configs/cspn/{args.cspn}.yaml")
    training_config = Path(f"./configs/training/{args.training}.yaml")

    with open(dataset_config) as f:
        dataset = DatasetConfig(**yaml.safe_load(f))

    with open(autoencoder_config) as f:
        ae_config_dict = yaml.safe_load(f)
        # Convert model_type string to enum if present
        if "model_type" in ae_config_dict:
            ae_config_dict["model_type"] = AutoencoderType(ae_config_dict["model_type"])
        autoencoder = AutoencoderConfig(**ae_config_dict)

    with open(cspn_config) as f:
        cspn = CSPNConfig(**yaml.safe_load(f))

    with open(training_config) as f:
        training = TrainingConfig(**yaml.safe_load(f))

    full_config = Config(
        seed=args.seed,
        dataset=dataset,
        autoencoder=autoencoder,
        cspn=cspn,
        training=training,
        wandb=args.wandb,
    )
    return full_config


if __name__ == "__main__":
    load_config()
