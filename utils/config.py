from dataclasses import dataclass, field
from pathlib import Path
import argparse
from typing import Literal
import yaml

_CONFIGS_DIR = Path(__file__).parent.parent / "configv2"
_RUNS_DIR = _CONFIGS_DIR / "runs"


@dataclass
class DatasetConfig:
    name: str = "mnist"
    channels: int = 1
    height: int = 28
    width: int = 28
    num_classes: int = 10


@dataclass
class AutoencoderConfig:
    model_type: str = "variational"
    latent_dim: int = 32
    base_channels: int = 32
    num_blocks: int = 2
    res_blocks: int = 1


@dataclass
class PretrainedAutoencoderConfig:
    name: str = "madebyollin/taesd"
    external: bool = True


@dataclass
class CSPNConfig:
    model_type: str = "custom"
    num_repetitions: int = 10
    num_input_distributions: int = 10
    num_sums: int = 10
    min_var: float = 0.1
    max_var: float = 1.0
    h_dims: list[int] = field(default_factory=lambda: [100])


@dataclass
class BaseTrainingConfig:
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass
class AutoencoderTrainingConfig(BaseTrainingConfig):
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 0
    beta_anneal_epochs: int = 25
    loss_type: str = "mse"


@dataclass
class WandbConfig:
    mode: Literal["online", "offline", "shared", "disabled"] = "online"
    project: str = "master-thesis"
    entity: str = "jmartini-tu-darmstadt"


@dataclass
class AERunConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    training: AutoencoderTrainingConfig = field(
        default_factory=AutoencoderTrainingConfig
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int | None = None


@dataclass
class CSPNRunConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: CSPNConfig = field(default_factory=CSPNConfig)
    autoencoder: PretrainedAutoencoderConfig = field(
        default_factory=PretrainedAutoencoderConfig
    )
    training: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int | None = None


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(section: str, name: str) -> dict:
    path = _CONFIGS_DIR / section / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config found at {path}")
    return _read_yaml(path)


def _build_ae_config(raw: dict) -> AERunConfig:
    cfg = AERunConfig()
    cfg.dataset = DatasetConfig(**_resolve("dataset", raw["dataset"]))
    cfg.model = AutoencoderConfig(**_resolve("model/ae", raw["model"]))
    cfg.training = AutoencoderTrainingConfig(**_resolve("training/ae", raw["training"]))
    cfg.wandb.mode = raw.get("wandb", cfg.wandb.mode)
    return cfg


def _build_cspn_config(raw: dict) -> CSPNRunConfig:
    cfg = CSPNRunConfig()
    cfg.dataset = DatasetConfig(**_resolve("dataset", raw["dataset"]))
    cfg.model = CSPNConfig(**_resolve("model/cspn", raw["model"]))
    cfg.training = BaseTrainingConfig(**_resolve("training/cspn", raw["training"]))
    cfg.autoencoder = PretrainedAutoencoderConfig(
        **_resolve("model/pretrained_ae", raw["autoencoder"])
    )
    cfg.wandb.mode = raw.get("wandb", cfg.wandb.mode)
    return cfg


_BUILDERS = {
    "ae": _build_ae_config,
    "cspn": _build_cspn_config,
}


def load_config() -> AERunConfig | CSPNRunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg_file = _RUNS_DIR / args.config_file
    raw = _read_yaml(cfg_file)
    run_type = raw.pop("type", None)

    builder = _BUILDERS.get(run_type)
    if builder is None:
        raise ValueError(f"Unknown run type: {run_type!r}")

    cfg = builder(raw)
    cfg.seed = args.seed
    return cfg
