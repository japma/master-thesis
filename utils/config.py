"""Helpers for parsing Hydra config objects into resolved DictConfigs."""

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def parse_mode(cfg: DictConfig) -> str:
    """Return the requested application mode from the Hydra config."""
    return cfg.get("mode", "train")


def infer_image_shape_from_input_size(input_size: int) -> tuple[int, int, int]:
    """Infer a channel-first image shape from a flattened input size."""
    known_shapes = {
        784: (1, 28, 28),
        3072: (3, 32, 32),
        12288: (3, 64, 64),
    }
    if input_size not in known_shapes:
        raise ValueError(
            "Unable to infer image shape from input_size. "
            "Please set channels, height, and width in data config."
        )
    return known_shapes[input_size]


def _resolve_dataset_kwargs(data_cfg: DictConfig):
    dataset_kwargs = data_cfg.get("dataset_kwargs")
    if dataset_kwargs is None:
        return None

    resolved_kwargs = OmegaConf.to_container(dataset_kwargs, resolve=True)
    if isinstance(resolved_kwargs, dict):
        return resolved_kwargs

    raise ValueError("data.dataset_kwargs must resolve to a mapping.")


def _resolve_data_config(cfg: DictConfig) -> dict:
    data_cfg = cfg.data
    channels = data_cfg.get("channels")
    height = data_cfg.get("height")
    width = data_cfg.get("width")

    if channels is None or height is None or width is None:
        channels, height, width = infer_image_shape_from_input_size(data_cfg.input_size)

    return {
        "name": data_cfg.name,
        "input_size": data_cfg.input_size,
        "channels": channels,
        "height": height,
        "width": width,
        "image_shape": (channels, height, width),
        "num_classes": data_cfg.get("num_classes", 10),
        "latent_size": data_cfg.latent_size,
        "dataset_kwargs": _resolve_dataset_kwargs(data_cfg),
    }


def _resolve_model_config(cfg: DictConfig) -> dict:
    training_cfg = cfg.model.training
    cspn_cfg = cfg.model.cspn
    cspn_training_cfg = cspn_cfg.training

    return {
        "training": {
            "epochs": training_cfg.epochs,
            "batch_size": training_cfg.batch_size,
            "learning_rate": training_cfg.learning_rate,
        },
        "cspn": {
            "epochs": cspn_training_cfg.epochs,
            "learning_rate": cspn_training_cfg.learning_rate,
        },
    }


def parse_train_config(cfg: DictConfig) -> DictConfig:
    """Resolve the training config into a compact DictConfig."""
    data_cfg = _resolve_data_config(cfg)
    model_cfg = _resolve_model_config(cfg)

    return OmegaConf.create(
        {
            "mode": "train",
            "run_dir": Path(HydraConfig.get().runtime.output_dir),
            "data": data_cfg,
            "model": model_cfg,
        }
    )


def parse_inference_config(cfg: DictConfig) -> DictConfig:
    """Resolve the inference config into a compact DictConfig."""
    data_cfg = _resolve_data_config(cfg)
    model_cfg = _resolve_model_config(cfg)

    checkpoint_dir_value = cfg.inference.checkpoint_dir
    if checkpoint_dir_value in (None, ""):
        raise ValueError(
            "inference.checkpoint_dir must be set to a checkpoints directory "
            "produced by a training run."
        )

    checkpoint_dir = Path(checkpoint_dir_value)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            "inference.checkpoint_dir does not exist: "
            f"{checkpoint_dir}. Set a valid checkpoints directory from a train run."
        )

    return OmegaConf.create(
        {
            "mode": "inference",
            "run_dir": Path(HydraConfig.get().runtime.output_dir),
            "data": data_cfg,
            "model": model_cfg,
            "checkpoint_dir": checkpoint_dir,
            "num_samples": cfg.inference.get("num_samples", 10),
            "max_points": cfg.inference.get("max_points", 2500),
            "samples_per_label": cfg.inference.get("samples_per_label", 250),
            "visualize": {
                "autoencoder": cfg.inference.visualize.autoencoder,
                "latent_space": cfg.inference.visualize.latent_space,
                "cspn": cfg.inference.visualize.cspn,
                "cspn_latent_space": cfg.inference.visualize.cspn_latent_space,
            },
        }
    )