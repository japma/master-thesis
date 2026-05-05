"""Helpers for parsing Hydra config objects into resolved DictConfigs."""

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


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


def _resolve_training_config(cfg: DictConfig) -> dict:
    training_cfg = cfg.get("training")
    if training_cfg is None:
        model_cfg = cfg.get("model")
        if model_cfg is not None:
            training_cfg = model_cfg.get("training")

    if training_cfg is None:
        raise ValueError("Missing training config.")

    resolved_training = {}
    if training_cfg.get("batch_size") is not None:
        resolved_training["batch_size"] = training_cfg.batch_size
    if training_cfg.get("epochs") is not None:
        resolved_training["epochs"] = training_cfg.epochs
    if training_cfg.get("learning_rate") is not None:
        resolved_training["learning_rate"] = training_cfg.learning_rate

    return resolved_training


def _resolve_autoencoder_model_config(cfg: DictConfig) -> dict:
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Missing model config.")

    return {
        "architecture": model_cfg.get("architecture", "variational"),
        "base_channels": model_cfg.get("base_channels", 32),
    }


def _resolve_cspn_model_config(cfg: DictConfig) -> dict:
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Missing model config.")

    cspn_cfg = model_cfg.get("cspn")
    if cspn_cfg is None:
        raise ValueError("Missing model.cspn config.")

    return {
        "context_hidden_dim": cspn_cfg.get("context_hidden_dim", 128),
        "num_mixture_components": cspn_cfg.get("num_mixture_components", 4),
        "num_sum_components": cspn_cfg.get("num_sum_components", 2),
    }


def _resolve_checkpoint_path(checkpoint_value, field_name: str) -> Path:
    if checkpoint_value in (None, ""):
        raise ValueError(f"{field_name} must be set to a checkpoints directory.")

    checkpoint_dir = Path(checkpoint_value)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"{field_name} does not exist: {checkpoint_dir}. Set a valid checkpoints directory."
        )
    return checkpoint_dir


def parse_mode(cfg: DictConfig) -> str:
    """Return the requested task from the Hydra config.

    Kept for compatibility with the legacy dispatcher.
    """
    return cfg.get("task", cfg.get("mode", "train"))


def parse_autoencoder_train_config(cfg: DictConfig) -> DictConfig:
    """Resolve the autoencoder training config into a compact DictConfig."""
    data_cfg = _resolve_data_config(cfg)
    model_cfg = _resolve_autoencoder_model_config(cfg)
    training_cfg = _resolve_training_config(cfg)

    return OmegaConf.create(
        {
            "task": "train_autoencoder",
            "run_dir": Path(HydraConfig.get().runtime.output_dir),
            "data": data_cfg,
            "model": model_cfg,
            "training": training_cfg,
        }
    )


def parse_train_config(cfg: DictConfig) -> DictConfig:
    """Compatibility wrapper for the legacy training config parser."""
    return parse_autoencoder_train_config(cfg)


def parse_cspn_train_config(cfg: DictConfig) -> DictConfig:
    """Resolve the CSPN training config into a compact DictConfig."""
    data_cfg = _resolve_data_config(cfg)
    model_cfg = {
        "autoencoder": _resolve_autoencoder_model_config(cfg),
        "cspn": _resolve_cspn_model_config(cfg),
    }
    training_cfg = _resolve_training_config(cfg)

    checkpoints_cfg = cfg.get("checkpoints")
    if checkpoints_cfg is None:
        legacy_checkpoint_dir = cfg.get("checkpoint_dir")
        if legacy_checkpoint_dir is None:
            raise ValueError("Missing checkpoints config.")
        checkpoints_cfg = OmegaConf.create({"autoencoder_dir": legacy_checkpoint_dir})

    autoencoder_checkpoint_dir = _resolve_checkpoint_path(
        checkpoints_cfg.get("autoencoder_dir"),
        "checkpoints.autoencoder_dir",
    )

    return OmegaConf.create(
        {
            "task": "train_cspn",
            "run_dir": Path(HydraConfig.get().runtime.output_dir),
            "data": data_cfg,
            "model": model_cfg,
            "training": training_cfg,
            "checkpoints": {"autoencoder_dir": autoencoder_checkpoint_dir},
        }
    )


def parse_inference_config(cfg: DictConfig) -> DictConfig:
    """Resolve the inference config into a compact DictConfig."""
    data_cfg = _resolve_data_config(cfg)
    model_cfg = {
        "autoencoder": _resolve_autoencoder_model_config(cfg),
        "cspn": _resolve_cspn_model_config(cfg),
    }
    training_cfg = _resolve_training_config(cfg)

    checkpoints_cfg = cfg.get("checkpoints")
    if checkpoints_cfg is None:
        legacy_inference_cfg = cfg.get("inference")
        if legacy_inference_cfg is None:
            raise ValueError("Missing checkpoints config.")
        checkpoints_cfg = OmegaConf.create(
            {
                "autoencoder_dir": legacy_inference_cfg.get("checkpoint_dir"),
                "cspn_dir": legacy_inference_cfg.get("checkpoint_dir"),
            }
        )

    autoencoder_checkpoint_dir = _resolve_checkpoint_path(
        checkpoints_cfg.get("autoencoder_dir"),
        "checkpoints.autoencoder_dir",
    )
    cspn_checkpoint_dir = _resolve_checkpoint_path(
        checkpoints_cfg.get("cspn_dir"),
        "checkpoints.cspn_dir",
    )

    visualize_cfg = cfg.get("visualize")
    if visualize_cfg is None:
        legacy_inference_cfg = cfg.get("inference")
        if legacy_inference_cfg is None:
            raise ValueError("Missing visualize config.")
        visualize_cfg = legacy_inference_cfg.visualize

    return OmegaConf.create(
        {
            "task": "inference",
            "run_dir": Path(HydraConfig.get().runtime.output_dir),
            "data": data_cfg,
            "model": model_cfg,
            "training": training_cfg,
            "checkpoints": {
                "autoencoder_dir": autoencoder_checkpoint_dir,
                "cspn_dir": cspn_checkpoint_dir,
            },
            "num_samples": cfg.get("num_samples", 10),
            "max_points": cfg.get("max_points", 2500),
            "samples_per_label": cfg.get("samples_per_label", 250),
            "visualize": {
                "autoencoder": visualize_cfg.autoencoder,
                "latent_space": visualize_cfg.latent_space,
                "cspn": visualize_cfg.cspn,
                "cspn_latent_space": visualize_cfg.cspn_latent_space,
            },
        }
    )
