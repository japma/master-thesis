"""YAML loading, dataset-fragment composition, and the training-script CLI."""

import argparse
from pathlib import Path

import yaml

from utils.config.autoencoder import AERunConfig
from utils.config.cspn import CSPNRunConfig
from utils.config.neural_baseline import NeuralBaselineRunConfig


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


def load_config() -> (
    tuple[AERunConfig | CSPNRunConfig | NeuralBaselineRunConfig, int | None, bool]
):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the trained model, overriding training.compile",
    )
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
    if args.compile:
        raw["training"]["compile"] = True

    if run_type == "ae":
        return AERunConfig.model_validate(raw), seed, resume
    elif run_type == "cspn":
        return CSPNRunConfig.model_validate(raw), seed, resume
    elif run_type == "nn_baseline":
        return NeuralBaselineRunConfig.model_validate(raw), seed, resume
    else:
        raise ValueError(f"Unknown or missing run type: {run_type!r}")
