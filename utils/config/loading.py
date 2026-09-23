"""YAML loading, dataset-fragment composition, and the training-script CLI."""

import argparse
from pathlib import Path

import yaml

from utils.config.autoencoder import AERunConfig
from utils.config.classifier import ClassifierRunConfig
from utils.config.common import DatasetConfig
from utils.config.cspn import CSPNRunConfig
from utils.config.evaluation import EvaluationRunConfig, PoolRunConfig
from utils.config.joint_pc import JointPCRunConfig
from utils.config.label_pc import LabelPCRunConfig
from utils.config.neural_baseline import NeuralBaselineRunConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """override's keys win; nested dicts are merged recursively rather than replaced
    wholesale, so e.g. a config only needs to state the dataset fields that diverge
    from configs/datasets/{name}.yaml's defaults."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
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


def load_dataset_config(name: str) -> DatasetConfig:
    """The dataset fragment `configs/datasets/{name}.yaml` on its own.

    For scripts that are handed a dataset name rather than a run config -- an
    evaluation entry point, say -- so they read the same shape/class counts every
    training run does instead of restating them.
    """
    raw = _apply_dataset_defaults({"dataset": {"name": name}})["dataset"]
    if set(raw) == {"name"}:
        raise FileNotFoundError(f"No dataset fragment at configs/datasets/{name}.yaml")
    return DatasetConfig.model_validate(raw)


RunConfig = (
    AERunConfig
    | ClassifierRunConfig
    | CSPNRunConfig
    | EvaluationRunConfig
    | PoolRunConfig
    | JointPCRunConfig
    | LabelPCRunConfig
    | NeuralBaselineRunConfig
)

_RUN_TYPES: dict[str, type] = {
    "ae": AERunConfig,
    "classifier": ClassifierRunConfig,
    "cspn": CSPNRunConfig,
    "evaluation": EvaluationRunConfig,
    "joint_pc": JointPCRunConfig,
    "label_pc": LabelPCRunConfig,
    "nn_baseline": NeuralBaselineRunConfig,
    "pools": PoolRunConfig,
}


def load_config() -> tuple[RunConfig, int | None, bool]:
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
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode, overriding training.compile_mode. Implies --compile."
        ),
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
        if "training" in raw:
            raw["training"]["epochs"] = 1
            raw.setdefault("wandb", {})["mode"] = "disabled"
        else:
            # An evaluation has no epochs to cut and never opens a run; shrink the
            # sample schedule instead.
            raw.setdefault("generation", {})["n_per_cell"] = 1
    if (args.compile or args.compile_mode) and "training" in raw:
        raw["training"]["compile"] = True
        if args.compile_mode:
            raw["training"]["compile_mode"] = args.compile_mode

    config_type = _RUN_TYPES.get(str(run_type))
    if config_type is None:
        raise ValueError(
            f"Unknown or missing run type: {run_type!r} "
            f"(expected one of {sorted(_RUN_TYPES)})"
        )
    return config_type.model_validate(raw), seed, resume
