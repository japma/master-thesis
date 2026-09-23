"""Stage 2: score every generated model in a dataset's pool with a frozen judge.

Colour-MNIST only for now: every metric needs the digit classifier or reads colours off
the pixels. Writes one CSV per metric under `evaluation.results_root`, accumulating
across runs; re-evaluating a model replaces its rows.

    uv run evaluate_samples configs/pools/colour_mnist_skewed.yaml
"""

from evaluation.evaluate import evaluate_pools
from utils.config import PoolRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, PoolRunConfig)
    if cli_seed is not None:
        cfg.generation.seeds = [cli_seed]

    print(f"Judging the pool at {cfg.dataset_dir}")
    evaluate_pools(cfg, resolve_device())


if __name__ == "__main__":
    main()
