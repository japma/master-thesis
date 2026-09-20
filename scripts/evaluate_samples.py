"""Score a cached sample pool with a frozen digit classifier.

Stage 2 of two. Reads the pool `generate_samples` wrote for the same config and writes
one CSV per metric under `evaluation.results_root`, accumulating across runs so each
file plots directly. Re-evaluating a run replaces its rows.

    uv run evaluate_samples configs/evaluation/colour_mnist_uniform.yaml
"""

from evaluation.evaluate import evaluate_pool
from utils.config import EvaluationRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, EvaluationRunConfig)
    if cli_seed is not None:
        # The seed picks which pool to read, so it has to match the generation run.
        cfg.generation.seed = cli_seed

    print(f"Judging {cfg.pool_dir} with {cfg.classifier.name}")
    evaluate_pool(cfg, resolve_device())


if __name__ == "__main__":
    main()
