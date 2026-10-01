"""Stage 2 for marginalized queries: judge every marginal set in a dataset's pool.

`generate_pools` samples them when the pool config has a `marginal:` block: for each
query, training images matching it (the floor) and every listed model's answer by
each arm it supports. This scores how far each set's mix is from the training
conditional, into `colour_calibration.csv` and `colour_calibration_histogram.csv`
under `evaluation.results_root`; re-scoring a set replaces its rows.

    uv run evaluate_marginal configs/pools/colour_mnist_correlated.yaml
"""

from evaluation.marginal import evaluate_marginal
from utils.config import PoolRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, PoolRunConfig)
    if cli_seed is not None:
        cfg.generation.seeds = [cli_seed]

    print(f"Judging the marginal sets in {cfg.dataset_dir}")
    evaluate_marginal(cfg, resolve_device())


if __name__ == "__main__":
    main()
