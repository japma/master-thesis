"""Stage 1: fill a dataset's pool with real images, VAE round trips and every listed
model's samples (see `evaluation.pools` for the layout).

Anything already generated is skipped: the real split once per dataset, a VAE's round
trip once per VAE version, a model's samples once per version, seed and std_correction.
A listed model wandb does not have is reported and skipped, never fatal, so the closing
summary doubles as the list of models still to train.

    uv run generate_pools configs/pools/colour_mnist_skewed.yaml
    uv run generate_pools configs/pools/celeba.yaml --seed 1
"""

from evaluation.generate import generate_pools
from utils.config import PoolRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, PoolRunConfig)
    if cli_seed is not None:
        cfg.generation.seeds = [cli_seed]

    print(
        f"Pool for {cfg.dataset.name} at {cfg.dataset_dir} | {len(cfg.models)} models "
        f"| seeds {cfg.generation.seeds} | labels: {cfg.generation.labels}"
    )
    generate_pools(cfg, resolve_device())


if __name__ == "__main__":
    main()
