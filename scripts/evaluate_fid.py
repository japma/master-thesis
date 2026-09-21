"""FID for a cached sample pool: generated against real, with the VAE round trip beside it.

Its own entrypoint rather than a metric module: FID compares two whole image sets, while
a metric scores one source against its labels. It also needs no judge, which is what
makes it the way to score CelebA.

    uv run evaluate_fid configs/evaluation/celeba.yaml
"""

from evaluation.fid import run_fid
from utils.config import EvaluationRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, EvaluationRunConfig)
    if cli_seed is not None:
        cfg.generation.seed = cli_seed

    print(f"Scoring {cfg.pool_dir}")
    run_fid(cfg, resolve_device())


if __name__ == "__main__":
    main()
