"""CMMD for a cached sample pool: generated against real, VAE round trip beside it.

Its own entrypoint rather than a metric module, like FID: it compares two whole image
sets and needs no judge. The CLIP ViT-L pass is heavy; run it on the GPU machine.

    uv run evaluate_cmmd configs/evaluation/celeba.yaml
"""

from evaluation.cmmd import run_cmmd
from utils.config import EvaluationRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, EvaluationRunConfig)
    if cli_seed is not None:
        cfg.generation.seed = cli_seed

    print(f"Scoring {cfg.pool_dir}")
    run_cmmd(cfg, resolve_device())


if __name__ == "__main__":
    main()
