"""Set metrics for every generated model in a dataset's pool -- FID, KID,
precision/recall, CMMD -- with the VAE round trip and the real-vs-real floor beside
every number.

Its own entrypoint rather than a metric module: these compare whole image sets and need
no judge, which is what makes them the way to score CelebA. Which metrics run is
`evaluation.set_metrics` in the config. The CLIP and DINOv2 passes are heavy; run them on
the GPU machine.

    uv run evaluate_sets configs/pools/celeba.yaml
"""

from evaluation.sets import run_sets
from utils.config import PoolRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, PoolRunConfig)
    if cli_seed is not None:
        cfg.generation.seeds = [cli_seed]

    print(f"Scoring the pool at {cfg.dataset_dir}")
    run_sets(cfg, resolve_device())


if __name__ == "__main__":
    main()
