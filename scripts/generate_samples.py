"""Sample a generative model, decode through its VAE, and cache the result.

Stage 1 of two. Writes a sample pool -- latents, decoded images, labels and a manifest
naming every checkpoint involved -- plus the reference pool that gives the scores a
ceiling. Nothing is measured here; `evaluate_samples` reads what this writes.

The schedule is stratified: every one of the 180 (digit, fg, bg) combinations gets
`generation.n_per_cell` samples, held-out combinations included.

    uv run generate_samples configs/evaluation/colour_mnist_uniform.yaml
    uv run generate_samples configs/evaluation/colour_mnist_uniform.yaml --seed 1
"""

from evaluation.generate import generate_pool
from utils.config import EvaluationRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, EvaluationRunConfig)
    if cli_seed is not None:
        # Keep the pool directory, which is derived from the seed, in step with it.
        cfg.generation.seed = cli_seed

    print(
        f"Sampling {cfg.model.name} ({cfg.model.model_type}) on {cfg.dataset.name} | "
        f"{cfg.generation.n_per_cell} per cell | seed={cfg.generation.seed}"
    )
    pool_dir = generate_pool(cfg, resolve_device())
    print(f"\nSample pool: {pool_dir}")
    print("Score it with:  uv run evaluate_samples <the same config>")


if __name__ == "__main__":
    main()
