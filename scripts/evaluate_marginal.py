"""Score marginalized queries: "a 0, colours unspecified".

A single pass rather than the two-stage pool harness: a pool stores one fully specified
label per sample and has nowhere to record which factors a query left free.

Every query runs the mixture reference -- free factors drawn from the training
conditional, the model asked a fully specified question -- and, for a model with its
own `p(y)`, the same query marginalized inside the circuit. Two CSVs land under
`evaluation.results_root`.

    uv run evaluate_marginal configs/evaluation/colour_mnist_skewed.yaml
"""

from evaluation.marginal import run_marginal
from utils.config import EvaluationRunConfig, load_config
from utils.reproducibility import resolve_device


def main() -> None:
    cfg, cli_seed, _ = load_config()
    assert isinstance(cfg, EvaluationRunConfig)
    if cli_seed is not None:
        cfg.generation.seed = cli_seed

    print(f"Querying {cfg.model.name} on {cfg.dataset.name}")
    run_marginal(cfg, resolve_device())


if __name__ == "__main__":
    main()
