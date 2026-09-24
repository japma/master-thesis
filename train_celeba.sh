#!/bin/bash
# train_celeba.sh
# Trains the CelebA CSPN and every baseline the CelebA pool compares it against, all on
# the `variational_celeba:best` latents.
# Usage: bash train_celeba.sh [--dry-run] [--compile] ...
#
# Every argument is passed to each trainer, so `--dry-run` runs one epoch of each with
# wandb off. A failed step is reported and the rest still run, so one crash does not
# cost the whole night.
#
# The GMM goes first: it takes minutes, and it fails fast if the VAE does not load.
# Afterwards:
#   uv run generate_pools configs/pools/celeba.yaml
#   uv run evaluate_sets  configs/pools/celeba.yaml

source .venv/bin/activate

#   config | entrypoint
STEPS=(
    "configs/gmm/celeba.yaml|fit_gmm"
    "configs/cspn/celeba.yaml|train_cspn"
    "configs/spn/celeba.yaml|train_spn"
    "configs/nn_baseline/celeba_deterministic.yaml|train_nn_baseline"
    "configs/nn_baseline/celeba_mixture.yaml|train_nn_baseline"
)

FAILED=()
for step in "${STEPS[@]}"; do
    IFS='|' read -r config entrypoint <<< "$step"

    echo "========================================"
    echo "${entrypoint} ${config} $*"
    echo "  started at $(date)"
    echo "========================================"
    if uv run "$entrypoint" "$config" "$@"; then
        echo "Finished ${config} at $(date)"
    else
        echo "FAILED ${config} at $(date)"
        FAILED+=("$config")
    fi
    echo
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "${#FAILED[@]} of ${#STEPS[@]} failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All ${#STEPS[@]} done."
