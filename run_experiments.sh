#!/bin/bash
# run_experiments.sh
# Usage: bash run_experiments.sh

set -e

source .venv/bin/activate

configs=(
    "configs/cspn/mnist.yaml"
    "configs/cspn/flowers.yaml"
    "configs/cspn/cub.yaml"
)

for config in "${configs[@]}"; do
    echo "========================================"
    echo "Running config: $config"
    echo "Started at: $(date)"
    echo "========================================"

    uv run train_cspn "$config"

    echo "Finished $config at $(date)"
    echo ""
done

echo "All experiments done."
