#!/bin/bash
# run_experiments.sh
# Usage: bash run_experiments.sh

set -e  # stop on first error

source ./venv/bin/activate

configs=(
    "configs/cspn/mnist_psi.yaml"
    "configs/cspn/flowers_psi.yaml"
    "config/cspn/cub_psi.yaml"
)

for config in "${configs[@]}"; do
    echo "========================================"
    echo "Running config: $config"
    echo "Started at: $(date)"
    echo "========================================"

    python scripts/train_cspn.py "$config"

    echo "Finished $config at $(date)"
    echo ""
done

echo "All experiments done."