#!/bin/bash
# run_experiments.sh
# Usage: bash run_experiments.sh

set -e

source .venv/bin/activate

configs=(
    "cspn/mnist.yaml"
    "cspn/flowers.yaml"
    "cspn/cub.yaml"
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