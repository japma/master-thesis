#!/bin/bash
# train_autoencoders.sh
# Usage: bash train_autoencoders.sh

set -e

source .venv/bin/activate

configs=(
    "ae/mnist.yaml"
    "ae/flowers.yaml"
    "ae/cub.yaml"
)

for config in "${configs[@]}"; do
    echo "========================================"
    echo "Running config: $config"
    echo "Started at: $(date)"
    echo "========================================"

    python scripts/train_ae.py "$config"

    echo "Finished $config at $(date)"
    echo ""
done

echo "All experiments done."