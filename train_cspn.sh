#!/bin/bash
# train_autoencoders.sh
# Usage: bash train_autoencoders.sh

set -e

source .venv/bin/activate

configs=(
    "configs/cspn/mnist.yaml"
    "configs/cspn/colour_mnist.yaml"
    "configs/cspn/celeba.yaml"
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
