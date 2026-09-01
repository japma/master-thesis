#!/bin/bash
# train_autoencoders.sh
# Usage: bash train_autoencoders.sh

set -e

source .venv/bin/activate

ae_configs=(
    "configs/autoencoder/colour_mnist.yaml"
)

cspn_configs=(
    "configs/cspn/colour_mnist.yaml"
)

for config in "${ae_configs[@]}"; do
    echo "========================================"
    echo "Running config: $config"
    echo "Started at: $(date)"
    echo "========================================"

    uv run train_ae "$config"

    echo "Finished $config at $(date)"
    echo ""
done

for config in "${cspn_configs[@]}"; do
    echo "========================================"
    echo "Running config: $config"
    echo "Started at: $(date)"
    echo "========================================"

    uv run train_cspn "$config"

    echo "Finished $config at $(date)"
    echo ""
done

echo "All experiments done."
