#!/bin/bash
# train_autoencoders.sh
# Usage: bash train_autoencoders.sh

set -e

source .venv/bin/activate

configs=(
    "configs/autoencoder/mnist.yaml"
    "configs/autoencoder/mnist_beta8.yaml"
#    "configs/autoencoder/celeba.yaml"
#    "configs/autoencoder/flowers.yaml"
#    "configs/autoencoder/cub.yaml"
#    "configs/autoencoder/mnist_beta4.yaml"
#    "configs/autoencoder/mnist_beta8.yaml"
#    "configs/autoencoder/flowers_beta4.yaml"
#    "configs/autoencoder/flowers_beta8.yaml"
#    "configs/autoencoder/cub_beta4.yaml"
#    "configs/autoencoder/cub_beta8.yaml"
)

for config in "${configs[@]}"; do
    echo "========================================"
    echo "Running config: $config"
    echo "Started at: $(date)"
    echo "========================================"

    uv run train_ae "$config"

    echo "Finished $config at $(date)"
    echo ""
done

echo "All experiments done."
