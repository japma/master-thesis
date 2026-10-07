#!/bin/bash
# train_celeba_tcvae.sh
# The CelebA stack of train_celeba.sh on the beta-TCVAE VAE, variational_celeba:v3.
# Usage: bash train_celeba_tcvae.sh [--dry-run] [--compile] ...

VAE_TAG=v3 VAE_KIND=variational_tcvae exec bash "$(dirname "$0")/train_celeba.sh" "$@"
