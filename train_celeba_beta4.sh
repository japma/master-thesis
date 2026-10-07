#!/bin/bash
# train_celeba_beta4.sh
# The CelebA stack of train_celeba.sh on the beta 4 VAE, variational_celeba:v2.
# Usage: bash train_celeba_beta4.sh [--dry-run] [--compile] ...

VAE_TAG=v2 VAE_KIND=variational_beta4 exec bash "$(dirname "$0")/train_celeba.sh" "$@"
