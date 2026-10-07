#!/bin/bash
# train_celeba_beta0p5.sh
# The CelebA stack of train_celeba.sh on the beta 0.5 VAE, variational_celeba:v4.
# Usage: bash train_celeba_beta0p5.sh [--dry-run] [--compile] ...

VAE_TAG=v4 VAE_KIND=variational_beta0p5 exec bash "$(dirname "$0")/train_celeba.sh" "$@"
