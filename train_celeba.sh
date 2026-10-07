#!/bin/bash
# train_celeba.sh
# Trains the CelebA CSPN and every baseline the CelebA pool compares it against, all on
# the `variational_celeba:best` latents.
# Usage: bash train_celeba.sh [--dry-run] [--compile] ...
#
# Every argument is passed to each trainer, so `--dry-run` runs one epoch of each with
# wandb off. A failed step is reported and the rest still run, so one crash does not
# cost the whole night.
#
# VAE_TAG trains the same stack on another `variational_celeba` version; VAE_KIND is
# the kind that version must have recorded, since it names every artifact trained on
# it. Without that check a VAE logged without its kind falls back to `variational`, and
# the runs land as new versions of the main models. The ablation wrappers set both:
#   train_celeba_beta4.sh  train_celeba_tcvae.sh  train_celeba_beta0p5.sh
#
# The GMM goes first: it takes minutes, and it fails fast if the VAE does not load.
# Afterwards:
#   uv run generate_pools configs/pools/celeba.yaml
#   uv run evaluate_sets  configs/pools/celeba.yaml

source .venv/bin/activate

VAE_TAG="${VAE_TAG:-best}"
VAE_KIND="${VAE_KIND:-}"

#   config | entrypoint
STEPS=(
    "configs/gmm/celeba.yaml|fit_gmm"
    "configs/cspn/celeba.yaml|train_cspn"
    "configs/spn/celeba.yaml|train_spn"
    "configs/nn_baseline/celeba_deterministic.yaml|train_nn_baseline"
    "configs/nn_baseline/celeba_mixture.yaml|train_nn_baseline"
)

if [ "$VAE_TAG" != "best" ]; then
    if [ -z "$VAE_KIND" ]; then
        echo "VAE_TAG=$VAE_TAG needs VAE_KIND, the kind that version recorded." >&2
        exit 2
    fi
    recorded=$(uv run python -c "
from utils.wandb_utils import artifact_metadata
from utils.naming import VAE_KIND_KEY
print(artifact_metadata('variational_celeba:$VAE_TAG').get(VAE_KIND_KEY, ''))
" | tail -n 1)
    if [ "$recorded" != "$VAE_KIND" ]; then
        echo "variational_celeba:$VAE_TAG records kind '${recorded:-(none)}', expected '$VAE_KIND'." >&2
        echo "Its runs would not be named apart from the main ones -- aborting." >&2
        exit 2
    fi

    # The configs with the tag swapped; dataset fragments still resolve from the repo root.
    CONFIG_DIR=$(mktemp -d)
    trap 'rm -rf "$CONFIG_DIR"' EXIT
    for i in "${!STEPS[@]}"; do
        IFS='|' read -r config entrypoint <<< "${STEPS[$i]}"
        rewritten="$CONFIG_DIR/$(dirname "$config" | tr / _)_$(basename "$config")"
        sed "s/^\(  tag: \)\"best\"$/\1\"$VAE_TAG\"/" "$config" > "$rewritten"
        if ! grep -q "^  tag: \"$VAE_TAG\"$" "$rewritten"; then
            echo "Could not set the autoencoder tag in $config" >&2
            exit 2
        fi
        STEPS[$i]="$rewritten|$entrypoint"
    done
fi

echo "VAE: variational_celeba:$VAE_TAG${VAE_KIND:+ ($VAE_KIND)}"
echo

FAILED=()
for step in "${STEPS[@]}"; do
    IFS='|' read -r config entrypoint <<< "$step"

    echo "========================================"
    echo "${entrypoint} ${config} $*"
    echo "  started at $(date)"
    echo "========================================"
    if uv run "$entrypoint" "$config" "$@"; then
        echo "Finished ${config} at $(date)"
    else
        echo "FAILED ${config} at $(date)"
        FAILED+=("$config")
    fi
    echo
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "${#FAILED[@]} of ${#STEPS[@]} failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All ${#STEPS[@]} done."
