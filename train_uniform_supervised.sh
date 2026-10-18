#!/bin/bash
# train_uniform_supervised.sh
# Trains the supervised uniform colour-MNIST stack: the supervised autoencoder, then on
# it a CSPN and both neural baselines -- the models configs/pools/colour_mnist_uniform.yaml
# lists as `*_uniform_supervised*`, without the joint PC.
#
#   bash train_uniform_supervised.sh                # everything
#   SKIP_AE=1 bash train_uniform_supervised.sh      # reuse the supervised VAE on wandb
#   bash train_uniform_supervised.sh --resume       # extra flags for every run
#   COMPILE= bash train_uniform_supervised.sh       # without torch.compile

set -uo pipefail

source .venv/bin/activate

COMPILE="${COMPILE---compile}"
SKIP_AE="${SKIP_AE:-0}"
EXTRA=("$@")
WEIGHTS="configs/colour_mnist/uniform.csv"
DATA="data/colour-mnist/uniform/train"

AE_STEP="autoencoder|train_ae|configs/autoencoder/colour_mnist_uniform_supervised.yaml"
DOWNSTREAM_STEPS=(
    "cspn|train_cspn|configs/cspn/colour_mnist_uniform_supervised.yaml"
    "baseline_deterministic|train_nn_baseline|configs/nn_baseline/colour_mnist_uniform_supervised_deterministic.yaml"
    "baseline_mixture|train_nn_baseline|configs/nn_baseline/colour_mnist_uniform_supervised_mixture.yaml"
)

# --- preflight: every config present before anything long starts ---------------------
missing=0
for step in "$AE_STEP" "${DOWNSTREAM_STEPS[@]}"; do
    IFS='|' read -r label _ config <<< "$step"
    if [[ ! -f "$config" ]]; then
        echo "Missing config for $label: $config" >&2
        missing=1
    fi
done
if [[ $missing -eq 1 ]]; then
    exit 2
fi

echo "Skip AE   : $SKIP_AE"
echo "Compile   : ${COMPILE:-(off)}"
echo "Extra     : ${EXTRA[*]-(none)}"
echo

if [[ ! -d "$DATA" ]]; then
    echo "No dataset at $DATA -- generating it from $WEIGHTS."
    if ! uv run generate_colour_mnist "$WEIGHTS"; then
        echo "Dataset generation failed." >&2
        exit 1
    fi
    echo
fi

FAILED=()

run_step() {
    local label="$1" command="$2" config="$3"
    echo "========================================"
    echo "  $label"
    echo "  uv run $command $config ${COMPILE} ${EXTRA[*]-}"
    echo "  started $(date)"
    echo "========================================"

    if uv run "$command" "$config" ${COMPILE} ${EXTRA[@]+"${EXTRA[@]}"} < /dev/null; then
        echo "  finished $label at $(date)"
        echo
        return 0
    fi

    echo "  FAILED $label at $(date)"
    echo
    FAILED+=("$label")
    return 1
}

# --- the autoencoder first: every downstream config reads it with `tag: latest` -------
if [[ "$SKIP_AE" != "1" ]]; then
    IFS='|' read -r label command config <<< "$AE_STEP"
    if ! run_step "$label" "$command" "$config"; then
        echo "The supervised VAE failed to train; skipping everything on it." >&2
        exit 1
    fi
fi

for step in "${DOWNSTREAM_STEPS[@]}"; do
    IFS='|' read -r label command config <<< "$step"
    run_step "$label" "$command" "$config"
done

echo "========================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
    echo
    echo "Next: uv run generate_pools    configs/pools/colour_mnist_uniform.yaml"
    echo "Then: uv run evaluate_samples  configs/pools/colour_mnist_uniform.yaml"
    echo "      uv run evaluate_sets     configs/pools/colour_mnist_uniform.yaml"
    echo "      uv run evaluate_marginal configs/pools/colour_mnist_uniform.yaml"
else
    echo "${#FAILED[@]} run(s) failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
