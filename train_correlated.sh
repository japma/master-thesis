#!/bin/bash
# train_correlated.sh
# Trains the correlated colour-MNIST stack: three autoencoders (plain, anchored,
# supervised), then on each of them a CSPN with and without label dropout and both
# neural baselines -- 3 + 3 x 4 = 15 runs.
#
#   bash train_correlated.sh                          # everything
#   bash train_correlated.sh anchored supervised      # only these VAE kinds
#   SKIP_AE=1 bash train_correlated.sh anchored       # reuse the anchored VAE on wandb
#   bash train_correlated.sh -- --resume              # extra flags for every run
#   COMPILE= bash train_correlated.sh                 # without torch.compile
#   JOINT_PC=1 bash train_correlated.sh variational   # also the joint PC (plain VAE only)

set -uo pipefail

source .venv/bin/activate

ALL_KINDS=(variational anchored supervised)
COMPILE="${COMPILE---compile}"
SKIP_AE="${SKIP_AE:-0}"
JOINT_PC="${JOINT_PC:-0}"
WEIGHTS="configs/colour_mnist/correlated.csv"
DATA="data/colour-mnist/correlated/train"

# --- arguments: VAE kinds, then optional `--` followed by flags for every run ---------
KINDS=()
EXTRA=()
after_separator=0
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        after_separator=1
        continue
    fi
    if [[ $after_separator -eq 1 ]]; then
        EXTRA+=("$arg")
    else
        KINDS+=("$arg")
    fi
done
if [[ ${#KINDS[@]} -eq 0 ]]; then
    KINDS=("${ALL_KINDS[@]}")
fi

# The config suffix each VAE kind's files carry; the plain VAE has none.
suffix() {
    case "$1" in
        variational) echo "" ;;
        anchored) echo "_anchored" ;;
        supervised) echo "_supervised" ;;
        *) return 1 ;;
    esac
}

# --- what to run: "label|command|config" per step ------------------------------------
ae_step() {
    echo "$1/autoencoder|train_ae|configs/autoencoder/colour_mnist_correlated$(suffix "$1").yaml"
}

downstream_steps() {
    local s
    s="$(suffix "$1")"
    echo "$1/cspn|train_cspn|configs/cspn/colour_mnist_correlated${s}.yaml"
    echo "$1/cspn_dontcare|train_cspn|configs/cspn/colour_mnist_correlated${s}_dontcare.yaml"
    echo "$1/baseline_deterministic|train_nn_baseline|configs/nn_baseline/colour_mnist_correlated${s}_deterministic.yaml"
    echo "$1/baseline_mixture|train_nn_baseline|configs/nn_baseline/colour_mnist_correlated${s}_mixture.yaml"
    if [[ "$JOINT_PC" == "1" && "$1" == "variational" ]]; then
        echo "$1/joint_pc|train_joint_pc|configs/joint_pc/colour_mnist_correlated.yaml"
    fi
}

# --- preflight: every config present before anything long starts ---------------------
missing=0
for kind in "${KINDS[@]}"; do
    if ! suffix "$kind" > /dev/null; then
        echo "Unknown VAE kind '$kind'. Known: ${ALL_KINDS[*]}" >&2
        exit 2
    fi
    while IFS='|' read -r label command config; do
        if [[ ! -f "$config" ]]; then
            echo "Missing config for $label: $config" >&2
            missing=1
        fi
    done < <(ae_step "$kind"; downstream_steps "$kind")
done
if [[ $missing -eq 1 ]]; then
    exit 2
fi

echo "VAE kinds : ${KINDS[*]}"
echo "Skip AE   : $SKIP_AE"
echo "Joint PC  : $JOINT_PC"
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

    # stdin from /dev/null: the step lists are fed to `while read` loops on stdin.
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

# --- autoencoders first: every downstream config reads its VAE with `tag: latest` -----
BROKEN=()
if [[ "$SKIP_AE" != "1" ]]; then
    for kind in "${KINDS[@]}"; do
        IFS='|' read -r label command config < <(ae_step "$kind")
        run_step "$label" "$command" "$config" || BROKEN+=("$kind")
    done
fi

# --- then everything trained on them --------------------------------------------------
for kind in "${KINDS[@]}"; do
    if [[ " ${BROKEN[*]-} " == *" $kind "* ]]; then
        echo "  skipping everything on the $kind VAE: it failed to train"
        echo
        while IFS='|' read -r label _ _; do
            FAILED+=("$label (skipped)")
        done < <(downstream_steps "$kind")
        continue
    fi
    while IFS='|' read -r label command config; do
        run_step "$label" "$command" "$config"
    done < <(downstream_steps "$kind")
done

echo "========================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
    echo
    echo "Next: uv run generate_pools    configs/pools/colour_mnist_correlated.yaml"
    echo "Then: uv run evaluate_samples  configs/pools/colour_mnist_correlated.yaml"
    echo "      uv run evaluate_marginal configs/evaluation/colour_mnist_correlated.yaml"
else
    echo "${#FAILED[@]} run(s) failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
