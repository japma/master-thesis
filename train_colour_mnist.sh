#!/bin/bash
# train_colour_mnist.sh
# Trains the whole colour-MNIST stack -- autoencoder, CSPN, joint PC, and both neural
# baselines -- for one or more dataset variants.
#
#   bash train_colour_mnist.sh skewed
#   bash train_colour_mnist.sh rgb uniform
#   bash train_colour_mnist.sh                      # all three, in order
#   bash train_colour_mnist.sh skewed -- --resume   # extra flags for every run
#   COMPILE= bash train_colour_mnist.sh skewed      # without torch.compile

set -uo pipefail

source .venv/bin/activate

ALL_VARIANTS=(uniform skewed rgb)
# `COMPILE= bash ...` disables it; unset means compile, which is the default here.
COMPILE="${COMPILE---compile}"
CLASSIFIER="checkpoints/digit_classifier_colour_mnist.pt"

# --- arguments: variants, then optional `--` followed by flags for every run ----------
VARIANTS=()
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
        VARIANTS+=("$arg")
    fi
done
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

for variant in "${VARIANTS[@]}"; do
    found=0
    for known in "${ALL_VARIANTS[@]}"; do
        [[ "$variant" == "$known" ]] && found=1
    done
    if [[ $found -eq 0 ]]; then
        echo "Unknown variant '$variant'. Known: ${ALL_VARIANTS[*]}" >&2
        exit 2
    fi
done

# --- what to run, in dependency order ------------------------------------------------
# Each entry is "label|command|config template", with @ standing in for the variant.
STEPS=(
    "autoencoder|train_ae|configs/autoencoder/colour_mnist_@.yaml"
    "cspn|train_cspn|configs/cspn/colour_mnist_@.yaml"
    "joint_pc|train_joint_pc|configs/joint_pc/colour_mnist_@.yaml"
    "baseline_deterministic|train_nn_baseline|configs/nn_baseline/colour_mnist_@_deterministic.yaml"
    "baseline_mixture|train_nn_baseline|configs/nn_baseline/colour_mnist_@_mixture.yaml"
)

# --- preflight: every config and dataset present, before anything long starts ---------
missing=0
for variant in "${VARIANTS[@]}"; do
    if [[ ! -d "data/colour-mnist/$variant/train" ]]; then
        echo "Missing dataset: data/colour-mnist/$variant/train" >&2
        echo "  generate it with: uv run generate_colour_mnist configs/colour_mnist/$variant.csv" >&2
        missing=1
    fi
    for step in "${STEPS[@]}"; do
        config="${step##*|}"
        config="${config//@/$variant}"
        if [[ ! -f "$config" ]]; then
            echo "Missing config: $config" >&2
            missing=1
        fi
    done
done
if [[ $missing -eq 1 ]]; then
    exit 2
fi

echo "Variants : ${VARIANTS[*]}"
echo "Compile  : ${COMPILE:-(off)}"
echo "Extra    : ${EXTRA[*]-(none)}"
echo

# --- the eval judge, once: variant-independent and needed by eval_model ---------------
if [[ ! -f "$CLASSIFIER" ]]; then
    echo "No digit classifier at $CLASSIFIER -- training it now."
    echo "It judges generated digits during evaluation and is deliberately trained on"
    echo "the uniform variant, so it is not itself weaker on combinations a model never saw."
    if ! uv run train_digit_classifier; then
        echo "WARNING: classifier training failed; eval_model will need --skip digit"
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

    if uv run "$command" "$config" ${COMPILE} ${EXTRA[@]+"${EXTRA[@]}"}; then
        echo "  finished $label at $(date)"
        echo
        return 0
    fi

    echo "  FAILED $label at $(date)"
    echo
    FAILED+=("$label")
    return 1
}

for variant in "${VARIANTS[@]}"; do
    echo "########################################"
    echo "# colour-MNIST variant: $variant"
    echo "########################################"
    echo

    for step in "${STEPS[@]}"; do
        label="${step%%|*}"
        rest="${step#*|}"
        command="${rest%%|*}"
        config="${rest#*|}"
        config="${config//@/$variant}"

        if ! run_step "$variant/$label" "$command" "$config"; then
            # Everything downstream reads the autoencoder's artifact, so without it the
            # rest of this variant would only fail more slowly.
            if [[ "$label" == "autoencoder" ]]; then
                echo "Autoencoder failed for $variant -- skipping its remaining models."
                echo
                break
            fi
        fi
    done
done

echo "========================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
    echo
    echo "Next: uv run eval_model --model cspn --name psinet_colour_mnist_<variant> \\"
    echo "         --ae variational_colour_mnist_<variant> --variant <variant>"
else
    echo "${#FAILED[@]} run(s) failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
