#!/bin/bash
# train_colour_mnist.sh
# Trains the whole colour-MNIST stack for one or more dataset variants: the plain and the
# anchored autoencoder, then a CSPN, joint PCs and both neural baselines on each of them.
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
#CLASSIFIER="checkpoints/digit_classifier_colour_mnist_uniform.pt"

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
# Each entry is "label|command|config template|autoencoder step it needs", with @ standing
# in for the variant. A label ending in ? is optional: skipped where the variant has no
# such config (rgb has no anchored autoencoder, for instance). Every latent-space config
# names its autoencoder with `tag: latest`, so it picks up the one trained just before.
STEPS=(
    "autoencoder|train_ae|configs/autoencoder/colour_mnist_@.yaml|"
    "autoencoder_anchored?|train_ae|configs/autoencoder/colour_mnist_@_anchored.yaml|"
    "cspn|train_cspn|configs/cspn/colour_mnist_@.yaml|autoencoder"
    "cspn_anchored?|train_cspn|configs/cspn/colour_mnist_@_anchored.yaml|autoencoder_anchored"
    "joint_pc|train_joint_pc|configs/joint_pc/colour_mnist_@.yaml|autoencoder"
    "joint_pc_anchored?|train_joint_pc|configs/joint_pc/colour_mnist_@_anchored.yaml|autoencoder_anchored"
    "joint_pc_digit_only?|train_joint_pc|configs/joint_pc/colour_mnist_@_digit_only.yaml|autoencoder"
    "joint_pc_x2?|train_joint_pc|configs/joint_pc/colour_mnist_@_x2.yaml|autoencoder"
    "baseline_deterministic|train_nn_baseline|configs/nn_baseline/colour_mnist_@_deterministic.yaml|autoencoder"
    "baseline_mixture|train_nn_baseline|configs/nn_baseline/colour_mnist_@_mixture.yaml|autoencoder"
    "baseline_deterministic_anchored?|train_nn_baseline|configs/nn_baseline/colour_mnist_@_anchored_deterministic.yaml|autoencoder_anchored"
    "baseline_mixture_anchored?|train_nn_baseline|configs/nn_baseline/colour_mnist_@_anchored_mixture.yaml|autoencoder_anchored"
)

# Split one STEPS entry for `variant` into LABEL, OPTIONAL, COMMAND, CONFIG, NEEDS.
parse_step() {
    local step="$1" variant="$2"
    IFS='|' read -r LABEL COMMAND CONFIG NEEDS <<< "$step"
    OPTIONAL=0
    if [[ "$LABEL" == *"?" ]]; then
        OPTIONAL=1
        LABEL="${LABEL%\?}"
    fi
    CONFIG="${CONFIG//@/$variant}"
}

# --- preflight: every config and dataset present, before anything long starts ---------
missing=0
for variant in "${VARIANTS[@]}"; do
    datasets=("$variant")
    [[ -f "configs/joint_pc/colour_mnist_${variant}_x2.yaml" ]] && datasets+=("${variant}_x2")
    for dataset in "${datasets[@]}"; do
        if [[ ! -d "data/colour-mnist/$dataset/train" ]]; then
            echo "Missing dataset: data/colour-mnist/$dataset/train" >&2
            echo "  generate it with: uv run generate_colour_mnist configs/colour_mnist/$dataset.csv" >&2
            missing=1
        fi
    done
    for step in "${STEPS[@]}"; do
        parse_step "$step" "$variant"
        if [[ ! -f "$CONFIG" && $OPTIONAL -eq 0 ]]; then
            echo "Missing config: $CONFIG" >&2
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

# --- the eval judge, once: variant-independent and needed by evaluate_samples ---------
if [[ ! -f "$CLASSIFIER" ]]; then
    echo "No digit classifier at $CLASSIFIER -- training it now."
    if ! uv run train_classifier configs/classifier/colour_mnist_uniform.yaml; then
        echo "WARNING: classifier training failed; evaluate_samples has nothing to judge with"
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

    # Autoencoder steps that failed; everything trained on one reads its artifact, so
    # without it those runs would only fail more slowly.
    BROKEN=()
    for step in "${STEPS[@]}"; do
        parse_step "$step" "$variant"
        if [[ ! -f "$CONFIG" ]]; then
            echo "  (no $CONFIG -- skipping $variant/$LABEL)"
            echo
            continue
        fi
        if [[ -n "$NEEDS" && " ${BROKEN[*]-} " == *" $NEEDS "* ]]; then
            echo "  skipping $variant/$LABEL: its autoencoder ($NEEDS) failed"
            echo
            FAILED+=("$variant/$LABEL (skipped)")
            continue
        fi
        if ! run_step "$variant/$LABEL" "$COMMAND" "$CONFIG"; then
            [[ -z "$NEEDS" ]] && BROKEN+=("$LABEL")
        fi
    done
done

echo "========================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
    echo
    echo "Next: uv run generate_pools   configs/pools/colour_mnist_<variant>.yaml"
    echo "Then: uv run evaluate_samples configs/pools/colour_mnist_<variant>.yaml"
    echo "      uv run evaluate_sets    configs/pools/colour_mnist_<variant>.yaml"
else
    echo "${#FAILED[@]} run(s) failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
