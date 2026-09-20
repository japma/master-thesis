#!/bin/bash
# train_missing.sh
# Trains the colour-MNIST models a variant needs, skipping whatever wandb already has.
# Usage: bash train_missing.sh [variant] [--dry-run] [--with-optional]
#
#   variant           colour-MNIST variant, default "skewed"
#   --dry-run         list what would be trained, train nothing
#   --with-optional   also the factorized CSPN (E1) and the mixture nn_baseline
#   --force           retrain everything, even what wandb already has (a new version
#                     of an artifact; the old one stays, so nothing is lost)
#
# Order matters: everything downstream trains on the autoencoder's latents, so a missing
# autoencoder is trained first and the rest follow in the same run.

set -e

source .venv/bin/activate

VARIANT="skewed"
DRY_RUN=0
WITH_OPTIONAL=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=1 ;;
        --with-optional) WITH_OPTIONAL=1 ;;
        --force)         FORCE=1 ;;
        --*)             echo "Unknown option: $arg" >&2; exit 2 ;;
        *)               VARIANT="$arg" ;;
    esac
done

DATASET="colour_mnist_${VARIANT}"

# What the calibration experiment needs, in dependency order:
#   artifact name | config | entrypoint
STEPS=(
    "variational_${DATASET}|configs/autoencoder/colour_mnist_${VARIANT}.yaml|train_ae"
    "psinet_${DATASET}|configs/cspn/colour_mnist_${VARIANT}.yaml|train_cspn"
    "joint_pc_${DATASET}|configs/joint_pc/colour_mnist_${VARIANT}.yaml|train_joint_pc"
    # The judge is deliberately trained on uniform for every variant, so that it is not
    # itself weaker on the combinations the model under test never saw.
    "digit_classifier_colour_mnist_uniform|configs/classifier/colour_mnist_uniform.yaml|train_classifier"
)

if [ "$WITH_OPTIONAL" -eq 1 ]; then
    STEPS+=(
        "psinet_${DATASET}_factorized|configs/cspn/colour_mnist_${VARIANT}_factorized.yaml|train_cspn"
        "nn_baseline_${DATASET}|configs/nn_baseline/colour_mnist_${VARIANT}_mixture.yaml|train_nn_baseline"
    )
fi

echo "Variant: ${VARIANT}   (dataset ${DATASET})"
[ "$FORCE" -eq 1 ] && echo "--force: retraining everything, existing checkpoints ignored"
echo "Asking wandb which checkpoints already exist ..."

EXISTING="$(uv run python - <<'PYEOF'
"""Every artifact collection in the project, one name per line."""
import wandb
from utils.wandb_utils import ENTITY, PROJECT

api = wandb.Api()
for artifact_type in api.artifact_types(f"{ENTITY}/{PROJECT}"):
    if artifact_type.name.startswith("wandb-"):
        continue
    for collection in artifact_type.collections():
        if not collection.name.startswith(("run-", "intermediate_")):
            print(collection.name)
PYEOF
)"
if [ -z "$EXISTING" ]; then
    echo "wandb returned no checkpoints. Run 'wandb login' and check the project in" >&2
    echo "utils/wandb_utils.py before trusting this -- training everything blindly" >&2
    echo "would overwrite work that may already exist." >&2
    exit 1
fi
echo "Found $(echo "$EXISTING" | grep -c . || true) checkpoint collections."
echo

TRAINED=0
SKIPPED=0
for step in "${STEPS[@]}"; do
    IFS='|' read -r artifact config entrypoint <<< "$step"

    if [ "$FORCE" -eq 0 ] && grep -qxF "$artifact" <<< "$EXISTING"; then
        echo "skip   ${artifact}  (already on wandb)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if [ ! -f "$config" ]; then
        echo "MISSING CONFIG  ${config}  -- cannot train ${artifact}" >&2
        exit 1
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would train  ${artifact}  <-  uv run ${entrypoint} ${config}"
        TRAINED=$((TRAINED + 1))
        continue
    fi

    echo "========================================"
    echo "Training ${artifact}"
    echo "  ${entrypoint} ${config}"
    echo "  started at $(date)"
    echo "========================================"
    uv run "$entrypoint" "$config"
    echo "Finished ${artifact} at $(date)"
    echo
    TRAINED=$((TRAINED + 1))
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run: ${TRAINED} to train, ${SKIPPED} already there."
else
    echo "Done: ${TRAINED} trained, ${SKIPPED} skipped."
fi
