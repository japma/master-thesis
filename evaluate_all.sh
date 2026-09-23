#!/bin/bash
# evaluate_all.sh
# Scores every colour-MNIST model there is an evaluation config for, then draws the
# figures. Digit accuracy, colour accuracy, colour drift/contrast and FID, all landing
# in results/*.csv where every row carries the checkpoint that produced it.
#
# Usage: bash evaluate_all.sh [skewed|uniform|all] [--skip-fid] [--regenerate]
#
#   --skip-fid    accuracies only. FID pushes every image through InceptionV3 at
#                 299x299, which dominates the runtime here.
#   --regenerate  re-sample pools that already exist (otherwise they are reused, and
#                 only the scoring re-runs).
#
# A config whose checkpoint is not on wandb is reported and skipped, so this is safe to
# run while training is still filling the grid in.

set -u

source .venv/bin/activate

VARIANTS=(skewed uniform)
SKIP_FID=0
REGENERATE=0
for arg in "$@"; do
    case "$arg" in
        skewed|uniform)  VARIANTS=("$arg") ;;
        all)             VARIANTS=(skewed uniform) ;;
        --skip-fid)      SKIP_FID=1 ;;
        --regenerate)    REGENERATE=1 ;;
        *)               echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

pool_dir() {
    # The pool path is derived from the config, so ask the config rather than rebuild it.
    uv run python - "$1" <<'PYEOF'
import sys, yaml
from pathlib import Path
from utils.config import EvaluationRunConfig
from utils.config.loading import _apply_dataset_defaults

raw = yaml.safe_load(Path(sys.argv[1]).read_text())
print(EvaluationRunConfig(**_apply_dataset_defaults(raw)).pool_dir)
PYEOF
}

DONE=()
FAILED=()

for variant in "${VARIANTS[@]}"; do
    echo
    echo "############ ${variant} ############"
    for config in configs/evaluation/colour_mnist_${variant}.yaml \
                  configs/evaluation/colour_mnist_${variant}_*.yaml; do
        [ -f "$config" ] || continue
        name="$(basename "$config" .yaml)"
        echo
        echo "=== ${name} ==="

        pool="$(pool_dir "$config")"
        if [ "$REGENERATE" -eq 1 ] || [ ! -d "$pool" ]; then
            if ! uv run generate_samples "$config"; then
                echo "  !! generation failed -- checkpoint missing? skipping ${name}"
                FAILED+=("$name")
                continue
            fi
        else
            echo "  reusing pool at ${pool}"
        fi

        uv run evaluate_samples "$config" || {
            echo "  !! scoring failed for ${name}"
            FAILED+=("$name")
            continue
        }

        if [ "$SKIP_FID" -eq 0 ]; then
            uv run evaluate_fid "$config" || echo "  !! fid failed for ${name}"
        fi
        DONE+=("$name")
    done

    echo
    echo "--- figures for ${variant} ---"
    uv run plot_results --dataset "colour_mnist_${variant}" \
        --out "results/figures/${variant}" || true
done

echo
echo "scored:  ${#DONE[@]}"
for name in "${DONE[@]:-}"; do [ -n "$name" ] && echo "  $name"; done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "skipped: ${#FAILED[@]}"
    for name in "${FAILED[@]}"; do echo "  $name"; done
fi
