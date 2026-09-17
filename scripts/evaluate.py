"""Stage 2: score a run against a reference run and upsert the results table.

    uv run evaluate_run --run eval_runs/colour_mnist_uniform_test__cspn_std0.6__seed0 \\
        --reference eval_runs/colour_mnist_uniform_test__real__seed0 --metrics fid
"""

import argparse
from pathlib import Path

from evaluation.evaluate import METRICS, evaluate_run
from evaluation.results import upsert_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--results", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    rows = evaluate_run(
        args.run, args.reference, args.metrics, batch_size=args.batch_size
    )
    upsert_results(args.results, rows)
    for row in rows:
        print(f"{row['metric_name']:<20} {row['value']}")


if __name__ == "__main__":
    main()
