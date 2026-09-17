"""The long-format results table: one row per run and metric."""

from pathlib import Path

import pandas as pd

from evaluation.run import RunManifest

RESULT_COLUMNS: list[str] = [
    "model",
    "dataset",
    "checkpoint",
    "seed",
    "metric_name",
    "value",
]
KEY_COLUMNS: list[str] = ["model", "dataset", "seed", "metric_name"]


def result_row(
    manifest: RunManifest, metric_name: str, value: float
) -> dict[str, object]:
    return {
        "model": manifest.model_name,
        "dataset": manifest.dataset,
        "checkpoint": manifest.checkpoint_path,
        "seed": manifest.seed,
        "metric_name": metric_name,
        "value": value,
    }


def upsert_results(table_path: Path, rows: list[dict[str, object]]) -> None:
    """Writes `rows` into the CSV at `table_path`, replacing rows with the same key."""
    new = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if table_path.exists():
        old = pd.read_csv(table_path)
        replaced = set(new[KEY_COLUMNS].itertuples(index=False, name=None))
        keep = [
            key not in replaced
            for key in old[KEY_COLUMNS].itertuples(index=False, name=None)
        ]
        table = pd.concat([old[keep], new], ignore_index=True)
    else:
        table = new

    table_path.parent.mkdir(parents=True, exist_ok=True)
    partial = table_path.with_suffix(".partial")
    table.sort_values(KEY_COLUMNS).to_csv(partial, index=False)
    partial.replace(table_path)
