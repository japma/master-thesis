"""Figures for a meeting, straight from `results/*.csv`.

One panel per factor rather than one bar group per factor: with eight model variants a
grouped chart is two dozen bars and a legend nobody reads, while a horizontal bar
carries the model's name on the axis beside it.

    uv run plot_results                                     # colour_mnist_skewed
    uv run plot_results --dataset colour_mnist_uniform
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR = "#2a78d6"
CEILING = "#6b7280"
FLOOR = "#1f2328"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SURFACE = "#fcfcfb"

FACTORS = ["digit", "fg", "bg"]
FACTOR_LABELS = {"digit": "digit", "fg": "foreground", "bg": "background"}

# Reading order in every figure: the plain circuit first, then its variants, then the
# joint PC and the neural baselines.
MODEL_ORDER = ("cspn", "joint_pc", "nn")


def short_name(checkpoint: str, dataset: str) -> str:
    """`psinet_colour_mnist_skewed_anchored:v1` -> `cspn_anchored`."""
    name = checkpoint.split(":")[0].replace(dataset, "")
    name = name.replace("__", "_").strip("_")
    name = name.replace("psinet", "cspn").replace("nn_baseline", "nn")
    return name or "cspn"


def model_rank(name: str) -> tuple[int, str]:
    for rank, prefix in enumerate(MODEL_ORDER):
        if name.startswith(prefix):
            return rank, name
    return len(MODEL_ORDER), name


def style_axes(ax: plt.Axes) -> None:
    """Recessive grid and axes: the marks carry the chart, not the furniture."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


def scores(results: Path, dataset: str, std: float) -> pd.DataFrame:
    """One row per (model, factor): the generated score and the real-data ceiling."""
    path = results / "accuracy.csv"
    if not path.exists():
        raise SystemExit(f"No accuracy.csv in {results}")

    table = pd.read_csv(path)
    table = table[table["dataset"] == dataset]
    generated = table[
        (table["source"] == "generated") & (table["std_correction"] == std)
    ]
    if generated.empty:
        raise SystemExit(f"No rows for dataset={dataset} std_correction={std}")

    generated = generated.assign(
        model=generated["checkpoint"].map(lambda c: short_name(c, dataset))
    )
    ceiling = (
        table[table["source"] == "real"]
        .groupby("factor", as_index=False)["value"]
        .max()
        .rename(columns={"value": "ceiling"})
    )
    return generated.merge(ceiling, on="factor", how="left")


def plot_accuracy(table: pd.DataFrame, out: Path, dataset: str) -> Path:
    """One panel per factor, every model as a bar, the ceiling as a line."""
    models = sorted(table["model"].unique(), key=model_rank, reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.0 + 0.42 * len(models)), sharey=True)
    fig.set_facecolor(SURFACE)

    for ax, factor in zip(axes, FACTORS, strict=True):
        style_axes(ax)
        rows = table[table["factor"] == factor].set_index("model")
        values = [rows["value"].get(model, float("nan")) for model in models]
        ax.barh(models, values, height=0.68, color=BAR, zorder=2)
        for y, value in enumerate(values):
            if pd.notna(value):
                # Inside the bar: outside, a long bar's label crosses the ceiling line.
                ax.text(
                    value - 0.02,
                    y,
                    f"{value:.2f}",
                    va="center",
                    ha="right",
                    fontsize=9,
                    color="#ffffff",
                )

        ceiling = rows["ceiling"].dropna()
        if not ceiling.empty:
            ax.axvline(
                ceiling.iloc[0],
                color=CEILING,
                linewidth=2,
                linestyle=(0, (4, 3)),
                zorder=3,
            )
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(FACTOR_LABELS[factor], color=INK, fontsize=11, loc="left")

    axes[0].set_ylabel("")
    fig.suptitle(
        f"Conditioning accuracy on {dataset}   (dashed line = real data)",
        color=INK,
        fontsize=13,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "accuracy_by_factor.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# The set metrics `evaluate_sets` writes: how to print each, and whether lower is better.
SET_METRICS = {
    "fid": (".1f", True),
    "kid": (".4f", True),
    "precision": (".3f", False),
    "recall": (".3f", False),
    "cmmd": (".3f", True),
    "fd_dinov2": (".1f", True),
}


def plot_set_metric(results: Path, out: Path, dataset: str, metric: str) -> Path | None:
    """One set metric per model, against its VAE round trip and real vs real."""
    path = results / f"{metric}.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table = table[table["dataset"] == dataset]
    if table.empty:
        return None
    table = table.assign(
        model=table["checkpoint"].map(lambda c: short_name(c, dataset))
    )

    by_source = table.groupby(["source", "model"])["value"]
    means, stds = by_source.mean(), by_source.std()
    models = sorted(means["generated"].index, key=model_rank, reverse=True)
    values = [float(means["generated"].get(m, float("nan"))) for m in models]
    errors = [float(stds["generated"].get(m, float("nan"))) for m in models]

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.42 * len(models)))
    fig.set_facecolor(SURFACE)
    style_axes(ax)
    ax.barh(models, values, height=0.6, color=BAR, xerr=errors, ecolor=INK, zorder=2)

    for y, model in enumerate(models):
        for source, color in (("reconstruction", CEILING), ("real", FLOOR)):
            tick = means.get(source, pd.Series(dtype=float)).get(model)
            if pd.notna(tick):
                ax.plot(
                    [tick, tick],
                    [y - 0.34, y + 0.34],
                    color=color,
                    linewidth=2.5,
                    zorder=4,
                )
        if pd.notna(values[y]):
            ax.text(
                values[y] * 1.02 + (errors[y] if pd.notna(errors[y]) else 0),
                y,
                f"{values[y]:{SET_METRICS[metric][0]}}",
                va="center",
                ha="left",
                fontsize=9,
                color=INK,
            )

    ax.set_xlim(0, max(v for v in values if pd.notna(v)) * 1.2)
    better = "lower" if SET_METRICS[metric][1] else "higher"
    ax.set_xlabel(f"{metric.upper()} ({better} is better)", color=INK)
    ax.set_title(
        f"{metric.upper()} on {dataset}   (grey = VAE round trip, black = real vs real)",
        color=INK,
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    figure = out / f"{metric}.png"
    fig.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure


def plot_steering(results: Path, out: Path, dataset: str) -> Path | None:
    """How far each factor moves the latents, against how far the data moves them."""
    path = results / "factor_steering.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table = table[table["dataset"] == dataset]
    if table.empty:
        return None
    table = table.drop_duplicates("factor").set_index("factor")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)

    values = [table["ratio"].get(factor, float("nan")) for factor in FACTORS]
    limit = max(2.3, max(values) * 1.25)
    ax.bar([FACTOR_LABELS[f] for f in FACTORS], values, width=0.55, color=BAR, zorder=2)
    for x, value in enumerate(values):
        ax.text(
            x,
            value - limit * 0.02,
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=10,
            color="#ffffff",
        )
    ax.axhline(1.0, color=CEILING, linewidth=2, linestyle=(0, (4, 3)), zorder=3)
    ax.text(
        2.42,
        1.06,
        "steers the latents as much as the data does",
        ha="right",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    ax.set_ylim(0, limit)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("model spread / data spread", color=INK)
    ax.set_title(
        "Does each label factor move the latents the way the data does?",
        color=INK,
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    figure = out / "factor_steering.png"
    fig.savefig(figure, dpi=200)
    plt.close(figure.parent and fig)
    return figure


def print_table(table: pd.DataFrame, results: Path, dataset: str) -> None:
    """Every number behind the figures, as markdown for the slides."""
    wide = table.pivot_table(index="model", columns="factor", values="value").reindex(
        columns=FACTORS
    )
    wide = wide.reindex(sorted(wide.index, key=model_rank))

    for metric in SET_METRICS:
        path = results / f"{metric}.csv"
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        rows = rows[(rows["dataset"] == dataset) & (rows["source"] == "generated")]
        if not rows.empty:
            named = (
                rows.assign(
                    model=rows["checkpoint"].map(lambda c: short_name(c, dataset))
                )
                .groupby("model")["value"]
                .mean()
            )
            wide[metric] = [named.get(m) for m in wide.index]

    ceilings = table.drop_duplicates("factor").set_index("factor")["ceiling"]
    wide.loc["real data (ceiling)"] = [ceilings.get(f) for f in FACTORS] + [
        float("nan") for c in wide.columns if c in SET_METRICS
    ]

    columns = list(wide.columns)
    print(f"\n### {dataset}\n")
    print("| model | " + " | ".join(columns) + " |")
    print("|" + "---|" * (len(columns) + 1))
    for name, row in wide.iterrows():
        cells = " | ".join(
            "--" if pd.isna(v) else f"{v:{SET_METRICS.get(c, ('.3f',))[0]}}"
            for c, v in zip(columns, row, strict=True)
        )
        print(f"| {name} | {cells} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--dataset", default="colour_mnist_skewed")
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    table = scores(args.results, args.dataset, args.std)
    written = [plot_accuracy(table, out, args.dataset)]
    written += [
        figure
        for figure in (
            *(
                plot_set_metric(args.results, out, args.dataset, metric)
                for metric in SET_METRICS
            ),
            plot_steering(args.results, out, args.dataset),
        )
        if figure is not None
    ]
    print_table(table, args.results, args.dataset)
    print("\nwrote:")
    for figure in written:
        print(f"  {figure}")


if __name__ == "__main__":
    main()
