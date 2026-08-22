import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import umap
from torch._tensor import Tensor


def show(tensor: Tensor, title=None, width=8) -> None:
    grid = vutils.make_grid(tensor.cpu(), nrow=width, normalize=True)
    plt.figure(figsize=(16, 4), dpi=300)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_comparison(originals, reconstructions, title=None, dpi: int = 150) -> None:
    assert originals.shape == reconstructions.shape, (
        "originals and reconstructions must have the same shape"
    )

    n = originals.shape[0]

    # Stack as two blocks: first all originals, then all reconstructions
    # make_grid with nrow=n → row 1 = originals, row 2 = reconstructions
    combined = torch.cat(
        [originals.cpu(), reconstructions.cpu()], dim=0
    )  # (2n, C, H, W)

    grid = vutils.make_grid(combined, nrow=n, normalize=True, padding=2)

    fig_width = max(n * 1.5, 6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.5), dpi=dpi)
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")

    ax.text(
        -0.01,
        0.75,
        "Original",
        va="center",
        ha="right",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        -0.01,
        0.25,
        "Recon",
        va="center",
        ha="right",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_latent_comparison():
    raise NotImplementedError


def _get_class_colors(
    labels: torch.Tensor,
) -> tuple[list[int], dict[int, tuple[float, float, float, float]]]:
    classes: list[int] = sorted(np.unique(labels.detach().cpu().numpy()).tolist())
    n_classes: int = len(classes)

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    class_to_color: dict[int, tuple[float, float, float, float]] = {
        c: cmap(i / max(n_classes - 1, 1)) for i, c in enumerate(classes)
    }
    return classes, class_to_color


def plot_latent_space(
    latents: torch.Tensor,
    labels: torch.Tensor,
    title: str = "Latent space",
    class_names: list[str] | None = None,
) -> None:
    a = latents.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    classes, class_to_color = _get_class_colors(labels)

    reducer: umap.UMAP = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(a)

    _, ax = plt.subplots(figsize=(9, 8))

    for cls in classes:
        mask = labels_np == cls
        if not mask.any():
            continue
        label_str: str = class_names[cls] if class_names is not None else str(cls)
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[class_to_color[cls]],
            marker="x",
            alpha=0.7,
            s=60,
            label=label_str,
        )

    ax.legend(
        title="Class", loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0
    )

    ax.set_title(title)
    ax.set_xlabel("UMAP component 1")
    ax.set_ylabel("UMAP component 2")
    plt.tight_layout()
    plt.show()


def plot_latent_space_comparison(
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    labels: torch.Tensor,
    model_names: tuple[str, str] = ("Model A", "Model B"),
    title: str = "Latent space comparison",
    class_names: list[str] | None = None,
    markers: tuple[str, str] = ("x", "o"),
    joint_projection: bool = True,
) -> None:
    """Compare the latent spaces of two models sharing the same labels."""
    a_np = latents_a.detach().cpu().numpy()
    b_np = latents_b.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    classes, class_to_color = _get_class_colors(labels)

    reducer: umap.UMAP = umap.UMAP(n_components=2)

    projected_a: np.ndarray
    projected_b: np.ndarray
    if joint_projection:
        combined = np.concatenate([a_np, b_np], axis=0)
        projected = reducer.fit_transform(combined)
        n_a: int = a_np.shape[0]
        projected_a, projected_b = projected[:n_a], projected[n_a:]
    else:
        projected_a = reducer.fit_transform(a_np)
        projected_b = umap.UMAP(n_components=2).fit_transform(b_np)

    _, ax = plt.subplots(figsize=(10, 8))

    for projected, model_name, marker in zip(
        (projected_a, projected_b), model_names, markers, strict=False
    ):
        for cls in classes:
            mask = labels_np == cls
            if not mask.any():
                continue
            class_str: str = class_names[cls] if class_names is not None else str(cls)
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                c=[class_to_color[cls]],
                marker=marker,
                alpha=0.7,
                s=60,
                label=f"{class_str} ({model_name})",
            )

    class_handles: list[plt.Line2D] = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=class_to_color[cls],
            markersize=8,
            label=class_names[cls] if class_names is not None else str(cls),
        )
        for cls in classes
    ]
    model_handles: list[plt.Line2D] = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            markersize=8,
            label=model_name,
        )
        for model_name, marker in zip(model_names, markers, strict=False)
    ]

    legend1 = ax.legend(
        handles=class_handles,
        title="Class",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )

    ax.set_title(title)
    ax.set_xlabel("UMAP component 1")
    ax.set_ylabel("UMAP component 2")
    plt.tight_layout()
    plt.show()


def plot_latent_comparison_multiclass(
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    labels: torch.Tensor,
    reference: torch.Tensor | None = None,
    name_a: str = "Model A",
    name_b: str = "Model B",
    name_reference: str = "Ground truth",
    title: str = "Latent space comparison",
    class_names: list[str] | None = None,
) -> None:
    a = latents_a.detach().cpu().numpy()
    b = latents_b.detach().cpu().numpy()

    tensors = [a, b]
    if reference is not None:
        ref = reference.detach().cpu().numpy()
        tensors.append(ref)

    combined = np.concatenate(tensors, axis=0)

    classes = sorted(np.unique(labels).tolist())
    n_classes = len(classes)

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    class_to_color = {c: cmap(i / max(n_classes - 1, 1)) for i, c in enumerate(classes)}

    reducer = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(combined)

    n_a = len(a)
    n_b = len(b)
    proj_a = projected[:n_a]
    proj_b = projected[n_a : n_a + n_b]

    _, ax = plt.subplots(figsize=(9, 8))

    sources = [(proj_a, labels, "o", name_a, 60), (proj_b, labels, "x", name_b, 60)]
    if reference is not None:
        proj_ref = projected[n_a + n_b :]
        sources.append((proj_ref, labels, "s", name_reference, 80))

    for proj, labels, marker, source_name, size in sources:
        for cls in classes:
            mask = labels == cls
            if not mask.any():
                continue
            label_str = class_names[cls] if class_names is not None else str(cls)
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[class_to_color[cls]],
                marker=marker,
                alpha=0.7,
                s=size,
                label=f"{source_name} — class {label_str}" if mask.any() else None,
            )

    from matplotlib.lines import Line2D

    color_proxies = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=class_to_color[c],
            markersize=8,
            label=class_names[c] if class_names else f"Class {c}",
        )
        for c in classes
    ]
    shape_proxies = [
        Line2D(
            [0], [0], marker=m, color="grey", markersize=8, linestyle="None", label=name
        )
        for m, name in [("o", name_a), ("s", name_b)]
        + ([("x", name_reference)] if reference is not None else [])
    ]

    legend_classes = ax.legend(
        handles=color_proxies,
        title="Class",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    ax.legend(
        handles=shape_proxies,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )
    ax.add_artist(legend_classes)

    ax.set_title(title)
    ax.set_xlabel("UMAP component 1")
    ax.set_ylabel("UMAP component 2")
    plt.tight_layout()
    plt.show()


def plot_combination_heatmap(
    values,
    held_out=None,
    title: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Heatmap of a (digit, fg, bg) table: rows are digits, columns grouped by background.

    Combinations flagged in `held_out` (a boolean table of the same shape) get a red
    outline, so "did the held-out cells behave differently" is answerable at a glance.
    """
    import numpy as np
    from matplotlib.patches import Rectangle

    from dataset_loaders.colour_mnist import (
        BG_NAMES,
        FG_NAMES,
        NUM_BG,
        NUM_DIGITS,
        NUM_FG,
    )

    values = np.asarray(values)
    grid = np.full((NUM_DIGITS, NUM_FG * NUM_BG), np.nan)
    for fg in range(NUM_FG):
        for bg in range(NUM_BG):
            grid[:, bg * NUM_FG + fg] = values[:, fg, bg]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.025)

    ax.set_xticks(range(NUM_FG * NUM_BG))
    ax.set_xticklabels(
        [FG_NAMES[i % NUM_FG][:2] for i in range(NUM_FG * NUM_BG)], fontsize=7
    )
    for bg in range(NUM_BG):
        centre = bg * NUM_FG + (NUM_FG - 1) / 2
        ax.text(centre, -0.85, BG_NAMES[bg], ha="center", fontsize=9)
        if bg:
            ax.axvline(bg * NUM_FG - 0.5, color="white", linewidth=2)

    ax.set_yticks(range(NUM_DIGITS))
    ax.set_yticklabels([str(d) for d in range(NUM_DIGITS)], fontsize=8)
    ax.set_ylabel("digit")

    if held_out is not None:
        held_out = np.asarray(held_out)
        for digit in range(NUM_DIGITS):
            for fg in range(NUM_FG):
                for bg in range(NUM_BG):
                    if held_out[digit, fg, bg]:
                        ax.add_patch(
                            Rectangle(
                                (bg * NUM_FG + fg - 0.5, digit - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="red",
                                linewidth=1.5,
                            )
                        )

    ax.set_title(f"{title}   (red = held out of training)", pad=32)
    fig.tight_layout()
    return fig
