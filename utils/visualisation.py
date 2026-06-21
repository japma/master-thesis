import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import numpy as np
import umap


def show(tensor, title=None, width=8):
    grid = vutils.make_grid(tensor.cpu(), nrow=width, normalize=True)
    plt.figure(figsize=(16, 4), dpi=300)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def plot_latent_comparison():
    raise NotImplementedError


def plot_latent_space(
    latents: torch.Tensor,
    labels: torch.Tensor,
    title: str = "Latent space",
    class_names: list[str] | None = None,
):
    a = latents.detach().cpu().numpy()

    classes = sorted(np.unique(labels).tolist())
    n_classes = len(classes)

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    class_to_color = {c: cmap(i / max(n_classes - 1, 1)) for i, c in enumerate(classes)}

    reducer = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(a)

    fig, ax = plt.subplots(figsize=(9, 8))

    for cls in classes:
        mask = labels == cls
        if not mask.any():
            continue
        label_str = class_names[cls] if class_names is not None else str(cls)
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
):
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

    fig, ax = plt.subplots(figsize=(9, 8))

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
