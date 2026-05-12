"""Visualization helpers for training diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


def save_reconstructions(originals, reconstructions, labels, path):
    assert originals.shape == reconstructions.shape
    num_images = len(originals)

    originals = originals.clamp(min=0, max=1)
    reconstructions = reconstructions.clamp(min=0, max=1)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    cmap = "gray" if originals.shape[1] == 1 else None

    for i in range(num_images):
        # label
        axes[0, i].set_title(f"Label: {labels[i].item()}", fontsize=10)

        # original
        axes[0, i].imshow(originals[i].permute(1, 2, 0).squeeze(), cmap=cmap)
        axes[0, i].axis("off")

        # reconstructed
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).squeeze(), cmap=cmap)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def _to_numpy(array_like):
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def save_latent_umap(
    latents,
    labels=None,
    path=None,
    *,
    title="Latent UMAP",
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
):

    latents_np = _to_numpy(latents)
    if latents_np.ndim != 2:
        raise ValueError(
            f"Expected latents with shape (num_samples, latent_dim), got {latents_np.shape}"
        )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(latents_np)

    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        labels_np = _to_numpy(labels).reshape(-1)
        if labels_np.shape[0] != embedding.shape[0]:
            raise ValueError(
                "labels must have the same number of entries as latents "
                f"({labels_np.shape[0]} != {embedding.shape[0]})"
            )
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels_np,
            cmap="tab10",
            s=12,
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=12, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    plt.tight_layout()

    if path is not None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)
    return embedding
