"""Inference pipeline for loading checkpoints and generating visualizations."""

from models.autoencoder import AbstractAutoencoder
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import umap
import numpy as np

from models.cspn import AbstractCSPN
from scripts.visualization.visualization import save_latent_umap, _to_numpy


NUM_CLASSES = 10
SAMPLES_PER_CLASS = 100


def _collect_labeled_batch(
    data_loader: DataLoader, target_count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    images_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    collected = 0

    for images, labels in data_loader:
        remaining = target_count - collected
        if remaining <= 0:
            break
        take = min(images.shape[0], remaining)
        images_parts.append(images[:take].to(device))
        labels_parts.append(labels[:take].to(device))
        collected += take

    if collected < target_count:
        raise ValueError(
            f"Requested {target_count} samples, but data_loader only provided {collected}."
        )

    return torch.cat(images_parts, dim=0), torch.cat(labels_parts, dim=0)


def run_ae_inference(
    model: AbstractAutoencoder, data_loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run autoencoder inference and return latents and labels.

    Args:
        model: Autoencoder model to encode
        data_loader: Data loader with class labels
        device: Device to run on

    Returns:
        Tuple of (latents, labels)
    """

    model.eval()
    target_count = NUM_CLASSES * SAMPLES_PER_CLASS

    with torch.no_grad():
        sample_images, sample_labels = _collect_labeled_batch(
            data_loader=data_loader,
            target_count=target_count,
            device=device,
        )
        sampled_latents = model.encode(sample_images)

    save_latent_umap(sampled_latents, labels=sample_labels, path="ae.png")

    return sampled_latents, sample_labels


def run_cspn_inference(
    model: AbstractCSPN,
    data_loader: DataLoader | None,
    device: torch.device,
    autoencoder: AbstractAutoencoder | None = None,
    class_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CSPN inference and return sampled latents and labels.

    Args:
        model: CSPN model to sample from
        data_loader: Optional data loader (unused)
        device: Device to run on
        autoencoder: Optional autoencoder for visual sample generation
        class_names: Optional list of class names for visualization

    Returns:
        Tuple of (sampled_latents, labels)
    """
    model.eval()
    sample_labels = torch.arange(NUM_CLASSES, device=device).repeat(SAMPLES_PER_CLASS)
    with torch.no_grad():
        sampled_latents = model.sample(sample_labels)

    save_latent_umap(sampled_latents, labels=sample_labels, path="cspn.png")

    # Generate visual samples if autoencoder is available
    if autoencoder is not None:
        sample_and_visualize_cspn(
            model=model,
            autoencoder=autoencoder,
            device=device,
            class_names=class_names,
        )

    return sampled_latents, sample_labels


def sample_and_visualize_cspn(
    model: AbstractCSPN,
    autoencoder: AbstractAutoencoder,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    class_names: list[str] | None = None,
    path: str = "cspn_samples.png",
):
    """Sample from CSPN and visualize decoded samples as a grid (one per class).

    Args:
        model: CSPN model to sample from
        autoencoder: Autoencoder to decode latent samples to images
        device: Device to run on
        num_classes: Number of classes to sample from
        class_names: Optional list of class names for titles
        path: Path to save the visualization image
    """
    model.eval()
    autoencoder.eval()

    # Sample one image per class
    sample_labels = torch.arange(num_classes, device=device)

    with torch.no_grad():
        # Sample latents from CSPN
        sampled_latents = model.sample(sample_labels)
        # Decode to images
        decoded_images = autoencoder.decode(sampled_latents)

    # Ensure images are in [0, 1] range for visualization
    decoded_images = torch.clamp(decoded_images, 0, 1)

    # Create figure with grid (one column, num_classes rows)
    fig = plt.figure(figsize=(4, num_classes * 2))
    gs = gridspec.GridSpec(num_classes, 1, figure=fig)

    # Plot images
    for class_idx in range(num_classes):
        ax = fig.add_subplot(gs[class_idx, 0])
        img = decoded_images[class_idx]

        # Handle different image formats
        if img.dim() == 3:  # Assume C x H x W
            if img.shape[0] == 1:  # Grayscale
                img_np = img[0].cpu().numpy()
                ax.imshow(img_np, cmap="gray")
            else:  # RGB/Colored
                img_np = img.permute(1, 2, 0).cpu().numpy()
                ax.imshow(img_np)
        else:  # Already 2D
            img_np = img.cpu().numpy()
            ax.imshow(img_np, cmap="gray")

        # Use class name if provided, otherwise use index
        if class_names is not None and class_idx < len(class_names):
            title = class_names[class_idx]
        else:
            title = f"Class {class_idx}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved CSPN samples visualization to {path}")
    plt.close()


def save_combined_latent_umap(
    ae_latents: torch.Tensor,
    cspn_latents: torch.Tensor,
    ae_labels: torch.Tensor,
    cspn_labels: torch.Tensor,
    path: str = "combined_umap.png",
    title: str = "Autoencoder vs CSPN Latent Space",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
):
    """Visualize autoencoder and CSPN latent spaces in the same UMAP.

    Args:
        ae_latents: Autoencoder latent vectors (num_samples, latent_dim)
        cspn_latents: CSPN sampled latent vectors (num_samples, latent_dim)
        ae_labels: Labels for autoencoder samples
        cspn_labels: Labels for CSPN samples
        path: Path to save the visualization
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP metric parameter
        random_state: Random state for reproducibility
    """
    ae_latents_np = _to_numpy(ae_latents)
    cspn_latents_np = _to_numpy(cspn_latents)
    ae_labels_np = _to_numpy(ae_labels).reshape(-1)
    cspn_labels_np = _to_numpy(cspn_labels).reshape(-1)

    # Combine latents for fitting UMAP
    combined_latents = np.vstack([ae_latents_np, cspn_latents_np])

    # Fit UMAP on combined data
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    combined_embedding = reducer.fit_transform(combined_latents)

    # Split embeddings back
    ae_embedding = combined_embedding[: len(ae_latents_np)]
    cspn_embedding = combined_embedding[len(ae_latents_np) :]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10")

    # Plot autoencoder samples with circles
    for class_idx in range(NUM_CLASSES):
        mask_ae = ae_labels_np == class_idx
        ax.scatter(
            ae_embedding[mask_ae, 0],
            ae_embedding[mask_ae, 1],
            c=[cmap(class_idx)],
            marker="o",
            s=50,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            label=f"AE Class {class_idx}",
        )

    # Plot CSPN samples with stars
    for class_idx in range(NUM_CLASSES):
        mask_cspn = cspn_labels_np == class_idx
        ax.scatter(
            cspn_embedding[mask_cspn, 0],
            cspn_embedding[mask_cspn, 1],
            c=[cmap(class_idx)],
            marker="*",
            s=400,
            alpha=0.7,
            edgecolors="darkred",
            linewidths=0.8,
            label=f"CSPN Class {class_idx}",
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linewidth=0.3, alpha=0.3)

    # Add custom legend entries to distinguish models
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Autoencoder (circles)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markersize=15,
            markeredgecolor="darkred",
            markeredgewidth=0.8,
            label="CSPN (stars)",
        ),
    ]
    ax.legend(handles=custom_lines, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved combined UMAP visualization to {path}")
    plt.close(fig)
