"""Visualization helpers for training diagnostics."""

import os
import math

import matplotlib.pyplot as plt
import torch


def visualize_autoencoder(model, test_loader, device, output_dir, num_samples=10):
    """Visualize autoencoder reconstructions.

    Args:
        model: Trained autoencoder model.
        test_loader: DataLoader for test data.
        device: Device to run on (cpu or cuda).
        output_dir: Directory to save visualization.
        num_samples: Number of samples to visualize (default: 10).
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:num_samples].to(device)

    with torch.no_grad():
        reconstructed = model(images)

    reconstructed = reconstructed.view(images.shape)

    images = images.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))

    for i in range(num_samples):
        img = images[i].squeeze()
        # Handle both grayscale (H, W) and RGB (C, H, W) images
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        axes[0, i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        recon = reconstructed[i].squeeze()
        if recon.ndim == 3:
            recon = recon.permute(1, 2, 0)
        axes[1, i].imshow(recon, cmap="gray" if recon.ndim == 2 else None)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.tight_layout()

    save_path = os.path.join(output_dir, "reconstruction.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.close()


def visualize_losses(
    train_values,
    test_values,
    output_dir,
    filename,
    title,
    y_label,
    train_label="Train",
    test_label="Test",
):
    """Plot and save train/test curves.

    Args:
        train_values: Sequence of train metric values by epoch.
        test_values: Sequence of test/validation metric values by epoch.
        output_dir: Directory where the image is saved.
        filename: Output image filename.
        title: Plot title.
        y_label: Y-axis label.
        train_label: Legend label for train curve.
        test_label: Legend label for test curve.
    """
    if len(train_values) == 0 and len(test_values) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 4.5))

    if len(train_values) > 0:
        train_epochs = range(1, len(train_values) + 1)
        plt.plot(train_epochs, train_values, marker="o", label=train_label)

    if len(test_values) > 0:
        test_epochs = range(1, len(test_values) + 1)
        plt.plot(test_epochs, test_values, marker="o", label=test_label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Loss plot saved to {save_path}")
    plt.close()


def _to_display_image(tensor):
    """Convert image tensor to a matplotlib-compatible format."""
    img = tensor.squeeze()
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    return img


def _reshape_to_image_batch(batch, image_shape):
    """Ensure batch is in (N, C, H, W) format for visualization."""
    if batch.dim() == 2:
        channels, height, width = image_shape
        expected_size = channels * height * width
        if batch.shape[1] != expected_size:
            raise ValueError(
                f"Cannot reshape batch of shape {tuple(batch.shape)} "
                f"to image shape {(channels, height, width)}"
            )
        return batch.view(-1, channels, height, width)
    return batch


def _normalize_class_names(raw_classes, num_labels):
    """Return class names aligned with model label IDs [0, num_labels-1]."""
    class_names = [str(name) for name in raw_classes]

    # EMNIST letters provides classes like ["N/A", "a", ..., "z"] while labels
    # are remapped to [0, 25], so drop the placeholder to align indices.
    if (
        len(class_names) == num_labels + 1
        and class_names[0].strip().lower() in {"n/a", "na"}
    ):
        class_names = class_names[1:]

    if len(class_names) < num_labels:
        return None

    return class_names[:num_labels]


def _resolve_class_names(test_loader, num_labels, explicit_class_names=None):
    """Resolve class names for visualization titles; fallback to label IDs."""
    if explicit_class_names is not None:
        normalized = _normalize_class_names(explicit_class_names, num_labels)
        if normalized is not None:
            return normalized

    dataset = getattr(test_loader, "dataset", None)
    dataset_class_names = getattr(dataset, "class_names", None)
    if dataset_class_names is not None:
        normalized = _normalize_class_names(dataset_class_names, num_labels)
        if normalized is not None:
            return normalized

    dataset_classes = getattr(dataset, "classes", None)
    if dataset_classes is not None:
        normalized = _normalize_class_names(dataset_classes, num_labels)
        if normalized is not None:
            return normalized

    return [str(i) for i in range(num_labels)]


def _format_label_with_index(class_idx, class_name):
    """Format labels as '<index>: <name>' for display."""
    return f"{class_idx}: {class_name}"


def _project_latents_to_2d(latents: torch.Tensor) -> torch.Tensor:
    """Project latent vectors to 2D for visualization.

    Uses identity for 2D latents and PCA for higher-dimensional latents.
    """
    if latents.dim() != 2:
        raise ValueError(f"Expected latents with shape (N, D), got {tuple(latents.shape)}")

    n_samples, latent_dim = latents.shape
    if n_samples == 0:
        return torch.zeros((0, 2), dtype=latents.dtype)

    if latent_dim == 1:
        return torch.cat([latents, torch.zeros_like(latents)], dim=1)

    if latent_dim == 2:
        return latents

    centered = latents - latents.mean(dim=0, keepdim=True)
    q = min(2, centered.shape[0], centered.shape[1])
    if q <= 0:
        return torch.zeros((n_samples, 2), dtype=latents.dtype)

    _, _, right_vectors = torch.pca_lowrank(centered, q=q)
    projected = centered @ right_vectors[:, :q]

    if projected.shape[1] == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)

    return projected


def visualize_latent_space(
    autoencoder,
    test_loader,
    device,
    output_dir,
    num_labels,
    max_points=2000,
    class_names=None,
    label_transform=None,
):
    """Visualize encoded test-set latent vectors in 2D.

    The function projects latents with PCA when latent_dim > 2 and colors
    points by (possibly transformed) conditioning labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    autoencoder.eval()

    all_latents = []
    all_labels = []
    points_collected = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            if label_transform is not None:
                labels = label_transform(labels)

            z = autoencoder.encode(images).detach().cpu()
            labels = labels.detach().cpu()

            remaining = max_points - points_collected
            if remaining <= 0:
                break

            if z.shape[0] > remaining:
                z = z[:remaining]
                labels = labels[:remaining]

            all_latents.append(z)
            all_labels.append(labels)
            points_collected += z.shape[0]

            if points_collected >= max_points:
                break

    if points_collected == 0:
        return

    latents = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    projected = _project_latents_to_2d(latents)
    resolved_class_names = _resolve_class_names(
        test_loader=test_loader,
        num_labels=num_labels,
        explicit_class_names=class_names,
    )

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab10", num_labels)

    for class_idx in range(num_labels):
        class_mask = labels == class_idx
        if not class_mask.any():
            continue

        class_points = projected[class_mask]
        class_label = _format_label_with_index(class_idx, resolved_class_names[class_idx])
        plt.scatter(
            class_points[:, 0],
            class_points[:, 1],
            s=12,
            alpha=0.65,
            color=colors(class_idx),
            label=class_label,
        )

    plt.title("Latent Space (2D projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.25)
    if num_labels <= 20:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "latent_space.png")
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Latent space visualization saved to {save_path}")
    plt.close()


def visualize_cspn_latent_space(
    autoencoder,
    cspn,
    test_loader,
    device,
    output_dir,
    num_labels,
    max_points=2000,
    samples_per_label=200,
    class_names=None,
    label_transform=None,
):
    """Visualize encoded vs CSPN latent distributions in a shared 2D space."""
    os.makedirs(output_dir, exist_ok=True)
    autoencoder.eval()
    cspn.eval()

    encoded_latents = []
    encoded_labels = []
    points_collected = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            if label_transform is not None:
                labels = label_transform(labels)

            z = autoencoder.encode(images).detach().cpu()
            labels = labels.detach().cpu()

            remaining = max_points - points_collected
            if remaining <= 0:
                break

            if z.shape[0] > remaining:
                z = z[:remaining]
                labels = labels[:remaining]

            encoded_latents.append(z)
            encoded_labels.append(labels)
            points_collected += z.shape[0]

            if points_collected >= max_points:
                break

    if points_collected == 0:
        return

    encoded_latents = torch.cat(encoded_latents, dim=0)
    encoded_labels = torch.cat(encoded_labels, dim=0)

    with torch.no_grad():
        label_ids = torch.arange(num_labels, device=device, dtype=torch.long)
        centers = cspn.predict_latent(label_ids).detach().cpu()

    sampled_latents = []
    sampled_labels = []
    with torch.no_grad():
        for class_idx in range(num_labels):
            class_label = torch.full(
                (samples_per_label,),
                fill_value=class_idx,
                dtype=torch.long,
                device=device,
            )
            class_samples = cspn.sample(class_label, num_samples=samples_per_label)
            sampled_latents.append(class_samples.detach().cpu())
            sampled_labels.append(torch.full((samples_per_label,), class_idx, dtype=torch.long))

    sampled_latents = torch.cat(sampled_latents, dim=0)
    sampled_labels = torch.cat(sampled_labels, dim=0)

    merged = torch.cat([encoded_latents, sampled_latents, centers], dim=0)
    merged_projected = _project_latents_to_2d(merged)

    n_encoded = encoded_latents.shape[0]
    n_sampled = sampled_latents.shape[0]
    encoded_projected = merged_projected[:n_encoded]
    sampled_projected = merged_projected[n_encoded : n_encoded + n_sampled]
    centers_projected = merged_projected[n_encoded + n_sampled :]

    resolved_class_names = _resolve_class_names(
        test_loader=test_loader,
        num_labels=num_labels,
        explicit_class_names=class_names,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    colors = plt.cm.get_cmap("tab20", max(1, num_labels))

    for class_idx in range(num_labels):
        class_label = _format_label_with_index(class_idx, resolved_class_names[class_idx])

        encoded_mask = encoded_labels == class_idx
        if encoded_mask.any():
            points = encoded_projected[encoded_mask]
            axes[0].scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                alpha=0.55,
                color=colors(class_idx),
                label=class_label,
            )

        sampled_mask = sampled_labels == class_idx
        if sampled_mask.any():
            points = sampled_projected[sampled_mask]
            axes[1].scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                alpha=0.45,
                color=colors(class_idx),
                label=class_label,
            )

        center = centers_projected[class_idx]
        axes[1].scatter(
            center[0],
            center[1],
            marker="*",
            s=180,
            color=colors(class_idx),
            edgecolor="black",
            linewidth=0.8,
        )

    axes[0].set_title("Encoded Latents")
    axes[1].set_title("CSPN Samples + Centers")
    for axis in axes:
        axis.set_xlabel("Component 1")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Component 2")

    if num_labels <= 20:
        axes[0].legend(loc="best", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cspn_latent_space.png")
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"CSPN latent space visualization saved to {save_path}")
    plt.close()


def visualize_cspn(
    autoencoder,
    cspn,
    test_loader,
    device,
    output_dir,
    num_labels,
    num_samples=8,
    class_names=None,
    label_transform=None,
):
    """Visualize CSPN behavior in latent space.

    Produces two figures:
    1) Label-conditioned latent prototypes decoded to image space.
    2) Latent label-transfer on real samples (source label -> target label).
    """
    os.makedirs(output_dir, exist_ok=True)

    autoencoder.eval()
    cspn.eval()

    with torch.no_grad():
        preview_images, preview_labels = next(iter(test_loader))
        image_shape = tuple(preview_images.shape[1:])
        resolved_class_names = _resolve_class_names(
            test_loader=test_loader,
            num_labels=num_labels,
            explicit_class_names=class_names,
        )

        label_ids = torch.arange(num_labels, device=device, dtype=torch.long)
        proto_latents = cspn.predict_latent(label_ids)
        proto_images = autoencoder.decode(proto_latents)
        proto_images = _reshape_to_image_batch(proto_images, image_shape)

        proto_images = proto_images.detach().cpu()
        grid_cols = math.ceil(math.sqrt(num_labels))
        grid_rows = math.ceil(num_labels / grid_cols)
        fig, _ = plt.subplots(
            grid_rows,
            grid_cols,
            figsize=(2 * grid_cols, 2.4 * grid_rows),
        )
        proto_axes = fig.axes

        for class_idx, axis in enumerate(proto_axes):
            if class_idx >= num_labels:
                axis.axis("off")
                continue

            img = _to_display_image(proto_images[class_idx])
            axis.imshow(img, cmap="gray" if img.ndim == 2 else None)
            axis.set_title(
                _format_label_with_index(class_idx, resolved_class_names[class_idx])
            )
            axis.axis("off")

        plt.tight_layout()
        prototype_path = os.path.join(output_dir, "cspn_prototypes.png")
        plt.savefig(prototype_path, dpi=120, bbox_inches="tight")
        print(f"CSPN prototype visualization saved to {prototype_path}")
        plt.close()

        # transferred images
        images, labels = preview_images, preview_labels
        images = images[:num_samples].to(device)
        raw_labels = labels[:num_samples].to(device).long()
        labels = raw_labels
        if label_transform is not None:
            labels = label_transform(labels)
        target_labels = (labels + 1) % num_labels

        z = autoencoder.encode(images)
        transformed_z = cspn.transform_latent(
            z,
            source_labels=labels,
            target_labels=target_labels,
            strength=1.0,
        )

        reconstructed = autoencoder(images)
        transferred = autoencoder.decode(transformed_z)
        reconstructed = _reshape_to_image_batch(reconstructed, image_shape)
        transferred = _reshape_to_image_batch(transferred, image_shape)

        images_cpu = images.detach().cpu()
        reconstructed_cpu = reconstructed.detach().cpu()
        transferred_cpu = transferred.detach().cpu()
        raw_labels_cpu = raw_labels.detach().cpu()
        labels_cpu = labels.detach().cpu()
        target_labels_cpu = target_labels.detach().cpu()

        fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
        for i in range(num_samples):
            src = _to_display_image(images_cpu[i])
            recon = _to_display_image(reconstructed_cpu[i])
            trans = _to_display_image(transferred_cpu[i])

            axes[0, i].imshow(src, cmap="gray" if src.ndim == 2 else None)
            src_idx = int(labels_cpu[i])
            src_label = _format_label_with_index(src_idx, resolved_class_names[src_idx])
            raw_src_idx = int(raw_labels_cpu[i])
            if raw_src_idx != src_idx:
                axes[0, i].set_title(f"orig {raw_src_idx} ({src_label})")
            else:
                axes[0, i].set_title(f"orig {src_label}")
            axes[0, i].axis("off")

            axes[1, i].imshow(recon, cmap="gray" if recon.ndim == 2 else None)
            axes[1, i].set_title("recon")
            axes[1, i].axis("off")

            axes[2, i].imshow(trans, cmap="gray" if trans.ndim == 2 else None)
            target_idx = int(target_labels_cpu[i])
            target_label = _format_label_with_index(
                target_idx,
                resolved_class_names[target_idx],
            )
            axes[2, i].set_title(f"to {target_label}")
            axes[2, i].axis("off")

        plt.tight_layout()
        transfer_path = os.path.join(output_dir, "cspn_transfer.png")
        plt.savefig(transfer_path, dpi=120, bbox_inches="tight")
        print(f"CSPN transfer visualization saved to {transfer_path}")
        plt.close()
