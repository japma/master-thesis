"""Visualization helpers for training diagnostics."""

import os

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


def visualize_cspn(
    autoencoder,
    cspn,
    test_loader,
    device,
    output_dir,
    num_labels,
    num_samples=8,
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

        label_ids = torch.arange(num_labels, device=device, dtype=torch.long)
        proto_latents = cspn.predict_latent(label_ids)
        proto_images = autoencoder.decode(proto_latents)
        proto_images = _reshape_to_image_batch(proto_images, image_shape)

        proto_images = proto_images.detach().cpu()
        fig, _ = plt.subplots(1, num_labels, figsize=(2 * num_labels, 2.5))
        proto_axes = fig.axes

        for class_idx, axis in enumerate(proto_axes):
            img = _to_display_image(proto_images[class_idx])
            axis.imshow(img, cmap="gray" if img.ndim == 2 else None)
            axis.set_title(f"y={class_idx}")
            axis.axis("off")

        plt.tight_layout()
        prototype_path = os.path.join(output_dir, "cspn_prototypes.png")
        plt.savefig(prototype_path, dpi=120, bbox_inches="tight")
        print(f"CSPN prototype visualization saved to {prototype_path}")
        plt.close()

        # transfered images
        images, labels = preview_images, preview_labels
        images = images[:num_samples].to(device)
        labels = labels[:num_samples].to(device).long()
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
        labels_cpu = labels.detach().cpu()
        target_labels_cpu = target_labels.detach().cpu()

        fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
        for i in range(num_samples):
            src = _to_display_image(images_cpu[i])
            recon = _to_display_image(reconstructed_cpu[i])
            trans = _to_display_image(transferred_cpu[i])

            axes[0, i].imshow(src, cmap="gray" if src.ndim == 2 else None)
            axes[0, i].set_title(f"orig y={int(labels_cpu[i])}")
            axes[0, i].axis("off")

            axes[1, i].imshow(recon, cmap="gray" if recon.ndim == 2 else None)
            axes[1, i].set_title("recon")
            axes[1, i].axis("off")

            axes[2, i].imshow(trans, cmap="gray" if trans.ndim == 2 else None)
            axes[2, i].set_title(f"to y={int(target_labels_cpu[i])}")
            axes[2, i].axis("off")

        plt.tight_layout()
        transfer_path = os.path.join(output_dir, "cspn_transfer.png")
        plt.savefig(transfer_path, dpi=120, bbox_inches="tight")
        print(f"CSPN transfer visualization saved to {transfer_path}")
        plt.close()
