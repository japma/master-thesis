"""Utility functions for training and model management."""

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class TinyImageNetDataset(Dataset):
    """Local Tiny ImageNet dataset.

    Expected structure is the standard tiny-imagenet-200 folder layout.
    """

    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(
                f"Missing wnids.txt at {wnids_path}. "
                "Expected Tiny ImageNet root directory."
            )

        with wnids_path.open("r", encoding="utf-8") as file:
            self.classes = [line.strip() for line in file if line.strip()]

        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []

        if split == "train":
            self._load_train_samples()
        elif split == "val":
            self._load_val_samples()
        else:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for Tiny ImageNet split '{split}' in {self.root}."
            )

    def _load_train_samples(self):
        train_dir = self.root / "train"
        for class_name in self.classes:
            image_dir = train_dir / class_name / "images"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.JPEG")):
                self.samples.append((image_path, self.class_to_idx[class_name]))

    def _load_val_samples(self):
        val_dir = self.root / "val"
        annotations_path = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"

        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Missing val_annotations.txt at {annotations_path}."
            )

        with annotations_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in self.class_to_idx:
                    continue
                image_path = images_dir / image_name
                if image_path.exists():
                    self.samples.append((image_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_data_loaders(dataset_name, batch_size=32, dataset_kwargs=None):
    """Load dataset with data loaders.

    Args:
        dataset_name: Name of the dataset to load. Must match a class in torchvision.datasets.
        batch_size: Batch size for training and validation.
        dataset_kwargs: Optional kwargs passed to torchvision dataset constructors
            (e.g., {"split": "letters"} for EMNIST).

    Returns:
        Tuple of (train_loader, test_loader).
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_kwargs = dict(dataset_kwargs or {})

    if dataset_name == "TinyImageNet":
        tiny_root = dataset_kwargs.pop("root", "./data/tiny-imagenet-200")
        tiny_root = dataset_kwargs.pop("data_dir", tiny_root)
        image_size = dataset_kwargs.pop("image_size", 64)

        if len(dataset_kwargs) > 0:
            unknown_keys = ", ".join(sorted(dataset_kwargs.keys()))
            raise ValueError(
                f"Unsupported dataset_kwargs for TinyImageNet: {unknown_keys}"
            )

        tiny_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = TinyImageNetDataset(
            root=tiny_root,
            split="train",
            transform=tiny_transform,
        )
        test_dataset = TinyImageNetDataset(
            root=tiny_root,
            split="val",
            transform=tiny_transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    # Remap to [0, 25] for EMNIST
    if dataset_name == "EMNIST" and dataset_kwargs.get("split") == "letters":
        existing_target_transform = dataset_kwargs.get("target_transform")

        def _emnist_letters_to_zero_based(label):
            label = label - 1
            if existing_target_transform is not None:
                return existing_target_transform(label)
            return label

        dataset_kwargs["target_transform"] = _emnist_letters_to_zero_based

    dataset_class = getattr(datasets, dataset_name)
    train_dataset = dataset_class(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        **dataset_kwargs,
    )
    test_dataset = dataset_class(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        **dataset_kwargs,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def format_elapsed_time(seconds):
    """Format elapsed time in hours, minutes, and seconds for time measurements.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


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
        proto_images = autoencoder.decoder(proto_latents)
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
        transferred = autoencoder.decoder(transformed_z)
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
