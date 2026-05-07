"""Visualization helpers for training diagnostics."""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_reconstructions(originals, reconstructions, labels, path):
    assert originals.shape == reconstructions.shape
    num_images = len(originals)
    logger.info(f"Generating reconstruction visualization for {num_images} images")

    originals = originals.clamp(min=0, max=1)
    reconstructions = reconstructions.clamp(min=0, max=1)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # label
        axes[0, i].set_title(f"Label: {labels[i].item()}", fontsize=10)

        # original
        axes[0, i].imshow(originals[i].permute(1, 2, 0).squeeze())
        axes[0, i].axis("off")

        # reconstructed
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).squeeze())
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    logger.info(f"Saved reconstructions to {path}")
