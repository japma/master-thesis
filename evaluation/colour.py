"""How a colour is read off an image.

Colour-MNIST's palettes are fixed, so this is a readout rather than a judgement: no
model, no training, and it applies identically to generated, real and reconstructed
images.
"""

import numpy as np
import torch

from dataset_loaders.colour_mnist import BG_COLOURS, FG_COLOURS

# The palettes as float RGB in [0, 1], indexed the way labels are.
FG_PALETTE = np.array(list(FG_COLOURS.values()), dtype=np.float32) / 255.0
BG_PALETTE = np.array(list(BG_COLOURS.values()), dtype=np.float32) / 255.0

# MNIST digits never reach the image border, so the outer ring is pure background.
BORDER_MARGIN = 2
# Fraction of pixels, ranked by distance from the background, treated as foreground.
FG_QUANTILE = 0.9


def border_colour(images: torch.Tensor, margin: int = BORDER_MARGIN) -> torch.Tensor:
    """Mean colour of the outer ring of each `(N, C, H, W)` image."""
    top = images[:, :, :margin, :].mean(dim=(2, 3))
    bottom = images[:, :, -margin:, :].mean(dim=(2, 3))
    left = images[:, :, :, :margin].mean(dim=(2, 3))
    right = images[:, :, :, -margin:].mean(dim=(2, 3))
    return (top + bottom + left + right) / 4.0


def foreground_colour(
    images: torch.Tensor, quantile: float = FG_QUANTILE
) -> torch.Tensor:
    """Mean colour of the pixels furthest from each image's own border colour.

    Self-locating, so generated images (which have no paired original) are read the same
    way as real ones. A flat image with no digit still yields a colour here; it is caught
    by digit accuracy and by contrast, not by this.
    """
    background = border_colour(images).unsqueeze(-1).unsqueeze(-1)
    distance = (images - background).pow(2).sum(dim=1).flatten(1)
    k = max(1, round(distance.shape[1] * (1.0 - quantile)))
    idx = distance.topk(k, dim=1).indices

    flat = images.flatten(2)
    gathered = flat.gather(2, idx.unsqueeze(1).expand(-1, flat.shape[1], -1))
    return gathered.mean(dim=2)


def nearest_palette_index(colours: torch.Tensor, palette: np.ndarray) -> torch.Tensor:
    """Index of the palette entry closest to each `(N, 3)` colour, in RGB space."""
    reference = torch.as_tensor(palette, dtype=colours.dtype, device=colours.device)
    distance = (colours.unsqueeze(1) - reference.unsqueeze(0)).pow(2).sum(dim=2)
    return distance.argmin(dim=1)


def palette_colour(palette: np.ndarray, targets: torch.Tensor) -> torch.Tensor:
    """The colour each target index stands for, as a `(N, 3)` tensor."""
    return torch.as_tensor(palette, dtype=torch.float32)[targets]
