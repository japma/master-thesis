"""How a colour is read off an image."""

import numpy as np
import torch

from dataset_loaders.colour_mnist import (
    BG_COLOURS,
    FG_COLOURS,
    NUM_BG,
    NUM_DIGITS,
    NUM_FG,
    TABLE_SHAPE,
)

# Palettes as float RGB in [0, 1], indexed the same way labels are.
FG_PALETTE = np.array(list(FG_COLOURS.values()), dtype=np.float64) / 255.0
BG_PALETTE = np.array(list(BG_COLOURS.values()), dtype=np.float64) / 255.0

# MNIST digits never reach the image border, so the outer ring is pure background.
BORDER_MARGIN = 2
# Fraction of pixels, ranked by distance from the background colour, treated as foreground.
FG_QUANTILE = 0.9


def border_colour(images: torch.Tensor, margin: int = BORDER_MARGIN) -> torch.Tensor:
    """Mean colour of the outer ring"""
    top = images[:, :, :margin, :].mean(dim=(2, 3))
    bottom = images[:, :, -margin:, :].mean(dim=(2, 3))
    left = images[:, :, :, :margin].mean(dim=(2, 3))
    right = images[:, :, :, -margin:].mean(dim=(2, 3))
    return (top + bottom + left + right) / 4.0


def foreground_colour(
    originals: torch.Tensor, recons: torch.Tensor, quantile: float = FG_QUANTILE
) -> torch.Tensor:
    """Colour the reconstruction puts where the *original* is most clearly foreground.

    Picking the pixels from the original rather than the reconstruction is deliberate: a
    reconstruction that dropped the digit entirely should score badly, and it would score
    well if we let it choose its own foreground pixels.
    """
    batch = originals.shape[0]
    background = border_colour(originals).unsqueeze(-1).unsqueeze(-1)
    distance = (originals - background).pow(2).sum(dim=1).flatten(1)  # (B, H*W)

    num_pixels = distance.shape[1]
    k = max(1, round(num_pixels * (1.0 - quantile)))
    idx = distance.topk(k, dim=1).indices  # (B, k)

    flat_recon = recons.flatten(2)  # (B, C, H*W)
    gathered = flat_recon.gather(
        2, idx.unsqueeze(1).expand(-1, flat_recon.shape[1], -1)
    )
    return gathered.mean(dim=2) if batch else gathered


def nearest_palette_index(colours: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """Index of the palette entry closest to each colour, in RGB space."""
    reference = torch.tensor(palette, dtype=colours.dtype, device=colours.device)
    distance = (colours.unsqueeze(1) - reference.unsqueeze(0)).pow(2).sum(-1)
    return distance.argmin(dim=1).cpu().numpy()


assert FG_PALETTE.shape == (NUM_FG, 3)
assert BG_PALETTE.shape == (NUM_BG, 3)
assert TABLE_SHAPE == (NUM_DIGITS, NUM_FG, NUM_BG)
