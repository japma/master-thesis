"""RGB distance between foreground and background: near zero for a flat image.

Separates a wrongly-coloured digit from no digit at all, which both read as a colour
miss. Not a collapse detector -- a mode-proximate sample is a *cleaner*, higher-contrast
digit, so contrast moves the wrong way for that. Pair it with a spread metric.
"""

import pandas as pd
import torch

from evaluation.colour import border_colour, foreground_colour

FILENAME = "colour_contrast.csv"


def compute(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> pd.DataFrame:
    separation = (foreground_colour(images) - border_colour(images)).norm(dim=1)
    return pd.DataFrame(
        [{"value": float(separation.mean()), "n": int(separation.shape[0])}]
    )
