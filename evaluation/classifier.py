"""Loading the fixed digit classifier that judges generated images.

Deliberately trained on the `uniform` variant, where all 180 combinations appear
equally often: a judge trained on a skewed or held-out variant would itself be worse at
exactly the combinations a model under test was never shown, and the two failures would
be impossible to tell apart.

Train one with `uv run train_classifier configs/classifier/colour_mnist_uniform.yaml`.
"""

import torch

from models.classifier import DigitClassifier
from utils.checkpoints import load_classifier_from_path
from utils.wandb_utils import download_artifact

JUDGE_ARTIFACT = "digit_classifier_colour_mnist_uniform"


def load_digit_classifier(
    artifact: str = JUDGE_ARTIFACT,
    tag: str = "latest",
    device: torch.device | None = None,
) -> DigitClassifier:
    """The judge behind a wandb `name[:version]`.

    Goes through `download_artifact`, so a run that scores samples with this judge
    records which version of it produced the numbers.
    """
    path, resolved = download_artifact(artifact, tag)
    print(f"Judging digits with {resolved}")
    return load_classifier_from_path(path, device=device).eval()
