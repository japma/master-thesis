"""A fixed digit classifier, used only to judge generated images.

Deliberately trained on the `uniform` variant, where all 180 combinations appear
equally often: a judge trained on a skewed or held-out variant would itself be worse at
exactly the combinations a model under test was never shown, and the two failures would
be impossible to tell apart.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loaders.colour_mnist import NUM_DIGITS

CLASSIFIER_PATH = Path("checkpoints") / "digit_classifier_colour_mnist.pt"


class DigitClassifier(nn.Module):
    """Small CNN over 3x28x28 colour-MNIST images."""

    def __init__(self, num_classes: int = NUM_DIGITS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self(images).argmax(dim=1)


def save_digit_classifier(model: DigitClassifier, path: Path = CLASSIFIER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)
    print("Saved digit classifier to", path)


def load_digit_classifier(
    path: Path = CLASSIFIER_PATH, device: torch.device | None = None
) -> DigitClassifier:
    if not path.exists():
        raise FileNotFoundError(
            f"No digit classifier at {path}. Train one with "
            "`uv run train_digit_classifier`."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DigitClassifier()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model.to(device) if device is not None else model


def train_digit_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 1e-3,
) -> tuple[DigitClassifier, float]:
    """Returns the trained classifier and its validation accuracy."""
    model = DigitClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            digits = targets[:, 0].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss_fn(model(images), digits).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                digits = targets[:, 0].to(device, non_blocking=True)
                correct += int((model.predict(images) == digits).sum())
                total += digits.numel()
        accuracy = correct / max(total, 1)
        print(f"epoch {epoch + 1}/{epochs}  val digit accuracy {accuracy:.4f}")

    return model, accuracy
