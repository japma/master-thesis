"""The classifier the generation metrics judge samples with."""

import torch
import torch.nn as nn

from utils.config.classifier import ClassifierConfig


class DigitClassifier(nn.Module):
    """Small CNN over colour-MNIST images; one head per label factor on a shared trunk."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassifierConfig()
        first, second, third = self.config.conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(self.config.channels, first, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(first, second, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(second, third, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        pooled = self.config.image_size // 4
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(third * pooled * pooled, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.heads = nn.ModuleList(
            nn.Linear(self.config.hidden_dim, cardinality)
            for cardinality in self.config.cardinalities
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Logits per label factor, in label-column order."""
        hidden = self.trunk(self.features(images))
        return [head(hidden) for head in self.heads]

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """`(N, num_factors)` predicted classes, laid out like the labels."""
        return torch.stack([logits.argmax(dim=1) for logits in self(images)], dim=1)

    def get_config(self) -> dict:
        return self.config.model_dump(mode="json")
