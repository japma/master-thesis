"""The digit classifier the generation metrics judge samples with."""

import torch
import torch.nn as nn

from utils.config.classifier import ClassifierConfig


class DigitClassifier(nn.Module):
    """Small CNN over colour-MNIST images; predicts the digit factor only.    """

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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(third * pooled * pooled, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self(images).argmax(dim=1)

    def get_config(self) -> dict:
        return self.config.model_dump(mode="json")
