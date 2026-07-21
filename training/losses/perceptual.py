import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """Feature-space L1 loss using a frozen VGG16."""

    _FEATURE_LAYERS: tuple[int, int] = (9, 16)  # relu2_2, relu3_3 in VGG16 features

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg.children())[: self._FEATURE_LAYERS[-1] + 1])
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std  # type: ignore[operator]

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._normalise(x)
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self._FEATURE_LAYERS:
                feats.append(x)
        return feats

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Both recon and target must be in [0, 1]."""
        recon_feats = self._features(recon)
        with torch.no_grad():
            target_feats = self._features(target)
        # pyrefly: ignore [bad-return]
        return sum(
            F.l1_loss(r, t) for r, t in zip(recon_feats, target_feats, strict=False)
        ) / len(recon_feats)
