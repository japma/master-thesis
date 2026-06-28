"""Loss utilities."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, recon, target):
        return 0.5 * self.mse(recon, target) + 0.5 * self.l1(recon, target)


def beta_vae_loss(
    images: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_fn: nn.Module = nn.MSELoss(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = recon_loss_fn(recon, images)

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class BetaVAELoss(nn.Module):
    def __init__(self, beta: float = 1.0, free_bits: float = 0.5):
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits

    def forward(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logvar = logvar.clamp(-30.0, 20.0)
        B = images.size(0)
        recon_loss = (
            F.binary_cross_entropy_with_logits(recon, images, reduction="sum") / B
        )

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.clamp(min=self.free_bits).mean()

        effective_beta = beta if beta is not None else self.beta
        return recon_loss + effective_beta * kl_loss, recon_loss, kl_loss


class VGGPerceptualLoss(nn.Module):
    """Feature-space L1 loss using a frozen VGG16."""

    _FEATURE_LAYERS: tuple[int, ...] = (9, 16)  # relu2_2, relu3_3 in VGG16 features

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # Keep only up to the deepest layer we need
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
        recon_feats = self._features(recon)
        with torch.no_grad():
            target_feats = self._features(target)
        return sum(F.l1_loss(r, t) for r, t in zip(recon_feats, target_feats)) / len(
            recon_feats
        )  # type: ignore[return-value]


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(
        self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 3
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_channels
        for _i in range(1, num_layers):
            ch_next = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, ch_next, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next

        ch_next = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, ch_next, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_next),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_next, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VAELossOutput(dict):
    total_g: torch.Tensor
    recon: torch.Tensor  # BCE reconstruction loss
    kl: torch.Tensor  # KL divergence
    perceptual: torch.Tensor  # VGG perceptual loss
    adversarial_g: torch.Tensor  # generator adversarial loss
    discriminator: torch.Tensor  # discriminator loss


class VAELoss(nn.Module):
    def __init__(
        self,
        beta_vae_loss: BetaVAELoss,
        discriminator: PatchDiscriminator,
        lambda_perceptual: float = 1.0,
        lambda_adversarial: float = 0.1,
        adversarial_warmup_steps: int = 1000,
        adaptive_weight: bool = True,
    ) -> None:
        super().__init__()
        self.beta_vae = beta_vae_loss
        self.discriminator = discriminator
        self.perceptual = VGGPerceptualLoss()
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial
        self.adversarial_warmup_steps = adversarial_warmup_steps
        self.adaptive_weight = adaptive_weight

    def _adaptive_adversarial_weight(
        self,
        recon_loss: torch.Tensor,
        adv_loss: torch.Tensor,
        last_layer: nn.Module,
    ) -> torch.Tensor:
        recon_grads = torch.autograd.grad(
            recon_loss, last_layer.weight, retain_graph=True
        )[0]
        adv_grads = torch.autograd.grad(adv_loss, last_layer.weight, retain_graph=True)[
            0
        ]

        ratio = torch.norm(recon_grads) / (torch.norm(adv_grads) + 1e-8)
        return ratio.clamp(0.0, 1e4).detach() * self.lambda_adversarial

    def generator_loss(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        last_decoder_layer: nn.Module,
        step: int,
        beta: float | None = None,
    ) -> VAELossOutput:
        # recon + KL
        total_beta_vae, recon_loss, kl_loss = self.beta_vae(
            images, logits, mu, logvar, beta
        )

        recon_img = torch.sigmoid(logits)
        p_loss = self.perceptual(recon_img, images)

        adv_loss: torch.Tensor
        adv_weight: float | torch.Tensor
        if step < self.adversarial_warmup_steps:
            adv_loss = torch.tensor(0.0, device=images.device)
            adv_weight = 0.0
        else:
            fake_logits = self.discriminator(recon_img)
            adv_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )
            if self.adaptive_weight:
                adv_weight = self._adaptive_adversarial_weight(
                    recon_loss, adv_loss, last_decoder_layer
                )
            else:
                adv_weight = self.lambda_adversarial

        total_g = (
            total_beta_vae + self.lambda_perceptual * p_loss + adv_weight * adv_loss
        )

        out = VAELossOutput(
            total_g=total_g,
            recon=recon_loss,
            kl=kl_loss,
            perceptual=p_loss,
            adversarial_g=adv_loss,
            discriminator=torch.tensor(float("nan")),
        )
        return out

    def discriminator_loss(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:
        real_logits = self.discriminator(images)
        fake_logits = self.discriminator(recon)

        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        return 0.5 * (real_loss + fake_loss)


def negative_log_likelihood_loss(
    outputs: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for SPN outputs."""
    return -outputs.mean()


def get_ae_loss_fn(loss_type: str) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "hybrid":
        return HybridLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
