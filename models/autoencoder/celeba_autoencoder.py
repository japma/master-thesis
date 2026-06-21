from abc import ABC

import torch
from diffusers import AutoencoderTiny

from models.autoencoder import AbstractAutoencoder

import torch.nn as nn
import torch.nn.functional as F
# from torch.amp import autocast
# import lpips

# --- Taken from https://github.com/AyushShahh/Beta-VAE-CelebA/ ---


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.gn1 = nn.GroupNorm(32, out_channels)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.gn2 = nn.GroupNorm(32, out_channels)
        self.lrelu2 = nn.LeakyReLU(0.2)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + residual
        out = self.lrelu2(out)
        return out


class CelebAVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=2),
            # ResBlock(256, 512, stride=2),
            # ResBlock(512, 512, stride=2),
        )

        # self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        # self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder
        # self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # ResBlock(512, 256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(64, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            # nn.Tanh()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # if isinstance(m, nn.Conv2d) and m.out_channels == 3 and m.kernel_size == (3, 3):
            #     nn.init.normal_(m.weight, mean=0.0, std=0.01) # Very small weights for Tanh
            # else:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc(z)
        # h = h.view(-1, 512, 4, 4)
        h = h.reshape(h.size(0), -1, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class ELBOLoss(nn.Module):
    def __init__(self, gamma=1000.0, C_max=35.0, device="cuda"):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.C_max = C_max
        # self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        # self.l1_loss = nn.L1Loss()
        # self.mse_loss = nn.MSELoss()

    def forward(self, x, x_hat, mu, logvar, step=None, total_steps=None):
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_batch = torch.sum(kl_per_dim, dim=1)
        kl = torch.mean(kl_batch)

        # Perceptual Loss
        # p_loss = 0.0
        # if lpips:
        #     with autocast(device_type=self.device, enabled=False):
        #         p_loss = self.lpips_loss(x_hat.float(), x.float()).mean()

        # L1 Loss
        # l1 = self.l1_loss(x_hat, x)
        # mse = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        bce = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum") / x.size(0)

        recon = bce  # + lpips * p_loss

        if self.C_max is None:
            C = 0
            loss = recon + self.gamma * kl
        else:
            C = self.C_max * step / total_steps
            C = min(C, self.C_max)
            loss = recon + self.gamma * torch.abs(kl - C)

        return loss, recon, kl, torch.mean(kl_per_dim, dim=0), C, mu, logvar


class CelebAAutoencoderWrapper(AbstractAutoencoder, ABC):
    def __init__(self):
        super().__init__()
        self.vae = CelebAVAE(latent_dim=32)
        self.name = "celeba_autoencoder"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(x)
        return latent[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.vae.decode(z)
        return decoded

    def get_config(self) -> dict:
        return {
            "model_type": "tiny",
            "name": self.name,
        }

    def get_latent_dim(self) -> int:
        # TODO place holder until it is resolved if this is needed
        return 32
