from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import AutoencoderConfig

from .abstract_autoencoder import AbstractAutoencoder, AutoencoderForwardOutput

LOG_VAR_MIN: float = -10.0
LOG_VAR_MAX: float = 10.0


@dataclass
class VAEForwardOutput(AutoencoderForwardOutput):
    mu: torch.Tensor
    log_var: torch.Tensor


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def make_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = max(g for g in range(1, max_groups + 1) if channels % g == 0)
    return nn.GroupNorm(num_groups, channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            make_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttention2d(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: int,
        num_heads: int = 1,
        init_gate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm = make_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.full((channels,), init_gate))
        self.pos_embed = nn.Parameter(
            torch.randn(1, spatial_size * spatial_size, channels) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        h = self.norm(x)
        h = h.view(batch_size, channels, height * width).permute(0, 2, 1)
        h = h + self.pos_embed
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).view(batch_size, channels, height, width)
        return x + self.gate.view(1, channels, 1, 1) * h


class BlurPool2d(nn.Module):
    """Fixed binomial low-pass filter + stride-2 subsample (anti-aliased downsampling)."""

    kernel: torch.Tensor

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.expand(channels, 1, 3, 3).contiguous()
        self.register_buffer("kernel", kernel)
        self.channels: int = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=2, padding=1, groups=self.channels)


class EncoderBlock(nn.Module):
    """Conv + norm + activation, then anti-aliased downsample, then `num_resblocks` residual blocks."""

    def __init__(
        self, in_channels: int, out_channels: int, num_resblocks: int = 1
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            make_norm(out_channels),
            nn.ReLU(inplace=True),
            BlurPool2d(out_channels),
        )
        self.res = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(num_resblocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class DecoderBlock(nn.Module):
    """Upsample followed by `num_resblocks` residual blocks."""

    def __init__(
        self, in_channels: int, out_channels: int, num_resblocks: int = 1
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(num_resblocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


class VariationalAutoencoder(AbstractAutoencoder):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        num_encoder_resblocks: int = config.num_encoder_resblocks
        num_decoder_resblocks: int = config.num_decoder_resblocks

        self.input_proj = nn.Sequential(
            nn.Conv2d(
                config.channels, config.base_channels, kernel_size=3, padding=1, bias=False
            ),
            make_norm(config.base_channels),
            nn.ReLU(inplace=True),
        )

        # --- Encoder ---
        encoder_blocks: list[nn.Module] = []
        current_channels: int = config.base_channels
        for _ in range(config.num_blocks):
            out_channels = current_channels * 2
            encoder_blocks.append(
                EncoderBlock(
                    current_channels,
                    out_channels,
                    num_resblocks=num_encoder_resblocks,
                )
            )
            current_channels = out_channels

        self.encoder = nn.ModuleList(encoder_blocks)

        self.encoded_size: int = config.image_size // (2**config.num_blocks)

        self.bottleneck_attn = SelfAttention2d(current_channels, self.encoded_size)

        flat_dim: int = current_channels * self.encoded_size * self.encoded_size
        self.bottleneck_channels: int = current_channels

        self.mu_head = nn.Linear(flat_dim, config.latent_dim)
        self.log_var_head = nn.Linear(flat_dim, config.latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(config.latent_dim, flat_dim)

        decoder_blocks: list[nn.Module] = []
        for _ in range(config.num_blocks - 1):
            out_channels = current_channels // 2
            decoder_blocks.append(
                DecoderBlock(
                    current_channels,
                    out_channels,
                    num_resblocks=num_decoder_resblocks,
                )
            )
            current_channels = out_channels

        decoder_blocks.append(
            DecoderBlock(
                current_channels,
                config.base_channels,
                num_resblocks=num_decoder_resblocks,
            )
        )
        self.decoder = nn.ModuleList(decoder_blocks)

        self.output_proj = nn.Conv2d(
            config.base_channels, config.channels, kernel_size=3, padding=1
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.log_var_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.log_var_head.bias)

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.encoder:
            h = block(h)

        h = self.bottleneck_attn(h)
        flat = h.flatten(start_dim=1)
        mu = self.mu_head(flat)
        log_var = torch.clamp(self.log_var_head(flat), min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        return mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(x)
        return mu

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(
            z.shape[0], self.bottleneck_channels, self.encoded_size, self.encoded_size
        )
        for block in self.decoder:
            h = block(h)
        return self.output_proj(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x: torch.Tensor) -> AutoencoderForwardOutput:
        mu, log_var = self.encode_distribution(x)
        z = reparameterize(mu, log_var)
        logits = self.decode_logits(z)
        return VAEForwardOutput(
            reconstructed=logits,
            latent=z,
            mu=mu,
            log_var=log_var,
        )

    def get_config(self) -> dict[str, object]:
        return self.config.model_dump()

    def get_latent_dim(self) -> torch.Size:
        return torch.Size([self.config.latent_dim])
