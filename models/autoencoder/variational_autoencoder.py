import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import AutoencoderConfig

from .abstract_autoencoder import AbstractAutoencoder, AutoencoderForwardOutput


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    log_var = log_var.clamp(-30.0, 20.0)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)  # LayerNorm equivalent for 2-D
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, skip_channels: int = 0
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        fused_channels = out_channels + skip_channels
        self.fuse = (
            nn.Sequential(
                nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            if skip_channels > 0
            else nn.Identity()
        )
        self.res = ResBlock(out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = self.fuse(torch.cat([x, skip], dim=1))
        return self.res(x)


class VariationalAutoencoder(AbstractAutoencoder):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        self.use_skip_connections = config.use_skip_connections

        self.input_proj = nn.Sequential(
            nn.Conv2d(3, config.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        # --- Encoder ---
        encoder_blocks = []
        self._encoder_out_channels: list[int] = []
        current_channels = config.base_channels
        for _ in range(config.num_blocks):
            out_channels = current_channels * 2
            encoder_blocks.append(EncoderBlock(current_channels, out_channels))
            self._encoder_out_channels.append(out_channels)
            current_channels = out_channels

        self.encoder = nn.ModuleList(encoder_blocks)

        self.bottleneck_attn = SelfAttention2d(current_channels)

        self.encoded_size = config.image_size // (2**config.num_blocks)
        flat_dim = current_channels * self.encoded_size * self.encoded_size
        self.bottleneck_channels = current_channels

        self.mu_head = nn.Linear(flat_dim, config.latent_dim)
        self.log_var_head = nn.Linear(flat_dim, config.latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(config.latent_dim, flat_dim)

        decoder_blocks = []
        skip_channels_list = list(reversed(self._encoder_out_channels))
        for i in range(config.num_blocks - 1):
            out_channels = current_channels // 2
            skip_ch = skip_channels_list[i + 1]
            decoder_blocks.append(
                DecoderBlock(current_channels, out_channels, skip_channels=skip_ch)
            )
            current_channels = out_channels

        decoder_blocks.append(
            DecoderBlock(
                current_channels,
                config.base_channels,
                skip_channels=config.base_channels,
            )
        )
        self.decoder = nn.ModuleList(decoder_blocks)

        self.output_proj = nn.Conv2d(config.base_channels, 3, kernel_size=3, padding=1)

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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.log_var_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.log_var_head.bias)

    def _encode_distribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        x_proj = self.input_proj(x)
        skips: list[torch.Tensor] = [x_proj]

        h = x_proj
        for block in self.encoder:
            h = block(h)
            skips.append(h)

        h = self.bottleneck_attn(h)
        flat = h.flatten(start_dim=1)
        mu = self.mu_head(flat)
        log_var = self.log_var_head(flat)
        return mu, log_var, skips

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self._encode_distribution(x)
        return mu

    def decode_logits(
        self, z: torch.Tensor, skips: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(
            z.shape[0], self.bottleneck_channels, self.encoded_size, self.encoded_size
        )

        reversed_skips = list(reversed(skips)) if skips is not None else None

        for i, block in enumerate(self.decoder):
            skip = None
            if reversed_skips is not None:
                skip_idx = i + 1
                skip = (
                    reversed_skips[skip_idx] if skip_idx < len(reversed_skips) else None
                )
            h = block(h, skip)

        return self.output_proj(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z, skips=None))

    def forward(self, x: torch.Tensor) -> AutoencoderForwardOutput:
        mu, log_var, skips = self._encode_distribution(x)
        z = reparameterize(mu, log_var)
        logits = self.decode_logits(
            z, skips=skips if self.use_skip_connections else None
        )
        return logits, mu, log_var

    def get_config(self) -> dict:
        return self.config.model_dump()

    def get_latent_dim(self) -> int:
        return self.config.latent_dim
