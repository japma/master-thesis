"""FID between two streams of images, via torchmetrics."""

from collections.abc import Iterable, Iterator

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from evaluation.protocols import LatentCodec

# torch-fidelity 0.4.0 "inception-v3-compat", sha256 6726825d0af5f729...
INCEPTION_WEIGHTS = "weights-inception-2015-12-05-6726825d.pth"


@torch.no_grad()
def decode_batches(
    vae: LatentCodec,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    desc: str = "decoding",
) -> Iterator[torch.Tensor]:
    for z in tqdm(latents.split(batch_size), desc=desc):
        # FID quantises with (x * 255).byte(), which wraps around above 1.
        yield vae.decode(z.to(device)).clamp(0.0, 1.0)


@torch.no_grad()
def frechet_inception_distance(
    images: Iterable[torch.Tensor],
    reference_images: Iterable[torch.Tensor],
    device: torch.device,
) -> float:
    # The accumulated feature statistics are float64, which MPS does not support.
    metric_device = torch.device("cpu") if device.type == "mps" else device
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(metric_device)
    for batch in reference_images:
        metric.update(batch.to(metric_device), real=True)
    for batch in images:
        metric.update(batch.to(metric_device), real=False)
    return float(metric.compute())
