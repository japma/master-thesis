"""CMMD between two sets of images (https://github.com/google-research/google-research/tree/master/cmmd)"""

import torch
import torch.nn.functional as F
from rtpt import RTPT
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from evaluation.evaluate import GENERATED, REAL, RECONSTRUCTION
from evaluation.halvings import halvings, load_set_pool, write_halvings
from evaluation.samples import to_float
from utils.config import EvaluationRunConfig
from utils.progress import batch_count, start_rtpt
from utils.reproducibility import float64_device

CLIP_MODEL = "openai/clip-vit-large-patch14-336"
CLIP_SIZE = 336
SIGMA = 10.0
SCALE = 1000.0
KERNEL_BLOCK = 4096

FILENAME = "cmmd.csv"


def load_clip(device: torch.device) -> CLIPVisionModelWithProjection:
    return CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).eval().to(device)


@torch.no_grad()
def clip_embeddings(
    model: CLIPVisionModelWithProjection,
    images: torch.Tensor,
    device: torch.device,
    metric_device: torch.device,
    batch_size: int,
    desc: str = "embeddings",
    rtpt: RTPT | None = None,
) -> torch.Tensor:
    """`(N, D)` float64 unit-norm embeddings of `(N, C, H, W)` uint8 images, on
    `metric_device`."""
    mean = torch.tensor(OPENAI_CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD, device=device).view(1, 3, 1, 1)
    chunks = []
    for batch in tqdm(images.split(batch_size), desc=desc):
        x = to_float(batch.to(device)).expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=(CLIP_SIZE, CLIP_SIZE), mode="bicubic").clamp(0, 1)
        embeds = model(pixel_values=(x - mean) / std).image_embeds
        chunks.append(F.normalize(embeds, dim=-1).to(metric_device).double())
        if rtpt is not None:
            rtpt.step(subtitle=desc)
    return torch.cat(chunks)


def kernel_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Gaussian kernel over all pairs, in row blocks so no N x N matrix is held.

    The kernel values all sit near 1 and CMMD is a scaled difference of their means, so
    this runs in float64.
    """
    gamma = 1 / (2 * SIGMA**2)
    b_sq = b.pow(2).sum(1)
    total = 0.0
    for block in a.split(KERNEL_BLOCK):
        sq = block.pow(2).sum(1, keepdim=True) - 2 * block @ b.T + b_sq
        total += torch.exp(-gamma * sq).sum().item()
    return total / (a.shape[0] * b.shape[0])


def cmmd_against(
    reference: torch.Tensor, embedding_sets: dict[str, torch.Tensor]
) -> dict[str, float]:
    """CMMD of each set of embeddings against `reference`, sharing the k(ref, ref) term."""
    k_rr = kernel_mean(reference, reference)
    return {
        name: SCALE * (k_rr + kernel_mean(x, x) - 2 * kernel_mean(x, reference))
        for name, x in embedding_sets.items()
    }


def run_cmmd(cfg: EvaluationRunConfig, device: torch.device) -> None:
    """CMMD of a pool's samples against the real images, averaged over the same random
    halvings of the val split as FID (see `evaluation.halvings`).
    """
    manifest, originals, reconstructions, generated = load_set_pool(cfg)
    batch_size = cfg.evaluation.batch_size

    sets = {REAL: originals, RECONSTRUCTION: reconstructions, GENERATED: generated}
    total = sum(batch_count(images.shape[0], batch_size) for images in sets.values())
    rtpt = start_rtpt(f"cmmd_{manifest.dataset}", total)
    metric_device = float64_device(device)
    model = load_clip(device)
    embedded = {
        source: clip_embeddings(
            model, images, device, metric_device, batch_size, source, rtpt
        )
        for source, images in sets.items()
    }
    del model

    generator = torch.Generator().manual_seed(manifest.seed)
    n, splits = halvings(
        originals.shape[0], generated.shape[0], cfg.evaluation.halvings, generator
    )
    scores = [
        cmmd_against(
            embedded[REAL][reference],
            {
                REAL: embedded[REAL][held_out],
                RECONSTRUCTION: embedded[RECONSTRUCTION][held_out],
                GENERATED: embedded[GENERATED][sampled],
            },
        )
        for reference, held_out, sampled in splits
    ]
    write_halvings(cfg, manifest, FILENAME, "cmmd", scores, n)
