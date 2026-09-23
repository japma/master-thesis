"""The pretrained networks the set metrics compare images in.

Every network maps a `(B, 3, H, W)` uint8 batch to `(B, D)` float features.

    inception  torch-fidelity's port of the TF 2015-12-05 graph, with its TF1 resize --
               FID is defined by that network, and torchvision's inception_v3 differs
    vgg16      torchvision's ImageNet VGG16, fc2 after the ReLU, as torch-fidelity uses
               for precision/recall
    clip       CLIP ViT-L/14@336px projections, L2-normalised, preprocessed as the CMMD
               reference does
    dinov2     DINOv2 ViT-L/14 CLS token, as in Stein et al. (2023)
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
import torchvision
from rtpt import RTPT
from torch_fidelity import FeatureExtractorInceptionV3
from tqdm import tqdm
from transformers import AutoModel, CLIPVisionModelWithProjection
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)

from evaluation.samples import to_float

Network = Callable[[torch.Tensor], torch.Tensor]

CLIP_MODEL = "openai/clip-vit-large-patch14-336"
DINOV2_MODEL = "facebook/dinov2-large"


def normalise(
    images: torch.Tensor, size: int, mode: str, mean: list[float], std: list[float]
) -> torch.Tensor:
    """uint8 images resized to `size` and standardised per channel."""
    x = F.interpolate(to_float(images), size=(size, size), mode=mode).clamp(0, 1)
    mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def load_inception(device: torch.device) -> Network:
    model = FeatureExtractorInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device)
    return lambda images: model(images)[0]


def load_vgg16(device: torch.device) -> Network:
    vgg = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    )
    model = torch.nn.Sequential(
        vgg.features, vgg.avgpool, torch.nn.Flatten(), *vgg.classifier[:5]
    )
    model = model.eval().to(device)
    return lambda images: model(
        normalise(images, 224, "bilinear", IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    )


def load_clip(device: torch.device) -> Network:
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).eval().to(device)

    def embed(images: torch.Tensor) -> torch.Tensor:
        x = normalise(images, 336, "bicubic", OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        return F.normalize(model(pixel_values=x).image_embeds, dim=-1)

    return embed


def load_dinov2(device: torch.device) -> Network:
    model = AutoModel.from_pretrained(DINOV2_MODEL).eval().to(device)
    return lambda images: (
        model(
            pixel_values=normalise(
                images, 224, "bicubic", IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            )
        ).pooler_output
    )


NETWORKS: dict[str, Callable[[torch.device], Network]] = {
    "inception": load_inception,
    "vgg16": load_vgg16,
    "clip": load_clip,
    "dinov2": load_dinov2,
}


@torch.no_grad()
def extract(
    network: Network,
    images: torch.Tensor,
    device: torch.device,
    out_device: torch.device,
    batch_size: int,
    desc: str = "features",
    rtpt: RTPT | None = None,
) -> torch.Tensor:
    """`(N, D)` float32 features of `(N, C, H, W)` uint8 images, on `out_device`."""
    chunks = []
    for batch in tqdm(images.split(batch_size), desc=desc):
        features = network(batch.to(device).expand(-1, 3, -1, -1))
        chunks.append(features.float().to(out_device))
        if rtpt is not None:
            rtpt.step(subtitle=desc)
    return torch.cat(chunks)
