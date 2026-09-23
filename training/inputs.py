"""The autoencoder a latent-space model trains on, loaded the same way by every trainer."""

from dataclasses import dataclass

import torch

from models.autoencoder import AbstractAutoencoder
from models.autoencoder.pretrained import PretrainedVAE
from utils.checkpoints import load_ae_from_path
from utils.config import AutoencoderConfig, DatasetConfig, PretrainedAutoencoderConfig
from utils.naming import VAE_KIND_KEY, pretrained_vae_kind
from utils.wandb_utils import artifact_metadata, download_artifact, record_input


@dataclass
class TrainingAutoencoder:
    model: AbstractAutoencoder
    # The exact `name:vN`, or None for a Hugging Face autoencoder.
    ref: str | None
    kind: str

    def metadata(self) -> dict[str, str | None]:
        """What an artifact trained on this autoencoder records about it."""
        return {"autoencoder": self.ref, VAE_KIND_KEY: self.kind}


def load_training_autoencoder(
    cfg: PretrainedAutoencoderConfig, dataset: DatasetConfig, device: torch.device
) -> TrainingAutoencoder:
    """Consume the autoencoder through the active run, so its lineage records the exact
    version, and record that version in the run config too."""
    if cfg.external:
        model = PretrainedVAE(name=cfg.name, height=dataset.height, width=dataset.width)
        return TrainingAutoencoder(
            model.to(device), None, pretrained_vae_kind(cfg, {}, None)
        )
    path, ref = download_artifact(ckpt_name=cfg.name, tag=cfg.tag)
    record_input("autoencoder_resolved", ref)
    model = load_ae_from_path(path, device=device)
    config = getattr(model, "config", None)
    kind = pretrained_vae_kind(
        cfg,
        artifact_metadata(ref),
        config if isinstance(config, AutoencoderConfig) else None,
    )
    return TrainingAutoencoder(model.to(device), ref, kind)
