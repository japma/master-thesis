import torch

import wandb
from models.autoencoder import AbstractAutoencoder


def log_reconstructions(
    model: AbstractAutoencoder,
    sample_images: torch.Tensor,
    epoch: int,
) -> None:
    """Log reconstructions of fixed validation images."""
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(sample_images)
        recon = torch.sigmoid(logits)
    recon_u8 = (recon.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/recon_images": [wandb.Image(img) for img in recon_u8]},
        step=epoch,
    )


def log_generations(
    model: AbstractAutoencoder,
    device: torch.device,
    epoch: int,
    n: int = 16,
) -> None:
    """Decode random latent vectors as a latent space health diagnostic."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.get_latent_dim(), device=device)
        gen = model.decode(z)
    gen_u8 = (gen.clamp(0, 1) * 255).byte().cpu()
    wandb.log(
        {"samples/generated": [wandb.Image(img) for img in gen_u8]},
        step=epoch,
    )
