import torch
from tqdm import tqdm


class LatentNormalizer:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, autoencoder, dataloader, device) -> None:
        latents = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Fitting normalizer"):
                latents.append(autoencoder.encode(images.to(device)))
        latents = torch.cat(latents, dim=0)
        self.mean = latents.mean(dim=0).to(device)
        self.std = latents.std(dim=0).clamp(min=1e-6).to(device)
        print(f"Latent mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
        print(f"Latent std range:  [{self.std.min():.2f}, {self.std.max():.2f}]")

    def normalize(self, z):
        return (z - self.mean) / self.std

    def denormalize(self, z_norm):
        return z_norm * self.std + self.mean
