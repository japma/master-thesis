"""Combined model wrapping an autoencoder and a CSPN."""

import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self, autoencoder, cspn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cspn = cspn

    def encode(self, images):
        return self.autoencoder.encode(images)

    def decode(self, latents):
        return self.autoencoder.decode(latents)

    def reconstruct(self, images):
        return self.autoencoder(images)

    def predict_latent(self, labels):
        return self.cspn.predict_latent(labels)

    def modify_latent(self, latents, source_labels, target_labels, strength=1.0):
        return self.cspn.transform_latent(
            latents,
            source_labels=source_labels,
            target_labels=target_labels,
            strength=strength,
        )
