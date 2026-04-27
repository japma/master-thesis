import torch.nn as nn

from .abstract_encoder import AbstractEncoder


class SimpleEncoder(AbstractEncoder):
    def __init__(self, input_size, latent_size=32):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
            nn.ReLU(),
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        latent = self.network(x)
        return latent
