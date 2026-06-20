"""Factory for creating autoencoder models."""

from models.autoencoder import TinyAutoencoderWrapper


def load_pretrained_autoencoder(name: str) -> TinyAutoencoderWrapper:
    if name == "madebyollin/taesd":
        return TinyAutoencoderWrapper(name="madebyollin/taesd")
    else:
        raise ValueError
