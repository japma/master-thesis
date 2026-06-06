from pathlib import Path
import torch
from models.autoencoder import AbstractAutoencoder, create_autoencoder
from models.cspn import AbstractCSPN


def save_autoencoder(model: AbstractAutoencoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
        },
        path,
    )


def load_autoencoder(path: Path, device=None) -> AbstractAutoencoder:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = create_autoencoder(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model


def save_cspn(model: AbstractCSPN, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
        },
        path,
    )


def load_cspn(path: Path, device=None) -> AbstractCSPN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # model = create_cspn(**ckpt["model_cfg"])
    # model.load_state_dict(ckpt["model_state"])
    # return model
    pass
