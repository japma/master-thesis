from torch import nn

from models import VariationalAutoencoder, SPFlowCSPN
from models.autoencoder import AbstractAutoencoder
from models.cspn import AbstractCSPN
from models.cspn.einet import Einet


def build_autoencoder(cfg, device) -> AbstractAutoencoder:
    ae_cfg = cfg.autoencoder
    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

    return VariationalAutoencoder(
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        base_channels=ae_cfg.base_channels,
        num_blocks=ae_cfg.num_blocks,
        res_blocks=ae_cfg.res_blocks,
    ).to(device)


def build_cspn(cfg, device) -> AbstractCSPN:
    return SPFlowCSPN(
        latent_dim=cfg.dataset.latent_size,
        num_classes=cfg.dataset.num_classes,
    ).to(device)


def build_einet_cspn(cfg, device) -> AbstractCSPN:
    return Einet(
        num_vars=cfg.dataset.latent_size,
        # context is label-conditioned; use number of classes as input dim
        context_dim=cfg.dataset.num_classes,
        num_leaves=10,
        num_nodes=5,
    ).to(device)
