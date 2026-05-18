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
    cspn_cfg = cfg.cspn
    return Einet(
        num_vars=cfg.dataset.latent_size,
        context_dim=cfg.dataset.num_classes,
        num_leaves=cspn_cfg.num_leaves,
        num_nodes=cspn_cfg.num_nodes,
        nn_hidden_dim=cspn_cfg.nn_hidden_dim,
        nn_num_hidden_layers=cspn_cfg.nn_num_hidden_layers,
    ).to(device)
