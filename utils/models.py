from models import VariationalAutoencoder, SPFlowCSPN


def build_autoencoder(cfg, device):
    ae_cfg = cfg.autoencoder
    input_shape = (cfg.dataset.channels, cfg.dataset.height, cfg.dataset.width)

    return VariationalAutoencoder(
        input_shape=input_shape,
        latent_size=cfg.dataset.latent_size,
        base_channels=ae_cfg.base_channels,
        num_blocks=ae_cfg.num_blocks,
        res_blocks=ae_cfg.res_blocks,
    ).to(device)


def build_cspn(cfg, device):
    return SPFlowCSPN(
        latent_dim=cfg.dataset.latent_size,
        num_classes=cfg.dataset.num_classes,
    ).to(device)
