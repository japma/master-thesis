from dataset_loaders import build_data_loaders
from models.autoencoder import TinyAutoencoderWrapper
from models.autoencoder.pretrained import PretrainedVAE
from utils import resolve_device
from utils.config import load_config
from utils.visualisation import show, show_comparison


def main() -> None:
    cfg, _ = load_config()
    dataset_cfg = cfg.dataset

    dataloader, _ = build_data_loaders(dataset_cfg)

    ae = PretrainedVAE(
        name="xkronosx/AutoEncoder-CelebA-256",
        height=dataset_cfg.height,
        width=dataset_cfg.width,
    )

    device = resolve_device()
    ae.to(device)

    print(f"AE Latent dim {ae.get_latent_dim()}")

    inputs = next(iter(dataloader))[0][:16].to(device)
    print(f"inputs: {inputs.shape}")
    latent = ae.encode(inputs)
    print(f"latent: {latent.shape}")
    recon = ae.decode(latent)
    print(f"recon: {recon.shape}")

    show(recon, title="Reconstructed Images")
    show(inputs, title="Original Images")

    show_comparison(inputs, recon)


if __name__ == "__main__":
    main()
